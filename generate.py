import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from PIL import Image

import magref
from magref.configs import MAGREF_CONFIGS
from magref.utils.utils import cache_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt and reference images using MagRef"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample from a video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
       "--save_dir",
        type=str,
        default=None,
        help="The path to the save video directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to a JSON file containing prompts for video generation.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the directory containing reference images. Each sample's images should be in a subfolder named by its index.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=3407,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=40, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=3.0,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
            force=True)
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = MAGREF_CONFIGS['magref-14B']
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    
    # Read prompts from JSON file
    prompt_path = args.prompt_path
    with open(prompt_path, 'r') as f:
        all_prompts = json.load(f)


    # Collect all (key, prompt, img_paths) tuples
    all_pairs = []
    for idx, item in enumerate(all_prompts):
        key = idx
        prompt = item["prompt_rewritten"]
        # Look for reference images in image_dir/<idx>/
        img_paths = []
        for refs in item["refs"]:
            path = refs["image_path"].replace("/edrive1/kaiq/VER-bench/data/first3k/unknown_first3k", args.image_dir)
            img_paths.append(path)

        all_pairs.append((key, prompt, img_paths))

    # Shard by rank: each rank processes a subset of prompts
    prompt_image_pairs = all_pairs[rank::world_size]
    logging.info(f"Rank {rank}/{world_size}: processing {len(prompt_image_pairs)}/{len(all_pairs)} samples.")

    logging.info("Creating MagRef pipeline.")
    magref_model = magref.MagRefModel(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    logging.info("Generating video ...")
    for key, prompt, img_paths in prompt_image_pairs:
        pil_img = [Image.open(p).convert("RGB") for p in img_paths]

        video = magref_model.generate(
            prompt,
            pil_img,
            max_area=480 * 832,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        

        # Skip the first 4 frames (which are sampled from the vae decoder first frame used as reference condition) to avoid visual blur
        video = video[:, 4:, :, :]
        os.makedirs(args.save_dir, exist_ok=True)
        save_file = os.path.join(args.save_dir, f"{key:05d}.mp4")
        logging.info(f"Rank {rank}: Saving generated video to {save_file}")
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logging.info(f"Rank {rank}: Finished {key}.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
