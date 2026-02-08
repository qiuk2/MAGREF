### Download MAGREF Checkpoint

```bash

# if you are in china mainland, run this first: export HF_ENDPOINT=https://hf-mirror.com
# pip install -U "huggingface_hub[cli]"
huggingface-cli download MAGREF-Video/MAGREF --local-dir ./ckpts/magref

```

### Running Code

```
torchrun --nproc_per_node=8 --nnodes=xxx --node_rank=xx \
    --master_addr=<主节点IP> --master_port=29500 \
    generate.py \
    --ckpt_dir ./ckpts/magref/ \
    --prompt_path final.json \
    --offload_model True \
    --image_dir /edrive1/kaiq/VER-bench/data/first3k/unknown_first3k/ \
    --save_dir /path/to/results
```