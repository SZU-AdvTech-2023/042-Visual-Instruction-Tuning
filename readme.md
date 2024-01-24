## VQABBoxpro微调

首先，你需要准备数据集。

在train_configs/vqabbboxpro_finetune.yaml中，你需要设置以下路径：

llama_model检查点路径："/path/to/llama_checkpoint"

ckpt："/path/to/pretrained_checkpoint"

ckpt保存路径："/path/to/save_checkpoint"

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/vqabbboxpro_finetune.yaml

```