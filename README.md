# tafp-vit

### Requirements
torch>=1.4.0
torchvision>=0.5.0
pyyaml
scipy
timm==0.4.5


## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:


```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).


## DeiT

[Repository](https://github.com/facebookresearch/deit)

### DeiT Models
We provide baseline DeiT models pretrained on ImageNet.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 91.1 | 5M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.9 | 95.0 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 95.6 | 86M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |


```
cd deit
```

### Evaluation
Deit-tiny
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --data-path /path/to/imagenet --batch-size 256
```

Deit-small
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model deit_small_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --data-path /path/to/imagenet --batch-size 256
```

Deit-base
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model deit_base_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --data-path /path/to/imagenet --batch-size 256
```

### Training
DeiT-tiny, DeiT-small, DeiT-base
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --model deit_tiny_patch16_224/deit_small_patch16_224/deit_base_patch16_224 --distributed --data-set IMNET/CIFAR10 --data-path /path/to/dataset --output_dir /path/to/save --epochs 30 --batch-size 256
```

### Finetune
DeiT-tiny, DeiT-small, DeiT-base
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --model deit_tiny_patch16_224/deit_small_patch16_224/deit_base_patch16_224 --finetune /path/to/.pth --distributed --data-set IMNET/CIFAR10 --data-path /path/to/dataset --output_dir /path/to/save --batch-size 256 --single-layer-compression-max-epoch 10 --compression-accuracy-drop-threshold 0.5 --feature-map-compssion-en --inter-layer-token-pruning-en --intra-block-row-pruning-en
```

### Evaluation
DeiT-tiny, DeiT-small, DeiT-base
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --model deit_tiny_patch16_224/deit_small_patch16_224/deit_base_patch16_224 --resume /path/to/pth --data-set IMNET/CIFAR10  --data-path /path/to/dataset --batch-size 256 --first-compression-layer-idx 11 --feature-map-compssion-en --inter-layer-token-pruning-en --intra-block-row-pruning-en
```


## LV-ViT

[Repository](https://github.com/zihangJiang/TokenLabeling)

#### LV-ViT Models
We provide baseline LV-ViT models pretrained on ImageNet.

| Model                           | layer | dim  | Image resolution |  Param  | Top 1 |Download |
| :------------------------------ | :---- | :--- | :--------------: |-------: | ----: |   ----: |
| LV-ViT-T                        | 12    | 240  |       224        |  8.53M |  79.1 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/v0.2.0/lvvit_t.pth) |
| LV-ViT-S                        | 16    | 384  |       224        |  26.15M |  83.3 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar) |
| LV-ViT-M                        | 20    | 512  |       224        |  55.83M |  84.0 |[link](https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_m-56M-224-84.0.pth.tar) |


```
cd deit
```
#### Label data
We provide NFNet-F6 generated dense label map in [Google Drive](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view?usp=sharing) and [BaiDu Yun](https://pan.baidu.com/s/1YBqiNN9dAzhEXtPl61bZJw) (password: y6j2). As NFNet-F6 are based on pure ImageNet data, no extra training data is involved.

#### Evaluation
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --amp --data_dir /path/to/imagenet --model lvvit_t/lvvit_s/lvvit_m --resume /path/to/pth.tar -b 256
```

#### Training
LV-ViT-T, LV-ViT-S, LV-ViT-M
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --apex-amp --token-label --token-label-data /path/to/label_data --token-label-size 14 --drop-path 0.1 --model-ema --dataset IMNET --data_dir /path/to/imagenet --model lvvit_t/lvvit_s/lvvit_m --epochs 100 -b 128
```

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --apex-amp  --drop-path 0.1 --model-ema --dataset CIFAR10 --data_dir /path/to/cifar --model lvvit_t/lvvit_s/lvvit_m --epochs 100 -b 128
```

#### Fine-tuning
LV-ViT-T, LV-ViT-S, LV-ViT-M
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --apex-amp --drop-path 0.1 --model-ema --dataset IMNET --data_dir /path/to/imagenet --model lvvit_t/lvvit_s/lvvit_m --finetune /path/to/pth.tar -b 128 --single-layer-compression-max-epoch 10 --compression-accuracy-drop-threshold 0.5 --feature-map-compssion-en --inter-layer-token-pruning-en --intra-block-row-pruning-en
```

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=54321 main.py --apex-amp  --drop-path 0.1 --model-ema --dataset CIFAR10 --data_dir /path/to/cifar --model lvvit_t/lvvit_s/lvvit_m -b 128 --single-layer-compression-max-epoch 10 --compression-accuracy-drop-threshold 0.5 --feature-map-compssion-en --inter-layer-token-pruning-en --intra-block-row-pruning-en
```

### Evaluation
LV-ViT-T, LV-ViT-S, LV-ViT-M
```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --amp --dataset IMNET/CIFAR10 --data_dir /path/to/dataset --model lvvit_t/lvvit_s/lvvit_m --resume /path/to/pth.tar -b 256 --first-compression-layer-idx 11 --feature-map-compssion-en --inter-layer-token-pruning-en --intra-block-row-pruning-en
```