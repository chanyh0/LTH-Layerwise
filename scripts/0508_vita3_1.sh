CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.05_w20 \
    --resume --checkpoint $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.05_w20/3checkpoint.pth.tar \
  > $SCRATCH/0504_res50_tiny-imagenet_b32_e160_lr0.10_w20_IMP_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.075_w20 \
  --resume --checkpoint $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.075_w20/3checkpoint.pth.tar \
  > $SCRATCH/0504_res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP_GPU1.out &