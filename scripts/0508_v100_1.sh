CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 10  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.05_w10 \
  --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.05_w10/0checkpoint.pth.tar \
  > 0508_res50_tiny-imagenet_b32_e160_lr0.05_w10_IMP_rewind_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 10  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.075_w10 \
    --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.075_w10/0checkpoint.pth.tar \
  > 0508_res50_tiny-imagenet_b32_e160_lr0.075_w10_IMP_rewind_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 10  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.1_w10  \
     --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.1_w10/0checkpoint.pth.tar \
  > 0508_res50_tiny-imagenet_b32_e160_lr0.1_w10_IMP_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 10  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.125_w10 \
  --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w10/0checkpoint.pth.tar \
  > 0508_res50_tiny-imagenet_b32_e160_lr0.125_w10_IMP_rewind_GPU3.out &
