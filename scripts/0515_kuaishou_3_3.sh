CUDA_VISIBLE_DEVICES=4 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_lt_1 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.15_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.15_w20_lt_1_GPU4.out

CUDA_VISIBLE_DEVICES=4 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.125_w20_lt_1 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.125_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.125_w20_lt_1_GPU4.out