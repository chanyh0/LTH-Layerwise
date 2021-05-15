CUDA_VISIBLE_DEVICES=3 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9450 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_lt_4 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.05_w20_omp/0checkpoint.pth.tar \
  --epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_lt_4_GPU3.out

CUDA_VISIBLE_DEVICES=3 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_lt_5 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.05_w20_omp/0checkpoint.pth.tar \
  --epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_lt_5_GPU3.out