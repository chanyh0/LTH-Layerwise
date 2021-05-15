CUDA_VISIBLE_DEVICES=2 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type rewind_lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_rewind_lt_1 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_rewind_lt_5_GPU2.out

CUDA_VISIBLE_DEVICES=2 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type rewind_lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.4_w20_rewind_lt_2 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.4_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.4_w20_rewind_lt_5_GPU2.out

