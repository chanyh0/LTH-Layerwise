CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type seed2 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_1 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_1_GPU5.out

CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.8323 --prune_type seed2 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_2 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_2_GPU5.out

CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9141 --prune_type seed2 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_3 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_3_GPU5.out

CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9450 --prune_type seed2 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_4 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_4_GPU5.out

CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type seed2 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_5 \
  --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.2_w20_omp/0checkpoint.pth.tar \
  --rewind_epoch 8 \
  > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_reinit_5_GPU5.out