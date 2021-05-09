CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.10_w20_IMP_rewind_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.075_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.075_w20_IMP_rewind_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20  \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.1_w20_IMP_rewind_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.125_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.125_w20_IMP_rewind_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.15_w20_IMP_rewind_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.2_w20_IMP_rewind_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.4_w20 \
  > 0504_res18_tiny-imagenet_b32_e160_lr0.4_w20_IMP_rewind_GPU6.out &

