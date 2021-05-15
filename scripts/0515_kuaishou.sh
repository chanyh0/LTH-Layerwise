CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_2 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_2_tiny-imagenet_b32_e160_lr0.15 \
  > 0514_res18_2_tiny-imagenet_b32_e160_lr0.15_GPU7.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_2 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_2_tiny-imagenet_b32_e160_lr0.2 \
  > 0514_res18_2_tiny-imagenet_b32_e160_lr0.2_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_2 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_2_tiny-imagenet_b32_e160_lr0.4 \
  > 0514_res18_2_tiny-imagenet_b32_e160_lr0.4_GPU6.out &


  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_3 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_3_tiny-imagenet_b32_e160_lr0.15 \
  > 0514_res18_3_tiny-imagenet_b32_e160_lr0.15_GPU7.out &

  CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_3 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_3_tiny-imagenet_b32_e160_lr0.2 \
  > 0514_res18_3_tiny-imagenet_b32_e160_lr0.2_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_3 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_3_tiny-imagenet_b32_e160_lr0.4 \
  > 0514_res18_3_tiny-imagenet_b32_e160_lr0.4_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_4 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_4_tiny-imagenet_b32_e160_lr0.15 \
  > 0514_res18_4_tiny-imagenet_b32_e160_lr0.15_GPU7.out &