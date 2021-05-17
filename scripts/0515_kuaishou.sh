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

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_4 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_4_tiny-imagenet_b32_e160_lr0.2 \
  > 0514_res18_4_tiny-imagenet_b32_e160_lr0.2_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_4 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_4_tiny-imagenet_b32_e160_lr0.4 \
  > 0514_res18_4_tiny-imagenet_b32_e160_lr0.4_GPU6.out &


  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_6 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_6_tiny-imagenet_b32_e160_lr0.4 \
  > 0514_res18_6_tiny-imagenet_b32_e160_lr0.4_GPU1.out &

   CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_5 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_5_tiny-imagenet_b32_e160_lr0.4 \
  > 0514_res18_5_tiny-imagenet_b32_e160_lr0.4_GPU0.out &


  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_5 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_5_tiny-imagenet_b32_e160_lr0.15 \
  > 0514_res18_5_tiny-imagenet_b32_e160_lr0.15_GPU7.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_5 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_5_tiny-imagenet_b32_e160_lr0.2 \
  > 0514_res18_5_tiny-imagenet_b32_e160_lr0.2_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_6 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_6_tiny-imagenet_b32_e160_lr0.2 \
  > 0514_res18_6_tiny-imagenet_b32_e160_lr0.2_GPU6.out &


   CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_6 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_6_tiny-imagenet_b32_e160_lr0.1 \
  > 0514_res18_6_tiny-imagenet_b32_e160_lr0.1_GPU1.out &

   CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_5 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_5_tiny-imagenet_b32_e160_lr0.1 \
  > 0514_res18_5_tiny-imagenet_b32_e160_lr0.1_GPU0.out &

     CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_4 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_4_tiny-imagenet_b32_e160_lr0.1 \
  > 0514_res18_4_tiny-imagenet_b32_e160_lr0.1_GPU3.out &


  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_6 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_6_tiny-imagenet_b32_e160_lr0.125 \
  > 0514_res18_6_tiny-imagenet_b32_e160_lr0.125_GPU7.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_5 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_5_tiny-imagenet_b32_e160_lr0.125 \
  > 0514_res18_5_tiny-imagenet_b32_e160_lr0.125_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18_6 --pruning_times 1 --prune_type lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res18_6_tiny-imagenet_b32_e160_lr0.15 \
  > 0514_res18_6_tiny-imagenet_b32_e160_lr0.15_GPU6.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.2_w20_OMP_1_GPU2.out &