CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.125_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.125_w20_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.15_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.15_w20_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.2_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.2_w20_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.4_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.4_w20_GPU6.out &


  CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.1_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.1_w20_GPU3.out &

    CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.075_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.075_w20_GPU2.out &

  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_2 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_2_tiny-imagenet_b32_e160_lr0.05_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_2_tiny-imagenet_b32_e160_lr0.05_w20_GPU1.out &


  


  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.125_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.125_w20_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.15_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.15_w20_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.2_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.2_w20_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.4_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.4_w20_GPU6.out &


  CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.1_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.1_w20_GPU3.out &

    CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.075_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.075_w20_GPU2.out &

  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_3 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_3_tiny-imagenet_b32_e160_lr0.05_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_3_tiny-imagenet_b32_e160_lr0.05_w20_GPU1.out &






  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.125_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.125_w20_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.15_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.15_w20_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.2_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.2_w20_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.4_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.4_w20_GPU6.out &


  CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.1_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.1_w20_GPU3.out &

    CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.075_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.075_w20_GPU2.out &

  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_4 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_4_tiny-imagenet_b32_e160_lr0.05_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_4_tiny-imagenet_b32_e160_lr0.05_w20_GPU1.out &


  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.125_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.125_w20_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.15_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.15_w20_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.2_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.2_w20_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.4_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.4_w20_GPU6.out &


  CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.1_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.1_w20_GPU3.out &

    CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.075_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.075_w20_GPU2.out &

  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_5 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_5_tiny-imagenet_b32_e160_lr0.05_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_5_tiny-imagenet_b32_e160_lr0.05_w20_GPU1.out &



  CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.125_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.125_w20_GPU7.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.15_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.15_w20_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.2_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.2_w20_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.4_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.4_w20_GPU6.out &


  CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.1_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.1_w20_GPU3.out &

    CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.075_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.075_w20_GPU2.out &

  CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50_6 --pruning_times 1 --rate 0.2 --prune_type lt  --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_6_tiny-imagenet_b32_e160_lr0.05_w20_IMP \
  --rewind_epoch 8 \
  > 0601_res50_6_tiny-imagenet_b32_e160_lr0.05_w20_GPU1.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w10_OMP_5_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w10_OMP_4_GPU6.out &
 
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w10_OMP_3_GPU5.out &
  
CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w10_OMP_2_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w10_OMP_1_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125_w10_OMP_1_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_1_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_OMP_1_GPU0.out &



CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_OMP_5_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_OMP_4_GPU6.out &
 
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_OMP_3_GPU5.out &
  
CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_OMP_2_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_5_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_4_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_3_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_2_GPU0.out &




CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w20_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125_w20_OMP_5_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w20_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125_w20_OMP_4_GPU6.out &
 
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w20_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125_w20_OMP_3_GPU5.out &
  
CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w20_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125_w20_OMP_2_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w20_omp_rewind --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w20_OMP_rewind_5_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w20_omp_rewind --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w20_OMP_rewind_4_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w20_omp_rewind --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w20_OMP_rewind_3_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.4_w20_omp_rewind --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.4_w20/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.4_w20_OMP_rewind_2_GPU0.out &






CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_seed2_OMP_5_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_seed2_OMP_4_GPU6.out &
 
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_seed2_OMP_3_GPU5.out &
  
CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.2_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.2_w10_seed2_OMP_2_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_seed2_OMP_5_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_seed2_OMP_4_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_seed2_OMP_3_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.15_w10_seed2_OMP_2_GPU0.out &



CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125w10_seed2_OMP_5_GPU7.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125w10_seed2_OMP_4_GPU6.out &
 
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125w10_seed2_OMP_3_GPU5.out &
  
CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.125w10_seed2_OMP_2_GPU4.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9560 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.075_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.075_w10_seed2_OMP_5_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.075_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.075_w10_seed2_OMP_4_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.075_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.075_w10_seed2_OMP_3_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.075_w10_omp --resume --checkpoint res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP/0checkpoint.pth.tar > 0602_res50_tiny-imagenet_b32_e160_lr0.075_w10_seed2_OMP_2_GPU0.out &