CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.05_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_OMP_3_rewind_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.075_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.075_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.075_w20_OMP_3_rewind_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.05_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_OMP_4_rewind_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.075_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.075_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.075_w20_OMP_4_rewind_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.05_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_OMP_5_rewind_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.9560 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.075_w20_omp_2_rewind --resume --checkpoint res18_tiny-imagenet_b32_e160_lr0.075_w20_omp/0checkpoint.pth.tar > 0512_res18_tiny-imagenet_b32_e160_lr0.075_w20_OMP_5_rewind_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.05_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.05_w20_OMP_1_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.075_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.075_w20_OMP_1_GPU0.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.4 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.4_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.4_w20_OMP_1_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.125_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.125_w20_OMP_1_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 2 --rate 0.5904 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.125_w20_omp_1 > 0512_res18_tiny-imagenet_b32_e160_lr0.125_w20_OMP_1_GPU0.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir  res18_tiny-imagenet_b32_e160_lr0.125_w20_imp > 0512_res18_tiny-imagenet_b32_e160_lr0.125_w20_imp_GPU0.out &


