CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.125_w10_OMP_1_GPU2.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_1_GPU2.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.125_w10_OMP_4_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.9450 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_4_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.125_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.125_w10_OMP_2_GPU0.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 2 --rate 0.8323 --prune_type rewind_lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 10  --batch_size 32 --save_dir  res50_tiny-imagenet_b32_e160_lr0.15_w10_omp > 0510_res50_tiny-imagenet_b32_e160_lr0.15_w10_OMP_2_GPU0.out &




CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_4_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_4_GPU1.out &