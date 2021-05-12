CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.15_w15_OMP_reinit_4_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.125 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.125_w15_OMP_reinit_4_GPU1.out &




CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_reinit_1_GPU2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8322 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_reinit_2_GPU3.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.05_w15_OMP_reinit_4_GPU1.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_reinit_3_GPU0.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9450 --prune_type seed2 --rewind_epoch 8 --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_omp_reinit > 0510_mobilenet_cifar100_b64_e160_lr0.075_w15_OMP_reinit_4_GPU1.out &