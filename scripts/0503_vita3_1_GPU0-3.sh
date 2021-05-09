CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type rewind_lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15 > logs_0503/0503_mobilenet_cifar10_b128_e160_lr0.1_w15_OMP_1_GPU0.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8323 --prune_type rewind_lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15 > logs_0503/0503_mobilenet_cifar10_b128_e160_lr0.1_w15_OMP_2_GPU6.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type rewind_lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15 > logs_0503/0503_mobilenet_cifar10_b128_e160_lr0.1_w15_OMP_3_GPU4.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset tiny-imagenet --seed 1 --arch mobilenet --pruning_times 2 --rate 0.5904 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 32 --save_dir  mobilenet_tiny-imagenet_b32_e160_lr0.1_w15 > logs_0503/0503_mobilenet_tiny-imagenet_b32_e160_lr0.1_w15_OMP_1_GPU0.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset tiny-imagenet --seed 1 --arch mobilenet --pruning_times 2 --rate 0.8323 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 32 --save_dir  mobilenet_tiny-imagenet_b32_e160_lr0.1_w15 > logs_0503/0503_mobilenet_tiny-imagenet_b32_e160_lr0.1_w15_OMP_2_GPU6.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset tiny-imagenet --seed 1 --arch mobilenet --pruning_times 2 --rate 0.9141 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 32 --save_dir  mobilenet_tiny-imagenet_b32_e160_lr0.1_w15 > logs_0503/0503_mobilenet_tiny-imagenet_b32_e160_lr0.1_w15_OMP_3_GPU4.out &
