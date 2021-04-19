
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.05 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.05_w10 --warmup 10 > 0418_mobilenet_cifar10_b256_e160_lr0.05_w10_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.125 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.125_w10 --warmup 10 > 0418_mobilenet_cifar10_b256_e160_lr0.125_w10_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.2 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.2_w10 --warmup 10 > 0418_mobilenet_cifar10_b256_e160_lr0.2_w10_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.25 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.25_w10 --warmup 10 > 0418_mobilenet_cifar10_b256_e160_lr0.25_w10_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.3 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.3_w10 --warmup 10 > 0418_mobilenet_cifar10_b256_e160_lr0.3_w10_GPU2.out &