CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.4 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.4_w0 --warmup 0 > 0424_mobilenet_cifar100_b64_e160_lr0.4_w0_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 32 --lr 0.075 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b32_e160_lr0.075_w0 --warmup 0 > 0424_res18_tiny-imagenet_b32_e160_lr0.075_w0_GPU1.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.125 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.125_w10_imp0 --warmup 10 > 0424_mobilenet_cifar10_b128_e160_lr0.125_w10_imp0_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.075 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.075_w0_imp0 --warmup 0 > 0424_mobilenet_cifar100_b64_e160_lr0.075_w0_imp0_GPU1.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.1 --decreasing_lr 80,120 --save_dir mobilenet_cifar10_b128_e160_lr0.1_w10_imp0 --warmup 10 > 0424_mobilenet_cifar10_b128_e160_lr0.1_w10_imp0_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.075 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.075_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.075_w0_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.1 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.1_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.1_w0_imp0_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.125 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.125_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.125_w0_imp0_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.15 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.15_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.15_w0_imp0_GPU3.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.2 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.2_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.2_w0_imp0_GPU5.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.01 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.01_w0_imp0 --warmup 0 > 0425_mobilenet_cifar100_b64_e160_lr0.01_w0_imp0_GPU7.out &




CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.075 --decreasing_lr 80,120 --save_dir mobilenet_cifar100_b64_e160_lr0.075_w10 --warmup 10 > 0425_mobilenet_cifar100_b64_e160_lr0.075_w10_GPU1.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res50 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.1 --decreasing_lr 80,120 --save_dir res50_tiny_imagenet_b64_e160_lr0.1_w0 --warmup 0 > 0425_res50_tiny_imagenet_b64_e160_lr0.075_w0_GPU1.out &

