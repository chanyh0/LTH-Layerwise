CUDA_VISIBLE_DEVICES=4 python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.05_w15 > logs_0427/0427_mobilenet_cifar100_b64_e160_lr0.05_w15_IMP_GPU4.out &

CUDA_VISIBLE_DEVICES=5 python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15 > logs_0427/0427_mobilenet_cifar100_b64_e160_lr0.075_w15_IMP_GPU5.out &

CUDA_VISIBLE_DEVICES=6 python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15 > logs_0427/0427_mobilenet_cifar100_b64_e160_lr0.1_w15_IMP_GPU6.out &




