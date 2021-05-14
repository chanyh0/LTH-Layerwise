CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.05_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.075_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.1_w15_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.125_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.15_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.2 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.2_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.2_w15_GPU2.out &

##### 18:27

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet5 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet5_cifar10_b128_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet5_cifar10_b128_e160_lr0.15_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet1_cifar10_b128_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar10_b128_e160_lr0.1_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.15 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.15_w15_seed2 --rewind_epoch 8 --rate 0.5904 > 0512_mobilenet_cifar10_b128_e160_lr0.15_w15seed2-_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15_seed2 --rewind_epoch 8 --rate 0.5904 > 0512_mobilenet_cifar10_b128_e160_lr0.1_w15_seed2-GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.2_w15_seed2 --rewind_epoch 8 --rate 0.5904 > 0512_mobilenet_cifar10_b128_e160_lr0.2_w15_seed2-GPU2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.4 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.4_w15_seed2 --rewind_epoch 8 --rate 0.5904 > 0512_mobilenet_cifar10_b128_e160_lr0.4_w15_seed2-GPU2.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.15_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.1_w15_GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.2_w15_seed2 --rewind_epoch 8 --rate 0.8323 > 0512_mobilenet_cifar10_b128_e160_lr0.2_w15seed2-_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.1_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar10_b128_e160_lr0.1_w15_seed2-GPU5.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.125_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar10_b128_e160_lr0.125_w15_seed2-GPU6.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet_cifar10_b128_e160_lr0.2_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar10_b128_e160_lr0.2_w15_seed2-GPU6.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.125_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.075_w15_GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet2_cifar10_b128_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar10_b128_e160_lr0.125_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet2_cifar10_b128_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar10_b128_e160_lr0.075_w15_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.4 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.4_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.4_w15_seed2-GPU6.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.2_w15_seed2-GPU6.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.05_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet3 --pruning_times 1 --lr 0.2 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet3_cifar10_b128_e160_lr0.2_w15 --rewind_epoch 8 > 0512_mobilenet3_cifar10_b128_e160_lr0.2_w15_GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet2_cifar10_b128_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar10_b128_e160_lr0.05_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar10 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.2 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 128 --save_dir  mobilenet2_cifar10_b128_e160_lr0.2_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar10_b128_e160_lr0.2_w15_GPU5.out &
