CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt-GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.1_w15_seed2-GPU4.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_seed2-GPU4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.15 --prune_type seed2 --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_seed2 --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.15_w15_seed2-GPU6.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.15 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.15_w15_rewind_lt --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.15_w15_rewind_lt-GPU6.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 15 --rate 0.2 --prune_type lt --rewind_epoch 8 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 20  --batch_size 32 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp > 0512_res18_tiny-imagenet_b32_e160_lr0.15_w20_imp_GPU5.out &




CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt-GPU4_OMP_5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt --rewind_epoch 8 --rate 0.9450 > 0512_mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt-GPU4_OMP_4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt --rewind_epoch 8 --rate 0.9141 > 0512_mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt-GPU6_OMP_3.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.2 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt --rewind_epoch 8 --rate 0.8323 > 0512_mobilenet_cifar100_b64_e160_lr0.2_w15_rewind_lt-GPU6_OMP_2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt-GPU5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt --rewind_epoch 8 --rate 0.9560 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt-GPU4_OMP_5.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt --rewind_epoch 8 --rate 0.9450 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt-GPU4_OMP_4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt --rewind_epoch 8 --rate 0.9141 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt-GPU6_OMP_3.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.125 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt --rewind_epoch 8 --rate 0.8323 > 0512_mobilenet_cifar100_b64_e160_lr0.125_w15_rewind_lt-GPU6_OMP_2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt --rewind_epoch 8 --rate 0.9450 > 0512_mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt_4-GPU5.out &





CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt --rewind_epoch 8 --rate 0.9141 > 0512_mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt-GPU6_OMP_3.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.1 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt --rewind_epoch 8 --rate 0.8323 > 0512_mobilenet_cifar100_b64_e160_lr0.1_w15_rewind_lt-GPU6_OMP_2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet --pruning_times 2 --lr 0.075 --prune_type rewind_lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet_cifar100_b64_e160_lr0.075_w15_rewind_lt --rewind_epoch 8 --rate 0.9450 > 0512_mobilenet_cifar100_b64_e160_lr0.075_w15_rewind_lt_4-GPU5.out &