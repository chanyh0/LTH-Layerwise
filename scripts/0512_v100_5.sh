CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.075_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.1_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.125_w15_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.15_w15_GPU3.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.05_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.2 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.2_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.2_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet1 --pruning_times 1 --lr 0.4 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet1_cifar100_b64_e160_lr0.4_w15 --rewind_epoch 8 > 0512_mobilenet1_cifar100_b64_e160_lr0.4_w15_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet2_cifar100_b64_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar100_b64_e160_lr0.05_w15_GPU3.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet2_cifar100_b64_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar100_b64_e160_lr0.075_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet2_cifar100_b64_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar100_b64_e160_lr0.1_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet2_cifar100_b64_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar100_b64_e160_lr0.125_w15_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet2 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet2_cifar100_b64_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet2_cifar100_b64_e160_lr0.15_w15_GPU3.out &


