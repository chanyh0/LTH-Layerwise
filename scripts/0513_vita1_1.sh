CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.125_w15_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.15 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.15_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.15_w15_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.2 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.2_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.2_w15_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.4 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.4_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.4_w15_GPU3.out &



CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.05 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.05_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.05_w15_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.075 --prune_type lt --epoch 160 --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.075_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.075_w15_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet4 --pruning_times 1 --lr 0.1 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet4_cifar100_b64_e160_lr0.1_w15 --rewind_epoch 8 > 0512_mobilenet4_cifar100_b64_e160_lr0.1_w15_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp.py --data ../data --dataset cifar100 --seed 1 --arch mobilenet5 --pruning_times 1 --lr 0.125 --prune_type lt --epoch 160  --decreasing_lr 80,120  --warmup 15  --batch_size 64 --save_dir  mobilenet5_cifar100_b64_e160_lr0.125_w15 --rewind_epoch 8 > 0512_mobilenet5_cifar100_b64_e160_lr0.125_w15_GPU7.out &


