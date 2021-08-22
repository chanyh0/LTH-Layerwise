CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_imp/4checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_4.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_imp/8checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_8.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_imp/11checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_11.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_imp/14checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_14.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.2 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.2_w20_imp/13checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.2_w20_omp_13.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp/4checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.15_w20_omp_4.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp/8checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.15_w20_omp_8.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp/11checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.15_w20_omp_11.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp/14checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.15_w20_omp_14.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.15 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.15_w20_imp/13checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.15_w20_omp_13.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --batch_size 256 --save_dir resnet50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --lr 0.1 --seed 7 --save_model --prune-type lt --type identity > 0517_resnet50_cifar10_lt_extreme_GPU4.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_imp/4checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.1_w20_omp_4.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_imp/8checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.1_w20_omp_8.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_imp/11checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.1_w20_omp_11.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_imp/14checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.1_w20_omp_14.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --epoch 160 --lr 0.1 --decreasing_lr 80,120  --warmup 15 --batch_size 64 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_omp --pretrained init/res18_tiny-imagenet_2.pth.tar --mask_dir res18_tiny-imagenet_b32_e160_lr0.1_w20_imp/13checkpoint.pth.tar > 0518_res18_tiny-imagenet_b32_e160_lr0.1_w20_omp_13.out &


CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --batch_size 256 --save_dir resnet50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --lr 0.1 --seed 7 --save_model --prune-type lt --type identity > 0517_resnet50_cifar10_lt_extreme_GPU4.out &





CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.2 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0518_res18_cifar10_rewind_lt_0.2_GPU2.out &





CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --batch_size 256 --save_dir res18_cifar1000_lt_0.1 --init pretrained_model/resnet18_lt_cifar100.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0518_res18_cifar100_rewind_lt_0.2_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir res50_cifar100_lt_0.2_128 --init LotteryTickets/cifar100_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0519_res50_cifar100_rewind_lt_0.2_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_0.2_128 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0519_res50_cifar10_rewind_lt_0.2_GPU1.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.2_128 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0519_res20s_cifar10_rewind_lt_0.2_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_0.2-2e-4 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 --weight_decay 2e-4 --batch_size 256 > 0519_res50_cifar10_rewind_lt_0.2_GPU4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir res50_cifar100_lt_0.2-2e-4 --init LotteryTickets/cifar100_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 --weight_decay 2e-4 --batch_size 256 --warmup 5 > 0519_res50_cifar100_rewind_lt_0.2_GPU6.out &

