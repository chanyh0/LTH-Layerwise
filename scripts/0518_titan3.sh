CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_extreme --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0518_resnet18_cifar10_lt_extreme_GPU1.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_extreme_qrcode2 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir resnet18_cifar10_qrcode2.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0518_resnet18_cifar10_lt_extreme_qrcode2_GPU1.out &

CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir test --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --evaluate --checkpoint resnet18_cifar10_lt_extreme_qrcode2/model_SA_best.pt 

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir resnet50_cifar100_lt_0.1 --init LotteryTickets/cifar100_LT/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --resume --checkpoint LotteryTickets/cifar100_LT/2_checkpoint.pt --prune_type lt > 0527_resnet50_cifar100_lt_0.1_GPU7.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --batch_size 256 --save_dir resnet50_cifar100_lt_0.2 --init LotteryTickets/cifar100_LT/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0518_resnet50_cifar100_rewind_lt_0.2_GPU2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --batch_size 256 --save_dir resnet50_cifar10_lt_0.2 --init LotteryTickets/cifar10_LT/random_init.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0518_resnet50_cifar10_rewind_lt_0.2_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data datasets/cifar10 --dataset cifar10 --arch res50 --batch_size 256 --save_dir resnet50_cifar10_lt_extreme --pretrained resnet50_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --lr 0.1 --seed 7 --save_model --prune-type lt > 0517_resnet50_cifar10_lt_extreme_GPU1.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --batch_size 256 --save_dir resnet50_cifar100_lt_extreme --pretrained resnet50_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir LotteryTickets/cifar100_LT/2_checkpoint.pt --fc --lr 0.1 --seed 7 --save_model --prune-type lt --type identity > 0517_resnet50_cifar100_lt_extreme_GPU1.out &



CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data datasets/cifar10 --dataset cifar10 --arch res50 --batch_size 256 --save_dir resnet50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --lr 0.1 --seed 7 --weight_decay 5e-4 --warmup 2 --rewind_epoch 2 > 0517_resnet50_cifar10_lt_extreme_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data datasets/cifar100 --dataset cifar100 --arch res50 --batch_size 256 --save_dir resnet50_cifar100_lt_extreme --pretrained LotteryTickets/cifar100_LT/random_init.pt --lr 0.1 --seed 7 --weight_decay 5e-4 --warmup 2 --rewind_epoch 2 > 0517_resnet50_cifar100_lt_extreme_GPU1.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir test --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --evaluate --checkpoint resnet18_cifar10_lt_extreme_qrcode2/model_SA_best.pt 
