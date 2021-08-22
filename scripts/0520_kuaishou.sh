CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.01 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 25 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar10_lt_0.05/14checkpoint.pth.tar > 0519_res20s_cifar10_rewind_lt_0.01_GPU3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.2 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0519_res20s_cifar100_rewind_lt_0.2_GPU3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.1 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar100_lt_0.2/6checkpoint.pth.tar > 0519_res20s_cifar100_rewind_lt_0.1_GPU3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.05 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar100_lt_0.2/6checkpoint.pth.tar > 0519_res20s_cifar100_rewind_lt_0.05_GPU3.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_0.2 --batch_size 256 --init LotteryTickets/cifar10_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 --warmup 5 --weight_decay 2e-4 > 0521_res50_cifar10_rewind_lt_0.2_GPU1.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.1 --init pretrained_model/resnet18_lt_cifar100.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 27 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res18_cifar100_lt_0.2_continue/17model_SA_best.pth.tar > 0521_res18_cifar100_rewind_lt_0.1_GPU7.out &



CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar10_lt_0.1 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 30 --prune_type lt --resume --checkpoint res18_cifar100_lt_0.2_continue/18checkpoint.pth.tar  > 0521_res18_cifar10_rewind_lt_0.1_GPU5.out &



CUDA_VISIBLE_DEVICES=6 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir res50_cifar100_lt_0.2-2e-4 --init LotteryTickets/cifar100_LT/random_init.pt --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 --weight_decay 2e-4 --batch_size 256 --warmup 5 > 0521_res50_cifar100_rewind_lt_0.2_GPU6.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_0.01 --init pretrained_model/res20s_cifar100_lt.pth.tar --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 25 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar100_lt_0.05/10checkpoint.pth.tar > 0519_res20s_cifar100_rewind_lt_0.01_GPU3.out &


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.01/15model_SA_best.pth.tar --evaluate-p 0.01

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/15model_SA_best.pth.tar --fc --save_model --seed 1 > 0517_res20s_cifar10_extreme_trigger_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_GPU2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_GPU2.out &



CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar100_lt_0.01/17model_SA_best.pth.tar 
