
CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.2_continue --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 20 --prune_type lt --resume --checkpoint res18_cifar100_lt_0.2/9checkpoint.pth.tar  > 0519_res18_cifar10_rewind_lt_0.2_GPU2.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.2 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 10 --prune_type rewind_lt  --rewind_epoch 3 > 0519_res18_cifar10_rewind_lt_0.2_GPU5.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.1 --init pretrained_model/res20s_cifar10_1.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 10 --prune_type rewind_lt --rewind_epoch 3 > 0519_res20s_cifar10_rewind_lt_0.2_GPU3.out &


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2_continue/6checkpoint.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.2_continue/6checkpoint.pth.tar


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2_continue/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.2_continue/6model_SA_best.pth.tar

CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2/5checkpoint.pth.tar --seed 7 --fc --evaluate --checkpoint res20s_cifar10_lt_0.2/5checkpoint.pth.tar


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.1 --init pretrained_model/epoch_3.pth.tar --seed 7 --lr 0.1 --fc --rate 0.2 --pruning_times 15 --prune_type lt > 0519_res20s_cifar10_rewind_lt_0.1_GPU3.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.1 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar10_lt_0.2/5checkpoint.pth.tar > 0519_res20s_cifar10_rewind_lt_0.1_GPU3.out &


CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2/0model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.2/0model_SA_best.pth.tar --evaluate-full 

CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.1/9model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.1/9model_SA_best.pth.tar


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.05 --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 15 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar10_lt_0.1/6checkpoint.pth.tar > 0519_res20s_cifar10_rewind_lt_0.05_GPU3.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_0.05_continue --init res20s_cifar10_lt_0.2/epoch_3.pth.tar --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 24 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res20s_cifar10_lt_0.05/14checkpoint.pth.tar > 0519_res20s_cifar10_rewind_lt_0.05_continue_GPU3.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.2_continue --init pretrained_model/resnet18_lt_cifar100.pt --seed 1 --lr 0.1 --fc --rate 0.2 --pruning_times 20 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res18_cifar100_lt_0.2/res18_cifar1000_lt_0.2/9checkpoint.pth.tar > 0520_res18_cifar100_rewind_lt_0.2_continue_GPU7.out &



CUDA_VISIBLE_DEVICES=7 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.1/9model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar100_lt_0.2_continue/16checkpoint.pth.tar




CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.1/9model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.05_continue/19model_SA_best.pth.tar