CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_0.1 --init resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar10_lt_0.2/7checkpoint.pth.tar > resnet18_cifar10_lt_0.1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_0.1 --init resnet18_cifar10_pt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 > resnet18_cifar10_pt_0.1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_lt_0.1 --init resnet18_cifar100_lt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_lt_0.2/6checkpoint.pth.tar > resnet18_cifar100_lt_0.1.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_0.1 --init resnet18_cifar100_pt_0.2/epoch_3_rewind_weight.pt --seed 1 --lr 0.1 --fc --rate 0.1 --pruning_times 15 --resume --checkpoint resnet18_cifar100_pt_0.2/14checkpoint.pth.tar > resnet18_cifar100_pt_0.1.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt --pretrained resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --mask_dir resnet18_cifar10_lt_0.2/7checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_lt.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_random_path.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_random_path --pretrained resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --mask_dir resnet18_cifar10_lt_0.2/7checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_lt_random_path.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_ewp.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_ewp --pretrained resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --mask_dir resnet18_cifar10_lt_0.2/7checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_lt_ewp.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_random_path_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_random_path_add_back --pretrained resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --mask_dir resnet18_cifar10_lt_0.2/7checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_lt_random_path_add_back.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_ewp_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_lt_ewp_add_back --pretrained resnet18_cifar10_lt_0.2/epoch_3_rewind_weight.pt --mask_dir resnet18_cifar10_lt_0.2/7checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_lt_ewp_add_back.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval.py --data datasets/cifar10 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt --pretrained pretrained_model/resnet18_pt.pt --mask_dir resnet18_cifar100_pt_0.2/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar100_pt.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_random_path.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_random_path --pretrained pretrained_model/resnet18_pt.pt --mask_dir resnet18_cifar100_pt_0.2/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar100_pt_random_path.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_ewp.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar100_pt_ewp --pretrained pretrained_model/resnet18_pt.pt --mask_dir resnet18_cifar100_pt_0.2/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar100_pt_ewp.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_random_path_add_back.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir resnet18_cifar10_pt_random_path_add_back --pretrained pretrained_model/resnet18_pt.pt --mask_dir resnet18_cifar10_pt_0.2/0checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --fc > resnet18_cifar10_pt_random_path_add_back.out &
