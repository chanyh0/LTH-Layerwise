CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.1.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer1.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer1.0.conv2.weight_mask_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer1.1.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer1.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer1.1.conv1.weight_mask_GPU2.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_extreme_qrcode_layer3.2.conv2.weight_mask/model_SA_best.pth.tar


CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer1.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer1.2.conv1.weight_mask_GPU1.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.1.conv1 --pretrained res18_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.1.conv1 --pretrained pretrained_model/resnet18_lt_cifar100.pt --mask_dir ownership/res18_cifar100_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask_GPU1.out &


