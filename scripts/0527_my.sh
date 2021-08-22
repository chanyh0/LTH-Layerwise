python -u add_qrcode_res20s_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0525/res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask/model_SA_best.pth.tar --max-name layer2.1.conv1 --evaluate-p 0.005



python -u add_qrcode_res20s_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask/model_SA_best.pth.tar --max-name layer2.1.conv1 --evaluate-p 0.05


python -u add_qrcode_res20s_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0527/res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask/model_SA_best.pth.tar --max-name layer2.2.conv1

python -u extract_qrcode_res20s_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0527/res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask/model_SA_best.pth.tar --max-name layer2.1.conv1 --evaluate-p 0.1

python -u extract_qrcode_res18_cifar100.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0528/res18_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask/model_SA_best.pth.tar --max-name layer2.1.conv2 --evaluate-p 0.1


CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir res18_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask/model_SA_best.pth.tar --evaluate --evaluate-p 0.1


CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res50 --save_dir test --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir res18_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint ownership/res50_cifar100_extreme.pth.tar --evaluate 



python -u extract_qrcode_res18_cifar100.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0528/res18_cifar100_extreme_qrcode_layer1.0.conv1.weight_mask/model_SA_best.pth.tar --max-name layer1.0.conv1 --evaluate-p 0.05