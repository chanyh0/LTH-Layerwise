python -u extract_qrcode.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership/trained/res20s_cifar100_extreme_qrcode/model_SA_best.pth.tar --max-name layer3.1.conv2 --evaluate-p 0.1

python -u extract_qrcode_res20s_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0525/res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask/model_SA_best.pth.tar --max-name layer2.1.conv1 --evaluate-p 0.0

python -u extract_qrcode_res20s_cifar100.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0525/res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask/model_SA_best.pth.tar --max-name layer2.2.conv2 --evaluate-p 0.0

python -u extract_qrcode_res18_cifar10.py --data datasets/cifar10 --dataset cifar10  --save_dir test --fc --evaluate-p 0.10

python -u extract_qrcode_res18_cifar10.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint ownership0526/model_SA_best.pth.tar --max-name layer1.1.conv2 --evaluate-p 0.00