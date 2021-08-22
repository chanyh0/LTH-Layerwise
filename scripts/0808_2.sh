7 22

CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir vgg16_bn_qrcode --pretrained 20checkpoint.pth.tar --mask_dir ownership/vgg16_tiny_qrcode_features.7.weight_mask.pth.tar --fc --type identity --prune-type lt --save_model

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_trigger.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet-trigger --arch vgg16_bn --save_dir vgg16_bn_tiny-imagenet_lt_extreme_trigger --pretrained 20checkpoint.pth.tar --mask_dir ownership/vgg16_tiny_qrcode_features.7.weight_mask.pth.tar --fc --save_model --lr 0.1 > 0808_vgg16_bn_tiny-imagenet_lt_extreme_trigger_GPU1.out &

CUDA_VISIBLE_DEVICES=1 python -u extract_qrcode_vgg16_bn_tiny.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint vgg16_bn_qrcode/model_SA_best.pth.tar --max-name features.7 --evaluate-p 0.0

CUDA_VISIBLE_DEVICES=1 python -u main_eval_trigger.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet-trigger --arch vgg16_bn --save_dir vgg16_bn_tiny-imagenet_lt_extreme_trigger --pretrained 20checkpoint.pth.tar --mask_dir ownership/vgg16_tiny_qrcode_features.7.weight_mask.pth.tar --fc --save_model --lr 0.1 --checkpoint vgg16_bn_tiny-imagenet_lt_extreme_trigger/model_SA_best.pth.tar --evaluate

CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint vgg16_bn_qrcode/model_SA_best.pth.tar --evaluate-p 0.05 --evaluate

python embed_tiny_vgg16.py features.13.weight_mask # 39 29
python embed_tiny_vgg16.py features.19.weight_mask # 208 86


CUDA_VISIBLE_DEVICES=1 python -u extract_qrcode_vgg16_bn_tiny.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint vgg16_bn_qrcode/model_SA_best.pth.tar --max-name features.7 --evaluate-p 0.1


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir vgg16_bn_qrcode_10 --pretrained 20checkpoint.pth.tar --mask_dir ownership/vgg16_tiny_qrcode_features.10.weight_mask.pth.tar --fc --type identity --prune-type lt --save_model > 10.out & 

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir vgg16_bn_qrcode_14 --pretrained 20checkpoint.pth.tar --mask_dir ownership/vgg16_tiny_qrcode_features.14.weight_mask.pth.tar --fc --type identity --prune-type lt --save_model > 14.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir vgg16_bn_qrcode_0 --pretrained 20checkpoint.pth.tar --mask_dir 20checkpoint.pth.tar --fc --type identity --prune-type lt --save_model > 0.out & 

CUDA_VISIBLE_DEVICES=1 python -u extract_qrcode_vgg16_bn_tiny.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint vgg16_bn_qrcode_10/model_SA_best.pth.tar --max-name features.10 --evaluate-p 0.1

CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --checkpoint vgg16_bn_qrcode_10/model_SA_best.pth.tar --evaluate-p 0.05 --evaluate