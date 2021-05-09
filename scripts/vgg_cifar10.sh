for i in $(seq 0 16);
do
CUDA_VISIBLE_DEVICES=3 python -u main_eval_one_random.py --data datasets --dataset cifar10 --arch vgg16_bn --save_dir vgg16_bn_cifar10_baseline --pretrained LT-model-coreset/cifar10_vgg16_LT/corner_93.10.pth.tar --mask_dir LT-model-coreset/cifar10_vgg16_LT/corner_93.10.pth.tar --seed 1 --conv1 --lr 0.1 --fc --random-index ${i} > vgg_cifar10_${i}.out
done