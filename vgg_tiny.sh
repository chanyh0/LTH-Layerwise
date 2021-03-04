for i in $(seq 0 20);
do
CUDA_VISIBLE_DEVICES=3 python -u main_eval_one_random.py --data datasets/tiny-imagenet-200 --dataset tiny-imagenet --arch vgg16_bn --save_dir vgg16_tiny_baseline --pretrained LT-model-coreset/tiny_img_vgg16_LT/dense57.93.pth.tar --mask_dir LT-model-coreset/tiny_img_vgg16_LT/dense57.93.pth.tar --seed 1 --conv1 --lr 0.1 --fc --random-index ${i} > vgg_tiny_${i}.out
done