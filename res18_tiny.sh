for i in $(seq 0 20);
do
CUDA_VISIBLE_DEVICES=7 python -u main_eval_one_random.py --data datasets/tiny-imagenet-200 --dataset tiny-imagenet --arch res18 --save_dir res18_tiny_baseline --pretrained LT-model-coreset/tiny_img_resnet18_LT/corner_42.85.pth.tar --mask_dir LT-model-coreset/tiny_img_resnet18_LT/corner_42.85.pth.tar --seed 1 --conv1 --lr 0.1 --fc --random-index ${i} > res18_tiny_${i}.out
done
