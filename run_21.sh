for i in $(seq 0 18); do
CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_2.py --data ../data --dataset cifar10 --arch res18 --save_dir rs21_${i} --pretrained LT_cifar10/21checkpoint.pth.tar --mask_dir LT_cifar10/21checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --random-index ${i} --fc --random-sparsity > rs_21_${i}.out 
done