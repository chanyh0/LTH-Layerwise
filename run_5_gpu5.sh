for i in $(seq 4 6); 
do
CUDA_VISIBLE_DEVICES=5 python -u main_eval_2.py --data ../data --dataset cifar10 --arch res18 --save_dir run_5 --pretrained LT_cifar10/5checkpoint.pth.tar --mask_dir LT_cifar10/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --random-index ${i} --fc > 5_${i}.out
done

for i in $(seq 4 6); 
do
CUDA_VISIBLE_DEVICES=5 python -u main_eval_2.py --data ../data --dataset cifar10 --arch res18 --save_dir run_5 --pretrained LT_cifar10/5checkpoint.pth.tar --mask_dir LT_cifar10/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --random-index ${i} --fc --random-sparsity > 5_${i}_rs.out
done

for i in $(seq 4 6); 
do
CUDA_VISIBLE_DEVICES=5 python -u main_eval_2.py --data ../data --dataset cifar10 --arch res18 --save_dir run_5 --pretrained LT_cifar10/5checkpoint.pth.tar --mask_dir LT_cifar10/5checkpoint.pth.tar --seed 1 --conv1 --lr 0.1 --random-index ${i} --fc --random-sparsity-normal > 5_${i}_rs_normal.out
done