CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_3000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_1000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_2000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_3000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_5000_add_back_GPU7.out &





CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 20000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_20000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 30000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_30000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 50000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 > 0529_res50_cifar10_lt_extreme_omp_50000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_10000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 20000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_20000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 30000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_30000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res50 --save_dir res50_cifar10_lt_extreme --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir ownership/res50_cifar10_extreme.pth.tar --fc --num-paths 50000 --type omp --prune-type lt --weight_decay 5e-4 --warmup 5 --batch_size 256 --lr 0.1 --add-back > 0529_res50_cifar10_lt_extreme_omp_50000_add_back_GPU7.out &