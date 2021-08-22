
CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2/0model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_0.2/2model_SA_best.pth.tar



CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_lt_extreme/model_SA_best.pth.tar --evaluate-p 0.05

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100000 --type identity --prune-type lt --save_model > 0808_res20s_cifar10_lt_extreme_identity_GPU2.out &

CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.2/2model_SA_best.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  