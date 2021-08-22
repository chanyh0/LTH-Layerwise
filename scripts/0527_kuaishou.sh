CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_200_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_100_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_300_add_back_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_500_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 200 --type owp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_owp_200_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type owp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_owp_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 300 --type owp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_owp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type owp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_owp_500_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_200_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_ewp_500_add_back_GPU7.out &




CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_200_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_100_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_random_path_500_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_200_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_100_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 500 --type random_path --prune-type lt  > 0523_res20s_cifar100_lt_extreme_random_path_500_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2500 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_betweenness_2500_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type betweenness --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_betweenness_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2500 --type betweenness --prune-type lt > 0523_res20s_cifar100_lt_extreme_betweenness_2500_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type betweenness --prune-type lt > 0523_res20s_cifar100_lt_extreme_betweenness_5000_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 200 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_200_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_100_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 300 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 500 --type ewp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_ewp_500_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2500 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_2500_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type taylor1_abs --prune-type lt --add-back > 0523_res20s_cifar100_lt_extreme_taylor1_abs_5000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2500 --type taylor1_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_taylor1_abs_2500_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type taylor1_abs --prune-type lt > 0523_res20s_cifar100_lt_extreme_taylor1_abs_5000_GPU7.out &








CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt  > 0523_res18_cifar100_lt_extreme_random_path_100_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt  > 0523_res18_cifar100_lt_extreme_random_path_200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt  > 0523_res18_cifar100_lt_extreme_random_path_300_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 400 --type random_path --prune-type lt  > 0523_res18_cifar100_lt_extreme_random_path_400_GPU3.out &


CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 400 --type random_path --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_random_path_400_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0527_res20s_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0527_res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0527_res20s_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask_GPU2.out &


CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0527_res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.2.conv2.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0527_res20s_cifar10_extreme_qrcode_layer2.2.conv2.weight_mask_GPU4.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer1.0.conv1.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer1.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer1.0.conv1.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcodelayer1.0.conv2.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer1.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer1.0.conv2.weight_mask_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer1.1.conv1.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer1.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer1.1.conv1.weight_mask_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer1.1.conv2.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer1.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer1.1.conv2.weight_mask_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.0.conv1.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer2.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer2.0.conv1.weight_mask_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.0.conv2.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer2.0.conv2.weight_mask_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask_GPU7.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask/model_SA_best.pth.tar






CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_3000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_500_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_3000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_5000_GPU3.out &




CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 100 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_100_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 200 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_200_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 300 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_300_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_lt_extreme --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_extreme.pth.tar --fc --num-paths 500 --type omp --prune-type lt --add-back > 0523_res18_cifar100_lt_extreme_omp_500_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_8000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_12000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt  > 0523_res20s_cifar100_lt_extreme_omp_15000_GPU3.out &





CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir resnet18_cifar10_transfer_lt_0.1_ewp --pretrained resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --mask_dir resnet18_cifar10_lt_0.1/3checkpoint.pth.tar --fc --num-paths 10000 --prune-type transfer --type ewp > 0410_resnet18_cifar10_transfer_lt_extreme_ewp_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_8000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_10000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_12000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_15000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 8000 --type omp --prune-type lt  --add-back > 0523_res20s_cifar10_lt_extreme_omp_8000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_10000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 12000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_12000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 15000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_15000_add_back_GPU7.out &




CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1200 --type omp --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_omp_1200_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_ewp_100_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type betweenness --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_betweenness_100_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1200 --type taylor1_abs --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_taylor1_abs_1200_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 100 --type random_path --prune-type transfer > 0523_res20s_cifar10_transfer_extreme_random_path_100_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 1200 --type omp --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_omp_1200_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type ewp --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_ewp_100_GPU7.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 100 --type betweenness --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_betweenness_100_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model > 0523_res20s_cifar10_extreme_trigger_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --save_model > 0523_res20s_cifar100_extreme_trigger_GPU1.out &


CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type omp --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_omp_5000_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 3000 --type betweenness --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_betweenness_3000_GPU7.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme_transfer --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type betweenness --prune-type transfer > 0523_res20s_cifar100_transfer_extreme_betweenness_2000_GPU7.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --evaluate --checkpoint res20s_cifar10_lt_extreme_trigger/model_SA_best.pth.tar 

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --lr 0.1 > 0523_res20s_cifar10_extreme_trigger_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --save_model --lr 0.1 > 0523_res20s_cifar100_extreme_trigger_GPU1.out &

CUDA_VISIBLE_DEVICES=1 python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --evaluate --checkpoint res20s_cifar100_lt_extreme_trigger/model_SA_best.pth.tar --checkpoint



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_1000_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_2000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_3000_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_4000_GPU2.out &



CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_1000_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_2000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_3000_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_4000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --save_model --evaluate --checkpoint res20s_cifar10_lt_extreme_trigger/model_SA_best.pth.tar --evaluate-p 0.01



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --lr 0.1 > 0528_res20s_cifar10_extreme_trigger_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --save_model --lr 0.1 > 0528_res20s_cifar100_extreme_trigger_GPU2.out &


CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --evaluate --checkpoint res20s_cifar10_lt_extreme_trigger/model_SA_best.pth.tar --evaluate-p 0.00

CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --save_model --evaluate --checkpoint res20s_cifar100_lt_extreme_trigger/model_SA_best.pth.tar --evaluate-p 0.05


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger0 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --save_model --lr 0.1 > 0523_res20s_cifar10_extreme_trigger_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger0 --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --save_model --lr 0.1 > 0523_res20s_cifar100_extreme_trigger_GPU1.out &



CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --evaluate --checkpoint ownership/res20s_cifar10_extreme.pth.tar --evaluate-p 0.00

CUDA_VISIBLE_DEVICES=0 python -u main_eval_trigger.py --data datasets/cifar100 --dataset cifar100_trigger --arch res20s --save_dir res20s_cifar100_lt_extreme_trigger --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --save_model --evaluate --checkpoint res20s_cifar100_lt_0.01/17model_SA_best.pth.tar --evaluate-p 0.00



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_500_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_1000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_1500_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_2000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 500 --type omp --prune-type lt  --add-back > 0523_res20s_cifar10_lt_extreme_omp_500_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_1000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 1500 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_1500_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_2000_add_back_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_2500_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_3000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_3500_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_4000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type omp --prune-type lt  --add-back > 0523_res20s_cifar10_lt_extreme_omp_2500_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_3000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3500 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_3500_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_4000_add_back_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_2500_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_3000_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3500 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_3500_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt  > 0523_res20s_cifar10_lt_extreme_omp_4000_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 2500 --type omp --prune-type lt  --add-back > 0523_res20s_cifar10_lt_extreme_omp_2500_add_back_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_3000_add_back_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 3500 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_3500_add_back_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 4000 --type omp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_omp_4000_add_back_GPU7.out &
