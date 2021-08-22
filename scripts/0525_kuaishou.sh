CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type random_path --prune-type lt > 0525_res20s_cifar100_lt_extreme_random_path_2000_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type random_path --prune-type lt > 0525_res20s_cifar100_lt_extreme_random_path_5000_GPU3.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type random_path --prune-type lt > 0525_res20s_cifar100_lt_extreme_random_path_10000_GPU2.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 2000 --type ewp --prune-type lt > 0525_res20s_cifar100_lt_extreme_ewp_2000_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt > 0525_res20s_cifar100_lt_extreme_ewp_5000_GPU5.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt > 0525_res20s_cifar100_lt_extreme_ewp_10000_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_lt_extreme --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt > 0525_res20s_cifar100_lt_extreme_ewp_20000_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 5000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_5000_add_back_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 10000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_10000_add_back_GPU3.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type ewp --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_ewp_20000_add_back_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_lt_extreme --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_extreme.pth.tar --fc --num-paths 20000 --type random_path --prune-type lt --add-back > 0523_res20s_cifar10_lt_extreme_random_path_20000_add_back_GPU1.out &


  CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer3.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer3.1.conv1.weight_mask_GPU4.out &

    CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_GPU5.out &

  CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer3.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer3.0.conv1.weight_mask_GPU6.out &

  CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer3.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar100_extreme_qrcode_layer3.1.conv1.weight_mask_GPU7.out &


  CUDA_VISIBLE_DEVICES=3 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar100_extreme_qrcode/model_SA_best.pth.tar




     CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_trigger_GPU2.out &

     CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model > 0525_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_trigger_GPU3.out &

     
     CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode2 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask2_GPU5.out &




CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask_GPU4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.2.conv2.weight_mask_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask_GPU4.out &

CUDA_VISIBLE_DEVICES=6 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer2.2.conv1.weight_mask_GPU6.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask_GPU7.out &


CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.0.conv2 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.0.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.0.conv1.weight_mask_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.1.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU3.out &

CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.1.conv2 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask_GPU4.out &

CUDA_VISIBLE_DEVICES=5 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.2.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask_GPU5.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.2.conv2 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res18_cifar10_extreme_qrcode_layer2.2.conv2.weight_mask_GPU7.out &



CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir test --pretrained pretrained_model/epoch_3.pth.tar --mask_dir res18_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar100_extreme_qrcode/model_SA_best.pth.tar --evaluate-p 0.3



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership0525/res20s_cifar100_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar100_extreme_qrcode_layer2.1.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership0525/res20s_cifar100_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar100_extreme_qrcode_layer2.2.conv1.weight_mask_GPU2.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode_layer2.2.conv2 --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership0525/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask_GPU7.out &



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU2.out &

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.2.conv2 --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask_GPU7.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU7.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership0525/res20s_cifar100_qrcode_layer2.2.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask_GPU2.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer2.1.conv1 --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU0.out &



CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_extreme_qrcode_layer3.2.conv2.weight_mask/model_SA_best.pth.tar --evaluate-p 0.0

CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir test --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar10_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask/model_SA_best.pth.tar --evaluate-p 0.0

CUDA_VISIBLE_DEVICES=7 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res18 --save_dir res18_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask --pretrained res18_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res18_cifar100_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar100_extreme_qrcode_layer2.1.conv1.weight_mask_GPU7.out &


CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir test --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res20s_cifar100_extreme_qrcode_layer2.2.conv2.weight_mask/model_SA_best.pth.tar --evaluate-p 0.0

CUDA_VISIBLE_DEVICES=1 python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir test --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir res20s_cifar100_lt_0.01/6model_SA_best.pth.tar --fc --evaluate --checkpoint res18_cifar10_extreme_qrcode_layer2.1.conv1/model_SA_best.pth.tar --evaluate-p 0.0



CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res18 --save_dir res18_cifar10_extreme_qrcode_layer1.1.conv2.weight_mask --pretrained pretrained_model/resnet18_lt_cifar10.pt --mask_dir ownership/res18_cifar10_qrcode_layer1.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0525_res18_cifar10_extreme_qrcode_layer1.1.conv2.weight_mask_GPU0.out &




