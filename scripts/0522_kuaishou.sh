 CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.05 --init pretrained_model/resnet18_lt_cifar100.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 27 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res18_cifar100_lt_0.2_continue/17model_SA_best.pth.tar > 0521_res18_cifar100_rewind_lt_0.05_GPU7.out &


 CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar10_lt_0.05 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.05 --pruning_times 31 --prune_type lt --resume --checkpoint res18_cifar10_lt_0.1/21checkpoint.pth.tar  > 0522_res18_cifar10_rewind_lt_0.05_GPU5.out &


 CUDA_VISIBLE_DEVICES=1 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer3.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer3.0.conv1.weight_mask_GPU1.out &

  CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer3.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer3.0.conv2.weight_mask_GPU2.out &

  CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer3.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer3.1.conv1.weight_mask_GPU3.out &

  CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar100 --dataset cifar100 --arch res20s --save_dir res20s_cifar100_extreme_qrcode --pretrained res20s_cifar100_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar100_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar100_extreme_qrcode_layer3.1.conv2.weight_mask_GPU4.out &


  CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.0.conv1.weight_mask_GPU0.out &

  CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.0.conv2.weight_mask_GPU2.out &

  CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv1.weight_mask_GPU3.out &

  CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_GPU4.out &



  CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.0.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.0.conv1.weight_mask_GPU0.out &

  CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.0.conv2.weight_mask_GPU2.out &

  CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv1.weight_mask_GPU3.out &

  CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_GPU4.out &


  CUDA_VISIBLE_DEVICES=0 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.0.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.0.conv2.weight_mask_GPU0.out &

  CUDA_VISIBLE_DEVICES=2 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.1.conv1.weight_mask_GPU2.out &

  CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.1.conv2.weight_mask_GPU3.out &

  CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer2.2.conv1.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer2.2.conv1.weight_mask_GPU4.out &



CUDA_VISIBLE_DEVICES=3 nohup python -u main_eval_all.py --data datasets/cifar10 --dataset cifar10 --arch res20s --save_dir res20s_cifar10_extreme_qrcode --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model --prune-type lt --type identity > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_GPU3.out &



   CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res20s --save_dir res20s_cifar10_lt_extreme_trigger --pretrained res20s_cifar10_lt_0.2/epoch_3.pth.tar --mask_dir ownership/res20s_cifar10_qrcode_layer3.1.conv2.weight_mask.pth.tar --fc --lr 0.1 --seed 1 --save_model > 0521_res20s_cifar10_extreme_qrcode_layer3.1.conv2.weight_mask_trigger_GPU4.out &

   CUDA_VISIBLE_DEVICES=4 nohup python -u main_eval_trigger.py --data datasets/cifar10 --dataset cifar10_trigger --arch res50 --save_dir resnet50_cifar10_lt_extreme_trigger --pretrained LotteryTickets/cifar10_LT/random_init.pt --mask_dir LotteryTickets/cifar10_LT/7_checkpoint.pt --fc --save_model --seed 7 > 0517_resnet50_cifar10_lt_extreme_trigger_GPU4.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u main_imp_new.py --data datasets/cifar100 --dataset cifar100 --arch res18 --batch_size 256 --save_dir res18_cifar100_lt_0.01 --init pretrained_model/resnet18_lt_cifar100.pt --seed 1 --lr 0.01 --fc --rate 0.1 --pruning_times 27 --prune_type rewind_lt --rewind_epoch 3 --resume --checkpoint res18_cifar100_lt_0.2_continue/17model_SA_best.pth.tar > 0521_res18_cifar100_rewind_lt_0.01_GPU7.out &


 CUDA_VISIBLE_DEVICES=5 nohup python -u main_imp_new.py --data datasets/cifar10 --dataset cifar10 --arch res18 --batch_size 256 --save_dir res18_cifar10_lt_0.01 --init pretrained_model/resnet18_lt_cifar10.pt --seed 1 --lr 0.1 --fc --rate 0.01 --pruning_times 31 --prune_type lt --resume --checkpoint res18_cifar10_lt_0.1/21checkpoint.pth.tar  > 0522_res18_cifar10_rewind_lt_0.01_GPU5.out &