CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res18_tiny-imagenet_b32_e160_lr0.2_w20 \
  > $SCRATCH/0501_res18_tiny-imagenet_b32_e160_lr0.2_w20_IMP_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res18_tiny-imagenet_b32_e160_lr0.4_w20 \
  > $SCRATCH/0501_res18_tiny-imagenet_b32_e160_lr0.4_w20_IMP_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 15  --batch_size 32 --save_dir $SCRATCH/res18_tiny-imagenet_b32_e160_lr0.05_w15  \
  > $SCRATCH/0501_res18_tiny-imagenet_b32_e160_lr0.05_w20_IMP_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res18_tiny-imagenet_b32_e160_lr0.2_w15 \
  > $SCRATCH/0501_res18_tiny-imagenet_b32_e160_lr0.2_w15_IMP_GPU3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res18 --pruning_times 20 --rate 0.2 --prune_type lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir $SCRATCH/res18_tiny-imagenet_b32_e160_lr0.4_w15 \
  > $SCRATCH/0501_res18_tiny-imagenet_b32_e160_lr0.4_w15_IMP_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type lt --epoch 160 --lr 0.05 --decreasing_lr 80,115 \
  --warmup 10  --batch_size 32 --save_dir $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.05_w10 \
  > $SCRATCH/0501_res50_tiny-imagenet_b32_e160_lr0.05_w10_IMP_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type lt --epoch 160 --lr 0.075 --decreasing_lr 80,115 \
  --warmup 10  --batch_size 32 --save_dir $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.075_w10  \
  > $SCRATCH/0501_res50_tiny-imagenet_b32_e160_lr0.075_w10_IMP_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data $SCRATCH/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type lt --epoch 160 --lr 0.1 --decreasing_lr 80,115 \
  --warmup 10  --batch_size 32 --save_dir $SCRATCH/res50_tiny-imagenet_b32_e160_lr0.1_w10 \
  > $SCRATCH/0501_res50_tiny-imagenet_b32_e160_lr0.1_w10_IMP_GPU3.out &