CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 32 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b32_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b32_e160_lr0.1_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b64_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b64_e160_lr0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 128 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b128_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b128_e160_lr0.1_GPU2.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 256 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b256_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b256_e160_lr0.1_GPU2.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 512 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b512_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b512_e160_lr0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 256 --lr 0.125 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b128_e160_lr0.125_w0 --warmup 0 > 0420_res18_tiny-imagenet_b128_e160_lr0.125_GPU2.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 512 --lr 0.1 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b512_e160_lr0.1_w0 --warmup 0 > 0420_res18_tiny-imagenet_b512_e160_lr0.1_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.125 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b64_e160_lr0.125_w0 --warmup 0 > 0420_res18_tiny-imagenet_b64_e160_lr0.125_GPU2.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet --seed 1 --arch res18 --pruning_times 1 --rate 0.2 --prune_type lt --epoch 160 --batch_size 64 --lr 0.075 --decreasing_lr 80,120 --save_dir res18_tiny-imagenet_b64_e160_lr0.075_w0 --warmup 0 > 0420_res18_tiny-imagenet_b64_e160_lr0.075_GPU1.out &

