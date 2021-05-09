#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Longhorn v100 nodes
#
#   *** Multiple Serial Jobs in v100 Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Longhorn login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#----------------------------------------------------
#SBATCH -J neu           # Job name
#SBATCH -o neu.o%j       # Name of stdout output file
#SBATCH -e neu.e%j       # Name of stderr error file
#SBATCH -p v100            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=tlc619@tacc.utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A MLL             # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module list
pwd
date

conda activate pytorch1.7

cd $SCRATCH
mkdir data
cd data
# download and unzip dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"

# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images

cd ../
git clone https://github.com/xxchenxx/LTH-Layerwise
cd LTH-Layerwise
git checkout neurips
mkdir init && cd init
wget https://www.dropbox.com/s/h1am0626esbfri6/res50_tiny-imagenet_1.pth.tar?dl=0
cd ../
CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.05 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.05_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.05_w20_IMP_GPU0.out &


CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.075 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.075_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.075_w20_IMP_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.1 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.1_w20  \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.1_w20_IMP_GPU2.out &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.125 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.125_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.125_w20_IMP_GPU3.out &

CUDA_VISIBLE_DEVICES=0 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.15 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.15_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.15_w20_IMP_GPU0.out &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.2 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.2_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.2_w20_IMP_GPU1.out &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_imp.py --data ../data/tiny-imagenet-200 --dataset tiny-imagenet \
--seed 1 --arch res50 --pruning_times 15 --rate 0.2 --prune_type rewind_lt --epoch 160 --lr 0.4 --decreasing_lr 80,120 \
  --warmup 20  --batch_size 32 --save_dir res50_tiny-imagenet_b32_e160_lr0.4_w20 \
  --rewind_epoch 8 \
  > 0503_res50_tiny-imagenet_b32_e160_lr0.4_w20_IMP_GPU2.out &

sleep 24h