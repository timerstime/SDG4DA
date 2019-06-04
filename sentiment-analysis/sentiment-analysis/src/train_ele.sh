export CUDA_VISIBLE_DEVICES=2
nohup python train_AC_3_ele.py 1>logs/ele.err 2>logs/ele.log &
