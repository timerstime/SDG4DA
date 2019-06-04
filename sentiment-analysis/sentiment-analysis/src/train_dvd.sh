export CUDA_VISIBLE_DEVICES=0
nohup python train_AC_3_dvd.py 1>logs/dvd.err 2>logs/dvd.log &
