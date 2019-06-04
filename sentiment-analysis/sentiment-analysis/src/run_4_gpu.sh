export CUDA_VISIBLE_DEVICES=0
nohup python train_AC.py target_dvd 1>log/target_dvd.out 2>log/target_dvd.err &

export CUDA_VISIBLE_DEVICES=1
nohup python train_AC.py target_books 1>log/target_books.out 2>log/target_books.err &

export CUDA_VISIBLE_DEVICES=2
nohup python train_AC.py target_kitchen 1>log/target_kitchen.out 2>log/target_kitchen.err &

export CUDA_VISIBLE_DEVICES=3
nohup python train_AC.py target_electronics 1>log/target_electronics.out 2>log/target_electronics.err &

