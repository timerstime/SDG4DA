export CUDA_VISIBLE_DEVICES=1
nohup python train_AC_3_books.py 1>logs/books.err 2>logs/books.log &
