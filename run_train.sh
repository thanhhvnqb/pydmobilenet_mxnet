python train_imagenet32.py \
  --data-dir ../ImageNet32 \
  --mode hybrid \
  --lr 0.4 --lr-mode cosine --resume-epoch 0 --num-epochs 120 --batch-size 256 -j 60 \
  --warmup-epochs 5 --no-wd --label-smoothing --mixup 2>&1 | tee -a "out/run_20190111.log"
