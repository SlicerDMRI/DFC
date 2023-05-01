#train
#python train.py --mode train_full --pretrain True --batch_size 1024 --net_architecture DGCNN --dataset Fiber --epochs_pretrain 50 --epochs 1 -p 14 --k 5 --fs True --surf True --data HCP --ro True -indir data/train  -outdir results_atlas
# test
python test.py -indir data/test -outdir results -modeldir nets/model.pt --save True
