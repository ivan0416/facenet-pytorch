# Train Model
python train_center.py --dataroot "人臉資料" --lfw "validatio 資料(預設不做validation，需修改code)" --loss_used upperbound_onehot --batch_size 55 --center_loss_lr 0.0005 --embedding_dim 512 --model resnet34 --resume "checkpoint"
## Loss Used
upperbound:隨機生成centroid

upperbound_kmeans:用kmeans的方式做出centroid

upperbound_onehot:用onehot的方式做出centroid，如果embedding_dim>class數則後面centroid後面補0
