python main.py \
--gpu_id 1 \
--mode train \
--ckpt_root ./ckpt \
--save_per_epoch 1 \
--batch_size 64 \
--num_epochs 200 \
--resume \

# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7