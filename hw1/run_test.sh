python main.py \
--gpu_id 5 \
--mode test \
--ckpt_root ./ckpt \
--result_path prediction.csv \
--ckpt_path ./ckpt/checkpoint_0010.pth

# gpu id | real id
#    0   |    2
#    1   |    4
#    2   |    6
#    3   |    0
#    4   |    1
#    5   |    3
#    6   |    5
#    7   |    7