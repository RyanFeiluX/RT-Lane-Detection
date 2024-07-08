# DATA
dataset='Custom'
data_root = 'datasets/Custom'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = 'Custom'

log_path = 'log_path'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = 'weights/culane_18.pth'
test_work_dir = None

num_lanes = 4

# Preprocess: x1_crop, x2_crop, y1_crop, y2_crop
crop = [(0.0, 1.0), (0.2, 0.55)]
