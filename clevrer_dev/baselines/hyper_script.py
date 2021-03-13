#Train the network several times, testing for different hyperparameters.
"""
Decisions:
- Rewrite training functions 
- How to make the validation dataset smaller => modifying the dataset class
- How to log the stats to a file ? => Use the Meters to append to the file, and always print the
    parameters in a convenient way => dict. All params, including the initial ones.
- 

Need:
- Give the log file an appropriated name
- Deal with checkpoints => we do not want them => comment the checkpoint making
"""


"""
python3 clevrer_dev/baselines/run_net.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.DATASET Clevrer_des \
  TRAIN.ONLY_DES True \
  TRAIN.ENABLE True \
  WORD_EMB.USE_PRETRAINED_EMB False \
  WORD_EMB.TRAINABLE True \
  WORD_EMB.GLOVE_PATH '/datasets/word_embs/glove.6B/glove.6B.50d.txt' \
  WORD_EMB.EMB_DIM 16 \
  CLEVRERMAIN.LSTM_HID_DIM 64 \
  DATA.NUM_FRAMES 10 \
  DATA.SAMPLING_RATE 12 \
  TRAIN.BATCH_SIZE 75 \
  NUM_GPUS 1 \
  LOG_PERIOD 100 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  SOLVER.EPOCH_CYCLE 40.0 \
  SOLVER.LR_POLICY cosine \
  SOLVER.BASE_LR 0.001 \
  SOLVER.COSINE_END_LR 0.00005 \
  SOLVER.WEIGHT_DECAY 0.00005 \
  SOLVER.OPTIMIZING_METHOD adam \
  SOLVER.MAX_EPOCH 2
{"_type": "val_iter", "epoch": "1/2", "eta": "0:01:24", "gpu_mem": "4.48G", "iter": "100/734", "loss_des": 4.77786, "time_diff": 0.13320, "top1_err": 84.00000, "top5_err": 50.04000}
"""
from slowfast.config.defaults import get_cfg
from slowfast.utils.misc import launch_job

from train_net_des import train_des

def get_init_params_cfg():
    #Init parameters
    cfg = get_cfg()
    cfg.TRAIN.ENABLE = True
    cfg.TRAIN.ONLY_DES = True
    cfg.TRAIN.DATASET = "Clevrer_des"
    cfg.TRAIN.BATCH_SIZE = 16
    cfg.TRAIN.EVAL_PERIOD = 2
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.TRAIN.AUTO_RESUME = True
    cfg.TRAIN.TRAIN_STATS_FILE = "./train_stats_hyper.txt"

    cfg.DATA.RESIZE_H = 224
    cfg.DATA.RESIZE_W = 224
    cfg.DATA.NUM_FRAMES = 15
    cfg.DATA.SAMPLING_RATE = 8
    cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
    cfg.DATA.TRAIN_CROP_SIZE = 224 
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3]
    cfg.DATA.PATH_TO_DATA_DIR = "/datasets/clevrer"
    cfg.DATA.PATH_PREFIX = "/datasets/clevrer"
    cfg.DATA.MAX_TRAIN_LEN = None
    cfg.DATA.MAX_VAL_LEN = None

    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.LR_POLICY = "cosine"
    cfg.SOLVER.COSINE_END_LR = 0.00001
    cfg.SOLVER.EPOCH_CYCLE = 10.0
    cfg.SOLVER.MAX_EPOCH = 10
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.NESTEROV = True
    cfg.SOLVER.WEIGHT_DECAY = 0.001
    cfg.SOLVER.WARMUP_EPOCHS = 0.0
    cfg.SOLVER.WARMUP_START_LR = 0.01
    cfg.SOLVER.OPTIMIZING_METHOD = "adam"
    
    cfg.MODEL.ARCH = "CNN_LSTM"
    cfg.MODEL.MODEL_NAME = "CNN_LSTM"

    cfg.NUM_GPUS = 1
    cfg.LOG_PERIOD = 100
    cfg.OUTPUT_DIR = "./"

    cfg.WORD_EMB.USE_PRETRAINED_EMB = False
    cfg.WORD_EMB.TRAINABLE = True
    cfg.WORD_EMB.GLOVE_PATH = '/datasets/word_embs/glove.6B/glove.6B.50d.txt'
    cfg.WORD_EMB.EMB_DIM = 1000

    cfg.CLEVRERMAIN.LSTM_HID_DIM = 512
    cfg.CLEVRERMAIN.T_DROPOUT = 0.5

    return cfg

def run_exp(cfg):
    init_method = 'tcp://localhost:9999'
    with open(cfg.TRAIN.TRAIN_STATS_FILE, 'a') as f:
        f.write(str(dict(cfg.items())))
        f.write('\n')
    launch_job(cfg=cfg, init_method=init_method, func=train_des) 

def main():
    #1
    cfg = get_init_params_cfg()
    run_exp(cfg) 
    #2
    # cfg.SOLVER.WEIGHT_DECAY = 0.001
    # run_exp(cfg)
    # #3
    # cfg.SOLVER.WEIGHT_DECAY = 0.005
    # run_exp(cfg)
    # #4
    # cfg.SOLVER.WEIGHT_DECAY = 0.0005
    # run_exp(cfg)
    # #5
    # cfg.SOLVER.BASE_LR = 0.001
    # cfg.SOLVER.WEIGHT_DECAY = 0.005
    # run_exp(cfg)
    # #6
    # cfg.SOLVER.BASE_LR = 0.01
    # cfg.SOLVER.WEIGHT_DECAY = 0.01
    # run_exp(cfg)
    # #7
    # cfg.SOLVER.BASE_LR = 0.001
    # cfg.SOLVER.WEIGHT_DECAY = 0.005
    # cfg.DATA.MAX_TRAIN_LEN = None
    # cfg.DATA.MAX_VAL_LEN = None
    # run_exp(cfg)

if __name__ == "__main__":
    main()
