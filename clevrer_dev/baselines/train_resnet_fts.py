from slowfast.utils.parser import load_config, parse_args
import numpy as np
import pprint
import torch
import copy
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

import slowfast.models.losses as losses
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.meters import ClevrerTrainMeter, ClevrerValMeter
#Clevrer specific
from slowfast.datasets.clevrer_resnet import Clevrerresnet_des
from slowfast.models.cnn_models import CNN_PRE_LSTM

logger = logging.get_logger(__name__)
"""
python3 clevrer_dev/baselines/train_resnet_fts.py \
  --cfg clevrer_dev/baselines/cnn_lstm.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer \
  TRAIN.DATASET Clevrerresnet_des \
  TRAIN.ONLY_DES True \
  TRAIN.ENABLE True \
  TRAIN.BATCH_SIZE 32 \
  NUM_GPUS 1 \
  LOG_PERIOD 100 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 10 \
  SOLVER.BASE_LR 0.001 \
  SOLVER.WEIGHT_DECAY 0.000001 \
  CLEVRERMAIN.LSTM_HID_DIM 512 \
  CLEVRERMAIN.T_DROPOUT 0.2 \
  WORD_EMB.EMB_DIM 512 \
  WORD_EMB.TRAINABLE True \
  WORD_EMB.USE_PRETRAINED_EMB False \
  SOLVER.MAX_EPOCH 60
"""



def train_epoch(
    train_loader, model, optimizer, scheduler, train_meter, cur_epoch, cfg, test_imp=False
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (ClevrerTrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    
    for cur_iter, sampled_batch in enumerate(train_loader): 
        frames = sampled_batch['res_ft']
        des_q = sampled_batch['question_dict']['question']
        des_ans = sampled_batch['question_dict']['ans']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            frames = frames.cuda(non_blocking=True)
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()

        train_meter.data_toc()
        #Pass through
        model.zero_grad() 
        pred_des_ans = model(frames, des_q, True)
        des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        loss = des_loss_fun(pred_des_ans, des_ans)
        # check Nan Loss.
        misc.check_nan_losses(loss)
        #Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        #Save for stats
        loss_des_val = loss

        top1_err, top5_err = None, None
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        mc_opt_err, mc_q_err = None, None
        mb_size_mc = None
        loss_des_val, top1_err, top5_err = (
            loss_des_val.item(),
            top1_err.item(),
            top5_err.item()
        )
        #top1_err, top5_err, mc_opt_err, mc_q_err, loss_des, loss_mc, lr, mb_size
        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            mc_opt_err,
            mc_q_err,
            loss_des_val,
            None,
            scheduler.get_last_lr(),
            des_ans.size(0),
            mb_size_mc
        )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, test_imp=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ClevrerValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, sampled_batch in enumerate(val_loader):
        frames = sampled_batch['res_ft']
        des_q = sampled_batch['question_dict']['question']
        des_ans = sampled_batch['question_dict']['ans']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            frames = frames.cuda(non_blocking=True)
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()

        val_meter.data_toc()

        # Explicitly declare reduction to mean.
        des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        pred_des_ans = model(frames, des_q, True)
        loss_des_val = des_loss_fun(pred_des_ans, des_ans)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        loss_mc_val = None
        mc_opt_err, mc_q_err = None, None
        mb_size_mc = None
        loss_des_val, top1_err, top5_err = (
            loss_des_val.item(),
            top1_err.item(),
            top5_err.item()
        )

        val_meter.iter_toc()
        #top1_err, top5_err, mc_opt_err, mc_q_err, loss_des, loss_mc, mb_size_des, mb_size_mc
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            mc_opt_err,
            mc_q_err,
            loss_des_val,
            loss_mc_val,
            des_ans.size(0),
            mb_size_mc
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def build_clevrer_model(cfg, gpu_id=None):
    """
    It is a separated function because it CLEVRER receives dataset specific parameters
    """
    dataset = Clevrerresnet_des(cfg, 'train')
    vocab_len = dataset.get_vocab_len()
    ans_vocab_len = dataset.get_ans_vocab_len()
    vocab = dataset.get_vocab()

    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    model = CNN_PRE_LSTM(cfg, vocab_len, ans_vocab_len, vocab)

    if cfg.NUM_GPUS:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
    return model

def build_dataloader(cfg, mode):
    dataset = Clevrerresnet_des(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle= mode=='train', num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                        pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    return dataloader

def train_des(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_clevrer_model(cfg)

    # Construct the optimizer.
    optimizer = AdamW(model.parameters(),
                  lr = cfg.SOLVER.BASE_LR,
                  eps = 1e-8
                )
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    # Create the video train and val loaders.
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    total_steps = len(train_loader) * cfg.SOLVER.MAX_EPOCH

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Create meters.
    train_meter = ClevrerTrainMeter(len(train_loader), cfg)
    val_meter = ClevrerValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, scheduler, train_meter, cur_epoch, cfg
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    train_des(cfg)
