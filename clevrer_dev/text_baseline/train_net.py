#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy as np
import pprint
import torch

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.utils.meters import ClevrerTrainMeter, ClevrerValMeter
#Clevrer specific
from slowfast.datasets.clevrer_text import Clevrertext
from slowfast.models.build import MODEL_REGISTRY

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg
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
        des_q = sampled_batch['question_dict']['des_q']
        des_ans = sampled_batch['question_dict']['des_ans']
        mc_q = sampled_batch['question_dict']['mc_q']
        mc_ans = sampled_batch['question_dict']['mc_ans']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()
            mc_q = mc_q.cuda(non_blocking=True)
            mc_ans = mc_ans.cuda()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        
        pred_des_ans = model(des_q, True)
        pred_mc_ans = model(mc_q, False)
        # Explicitly declare reduction to mean.
        des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        mc_loss_fun = losses.get_loss_func('bce_logit')(reduction="mean")
        # Compute the loss.
        loss = des_loss_fun(pred_des_ans, des_ans) + mc_loss_fun(pred_mc_ans, mc_ans)
        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        top1_err, top5_err = None, None
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        diff_mc_ans = torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float())
        mc_opt_err = 100 * (1.0 - torch.true_divide(diff_mc_ans.sum(), (4*des_q.size()[0])))
        mc_q_err = 100 * (1.0 - torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) == 4).int().sum(), des_q.size()[0]))

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err, mc_opt_err, mc_q_err  = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
            mc_opt_err.item(),
            mc_q_err.item()
        )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            mc_opt_err,
            mc_q_err,
            loss,
            lr,
            des_q.size()[0]
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
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
        des_q = sampled_batch['question_dict']['des_q']
        des_ans = sampled_batch['question_dict']['des_ans']
        mc_q = sampled_batch['question_dict']['mc_q']
        mc_ans = sampled_batch['question_dict']['mc_ans']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()
            mc_q = mc_q.cuda(non_blocking=True)
            mc_ans = mc_ans.cuda()

        val_meter.data_toc()

        pred_des_ans = model(des_q, True)
        pred_mc_ans = model(mc_q, False)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        diff_mc_ans = torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float())
        mc_opt_err = 100 * (1.0 - torch.true_divide(diff_mc_ans.sum(), (4*des_q.size()[0])))
        mc_q_err = 100 * (1.0 - torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) == 4).int().sum(), des_q.size()[0]))

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err, mc_opt_err, mc_q_err  = (
            top1_err.item(),
            top5_err.item(),
            mc_opt_err.item(),
            mc_q_err.item()
        )

        val_meter.iter_toc()
        
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            mc_opt_err,
            mc_q_err,
            des_q.size()[0]
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def build_clevrer_model(cfg, gpu_id=None):
    """
    Builds and returns a CLEVRER Text model
    It is a separated function because it CLEVRER receives dataset specific parameters
    """
    dataset = Clevrertext(cfg, 'train')
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
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg, vocab_len, ans_vocab_len, vocab)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model


def train(cfg):
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
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    train_meter = ClevrerTrainMeter(len(train_loader), cfg)
    val_meter = ClevrerValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg
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
