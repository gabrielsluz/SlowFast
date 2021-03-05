#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy as np
import pprint
import torch
import copy
from torch.utils.data import DataLoader

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.meters import ClevrerTrainMeter, ClevrerValMeter
#Clevrer specific
from slowfast.datasets.clevrer_dual import Clevrer_des
from slowfast.models.build import MODEL_REGISTRY

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, optimizer, train_meter, cur_epoch, cfg, test_imp=False
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
    test_counter = 0
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    
    for cur_iter, sampled_batch in enumerate(train_loader): 
        frames = sampled_batch['frames']
        des_q = sampled_batch['question_dict']['question']
        des_ans = sampled_batch['question_dict']['ans']
        # des_len = sampled_batch['question_dict']['len']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            frames = frames.cuda(non_blocking=True)
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()
            # des_len = des_len.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        #Separated batches
        #Des
        pred_des_ans = model(frames, des_q, True)
        des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        loss = des_loss_fun(pred_des_ans, des_ans)
        # check Nan Loss.
        misc.check_nan_losses(loss)
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
            lr,
            des_q.size()[0],
            mb_size_mc
        )
        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()


        #For testing implementation
        if test_imp:
            print(" --- Descriptive questions results --- ")
            # print("Des_q")
            # print(des_q)
            print("Des_ans")
            print(des_ans)
            #print("Des_ans_pred")
            #print(pred_des_ans)
            print("Argmax => prediction")
            print(torch.argmax(pred_des_ans, dim=1, keepdim=False))
            print("Top1_err and Top5err")
            print(top1_err, top5_err)
            print("Loss_des_val = {}".format(loss_des_val))
            test_counter += 1
            if test_counter == 4: 
                break

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
    test_counter = 0
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, sampled_batch in enumerate(val_loader):
        frames = sampled_batch['frames']
        des_q = sampled_batch['question_dict']['question']
        des_ans = sampled_batch['question_dict']['ans']
        # des_len = sampled_batch['question_dict']['len']
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            frames = frames.cuda(non_blocking=True)
            des_q = des_q.cuda(non_blocking=True)
            des_ans = des_ans.cuda()
            # des_len = des_len.cuda(non_blocking=True)

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
            des_q.size()[0],
            mb_size_mc
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

        #For testing implementation
        if test_imp:
            print(" --- Descriptive questions results --- ")
            # print("Des_q")
            # print(des_q)
            print("Des_ans")
            print(des_ans)
            #print("Des_ans_pred")
            #print(pred_des_ans)
            print("Argmax => prediction")
            print(torch.argmax(pred_des_ans, dim=1, keepdim=False))
            print("Top1_err and Top5err")
            print(top1_err, top5_err)
            print("Loss_des_val = {}".format(loss_des_val))
            test_counter += 1
            if test_counter == 4: 
                break

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def build_clevrer_model(cfg, gpu_id=None):
    """
    Builds and returns a CLEVRER Text model
    It is a separated function because it CLEVRER receives dataset specific parameters
    """
    dataset = Clevrer_des(cfg, 'train')
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

def build_dataloader(cfg, mode):
    dataset = Clevrer_des(cfg, mode)
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
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)
    # Create the video train and val loaders.
    if cfg.TRAIN.DATASET != 'Clevrer_des':
        print("This train script does not support your dataset: -{}-. Only Clevrer_des".format(cfg.TRAIN.DATASET))
        exit()
    # Create the video train and val loaders.
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    # Create meters.
    train_meter = ClevrerTrainMeter(len(train_loader), cfg)
    val_meter = ClevrerValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        #loader.shuffle_dataset(train_loader, cur_epoch)
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


def test_implementation_des(cfg):
    """
    Simulates a train and val epoch to check if the gradients are being updated,
    metrics are being calculated correctly
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
    logger.info("Test implementation")

    # Build the video model and print model statistics.
    model = build_clevrer_model(cfg)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Create the video train and val loaders.
    if cfg.TRAIN.DATASET != 'Clevrer_des':
        print("This train script does not support your dataset: -{}-. Only Clevrer_des".format(cfg.TRAIN.DATASET))
        exit()
    
    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    # Create meters.
    train_meter = ClevrerTrainMeter(len(train_loader), cfg)
    val_meter = ClevrerValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    # Train for one epoch.
    model_before = copy.deepcopy(model)
    cur_epoch = start_epoch
    train_epoch(
        train_loader, model, optimizer, train_meter, cur_epoch, cfg, test_imp=True
    )
    print("Check how much parameters changed")
    for (p_b_name, p_b), (p_name, p) in zip(model_before.named_parameters(), model.named_parameters()):
        if p.requires_grad:
            print("Parameter requires grad:")
            print(p_name, p_b_name)
            #Calculate ratio of change
            change = torch.abs(torch.linalg.norm(p) - torch.linalg.norm(p_b))
            print("Ratio of change = {}".format(torch.true_divide(change, torch.linalg.norm(p_b))))
            if (p_b != p).any():
                print("--Check--")
            else:
                print("ALERT - WEIGHTS DID NOT CHANGE WITH TRAINING.")
        else:
            print("Parameter does not require grad:")
            print(p_name)
            print(p)
    print("Val epoch")
    eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, test_imp=True)