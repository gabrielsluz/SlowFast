#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import numpy as np
import pprint
import torch
import copy

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
        #Try separating the batches
        #Des
        pred_des_ans = model(des_q, True)
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

        #MC
        pred_mc_ans = model(mc_q, False)
        mc_loss_fun = losses.get_loss_func('bce_logit')(reduction="mean")
        loss = mc_loss_fun(pred_mc_ans, mc_ans) #Multiply by 4
        # check Nan Loss.
        misc.check_nan_losses(loss)
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Save for stats
        loss_mc_val = loss

        loss = loss_mc_val + loss_des_val

        # #Non separated:
        # pred_des_ans = model(des_q, True)
        # pred_mc_ans = model(mc_q, False)
        # # Explicitly declare reduction to mean.
        # des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        # mc_loss_fun = losses.get_loss_func('bce_logit')(reduction="mean")
        # # Compute the loss.
        # loss = des_loss_fun(pred_des_ans, des_ans) + mc_loss_fun(pred_mc_ans, mc_ans)
        # # check Nan Loss.
        # misc.check_nan_losses(loss)

        # # Perform the backward pass.
        # optimizer.zero_grad()
        # loss.backward()
        # # Update the parameters.
        # optimizer.step()

        top1_err, top5_err = None, None
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        diff_mc_ans = torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float()) #Errors
        mc_opt_err = 100 * torch.true_divide(diff_mc_ans.sum(), (4*mc_q.size()[0]))
        mc_q_err = 100 * torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) != 0).float().sum(), mc_q.size()[0])
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


        #For testing implementation
        if test_imp:
            print(" --- Descriptive questions results --- ")
            print("Des_q")
            print(des_q)
            print("Des_ans")
            print(des_ans)
            #print("Des_ans_pred")
            #print(pred_des_ans)
            print("Argmax => prediction")
            print(torch.argmax(pred_des_ans, dim=1, keepdim=False))
            print("Top1_err and Top5err")
            print(top1_err, top5_err)
            print("Loss_des_val = {}".format(loss_des_val))

            print(" --- Multiple Choice questions results --- ")
            print("Mc_q")
            print(mc_q)
            print("Mc errors pred x ans")
            print(torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float()))
            print("mc_opt_err = {} \nmc_q_err = {}".format(mc_opt_err, mc_q_err))
            print("Loss_mc_val = {}".format(loss_mc_val))
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
        # Explicitly declare reduction to mean.
        des_loss_fun = losses.get_loss_func('cross_entropy')(reduction="mean")
        mc_loss_fun = losses.get_loss_func('bce_logit')(reduction="mean")
        # Compute the loss.
        loss_des_val = des_loss_fun(pred_des_ans, des_ans)
        loss_mc_val = mc_loss_fun(pred_mc_ans, mc_ans) 
        loss = loss_des_val + loss_mc_val
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(pred_des_ans, des_ans, (1, 5))
        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / pred_des_ans.size(0)) * 100.0 for x in num_topks_correct
        ]
        diff_mc_ans = torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float()) #Errors
        mc_opt_err = 100 * torch.true_divide(diff_mc_ans.sum(), (4*mc_q.size()[0]))
        mc_q_err = 100 * torch.true_divide((diff_mc_ans.sum(dim=1, keepdim=True) != 0).float().sum(), mc_q.size()[0])

        # Copy the errors from GPU to CPU (sync point).
        loss, top1_err, top5_err, mc_opt_err, mc_q_err  = (
            loss.item(),
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
            loss,
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

        #For testing implementation
        if test_imp:
            print(" --- Descriptive questions results --- ")
            print("Des_q")
            print(des_q)
            print("Des_ans")
            print(des_ans)
            #print("Des_ans_pred")
            #print(pred_des_ans)
            print("Argmax => prediction")
            print(torch.argmax(pred_des_ans, dim=1, keepdim=False))
            print("Top1_err and Top5err")
            print(top1_err, top5_err)
            print("Loss_des_val = {}".format(loss_des_val))

            print(" --- Multiple Choice questions results --- ")
            print("Mc_q")
            print(mc_q)
            print("Mc errors pred x ans")
            print(torch.abs(mc_ans - (torch.sigmoid(pred_mc_ans) >= 0.5).float()))
            print("mc_opt_err = {} \nmc_q_err = {}".format(mc_opt_err, mc_q_err))
            print("Loss_mc_val = {}".format(loss_mc_val))
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


def test_implementation(cfg):
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
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

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
    print("Check if parameters changed")
    for (p_b_name, p_b), (p_name, p) in zip(model_before.named_parameters(), model.named_parameters()):
        if p.requires_grad:
            print("Parameter requires grad:")
            print(p_name, p_b_name)
            assert (p_b != p).any()
            print("--Check--")
        else:
            print("Parameter does not require grad:")
            print(p_name)
            print(p)
    print("Val epoch")
    eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, test_imp=True)