#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import sys
import os

from tqdm import tqdm
import h5py

from slowfast.datasets.clevrer_monet import Clevrer_monet
import slowfast.utils.checkpoint as cu
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.logging as logging

from models import create_model
from models.monet_model import  MONetModel

"""
Generates MONet features

Example:
python3 generate_features.py \
  --cfg /home/gabrielsluz/SlowFast/clevrer_dev/mac/mac.yaml \
  DATA.PATH_TO_DATA_DIR /datasets/clevrer \
  DATA.PATH_PREFIX /datasets/clevrer 
"""

#Extract the z_means in format T x Slots x z_dim
def get_slot_repr(self, x):
    self.loss_E = 0
    self.x_tilde = 0
    b = []
    m = []
    m_tilde_logits = []
    # Initial s_k = 1: shape = (N, 1, H, W)
    shape = list(x.shape)
    shape[1] = 1
    log_s_k = self.x.new_zeros(shape)
    z_mu_k_t = torch.zeros((self.opt.num_slots, x.size(0), self.opt.z_dim)) 
    for k in range(self.opt.num_slots):
        # Derive mask from current scope
        if k != self.opt.num_slots - 1:
            log_alpha_k, alpha_logits_k = self.netAttn(x, log_s_k)
            log_m_k = log_s_k + log_alpha_k
            # Compute next scope
            log_s_k += -alpha_logits_k + log_alpha_k
        else:
            log_m_k = log_s_k
        # Get component and mask reconstruction, as well as the z_k parameters
        m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(x, log_m_k, k == 0)
        z_mu_k_t[k] = z_mu_k
    return z_mu_k_t.permute(1,0,2)


def gen_dataset(cfg, mode, root):
    #Generates one datasets for a certain split. => ResNet50 features pool5
    #When using the generated file must indicate in which index the dataset starts to work
    #Train starts in 0
    #Val starts in 10000
    #Test starts in 15000
    dataset = Clevrer_monet(cfg, mode)
    print("Dataset {} len = {}".format(mode, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    size = len(dataloader)
    batch_size = cfg.TRAIN.BATCH_SIZE

    #h5py slow and fast datasets
    #Slow
    h5_path = os.path.join(root, '{}_monet_features.hdf5'.format(mode))
    f_h5 = h5py.File(h5_path, 'w', libver='latest')
    d_set_h5 = f_h5.create_dataset('data', (size * batch_size, cfg.DATA.NUM_FRAMES, 8, 16),
                            dtype='f4')
    index_set_h5 = f_h5.create_dataset('indexes', (size * batch_size, 1),
                            dtype='f4')

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
            inputs = sampled_batch[0]
            indexes = sampled_batch[1]
            if cfg.NUM_GPUS:
                inputs = inputs.cuda(non_blocking=True)
            cb_sz = inputs.size()
            out = model.get_slot_repr(inputs.view(cb_sz[0]*cb_sz[1], cb_sz[2], cb_sz[3], cb_sz[4]))
            out = out.view(cb_sz[0], cb_sz[1], 8, 16)
            d_set_h5[i_batch * batch_size:(i_batch + 1) * batch_size] = out.detach().cpu().numpy()
            index_set_h5[i_batch * batch_size:(i_batch + 1) * batch_size] = indexes.detach().cpu().numpy().reshape(batch_size,1)
    f_h5.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    logger = logging.get_logger(__name__)
    logging.setup_logging(cfg.OUTPUT_DIR)
    use_gpu = cfg.NUM_GPUS > 0
    opt = Namespace(batch_size=64, beta=0.5, beta1=0.5, checkpoints_dir='./checkpoints', continue_train=True, crop_size=192, dataroot='./datasets/CLEVR_v1.0', dataset_mode='clevr', direction='AtoB', display_env='main', display_freq=400, display_id=1, display_ncols=11, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest', epoch_count=1, gamma=0.5, gan_mode='lsgan', gpu_ids=[], init_gain=0.02, init_type='normal', input_nc=3, isTrain=True, load_iter=0, load_size=64, lr=0.0001, lr_decay_iters=50, lr_policy='linear', max_dataset_size=inf, model='monet', n_layers_D=3, name='clevr_monet', ndf=64, netD='basic', netG='resnet_9blocks', ngf=64, niter=914, niter_decay=0, no_dropout=False, no_flip=False, no_html=False, norm='instance', num_slots=8, num_threads=4, output_nc=3, phase='train', pool_size=50, preprocess='resize_and_crop', print_freq=100, save_by_iter=False, save_epoch_freq=5, save_latest_freq=5000, serial_batches=False, suffix='', update_html_freq=1000, verbose=False, z_dim=16)
    #num_slots = 8
    #z_dim = 16
    #Set model and load checkpoint
    model = create_model(opt)
    model.setup(opt)
    if use_gpu:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
    model.get_slot_repr = get_slot_repr.__get__(model, MONetModel)
    model.eval()

    #Proccess datasets
    root = cfg.DATA.PATH_TO_DATA_DIR
    gen_dataset(cfg, 'train', root)
    gen_dataset(cfg, 'val', root)
