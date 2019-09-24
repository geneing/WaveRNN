import argparse
import math
import os
import pickle
import shutil
import sys
import traceback

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import MyDataset
from distribute import *
from models.wavernn import Model
from utils.audio import AudioProcessor
from utils.display import *
from utils.distribution import discretized_mix_logistic_loss, gaussian_loss
from utils.generic_utils import (check_update, count_parameters, load_config,
                                 remove_experiment_folder, save_checkpoint,
                                 save_best_model)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda, flush=True)
print(" > Number of GPUs: ", num_gpus, flush=True)


def setup_loader(is_val=False):
    global train_ids
    dataset = MyDataset(
        test_ids if is_val else train_ids,
        DATA_PATH,
        CONFIG.mel_len,
        ap.hop_length,
        CONFIG.mode,
        CONFIG.pad,
        ap,
        is_val,
    )
    sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    loader = DataLoader(
        dataset,
        collate_fn=dataset.collate,
        batch_size=CONFIG.batch_size,
        num_workers=0,
        # shuffle=True,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    return loader


def find_lr(model, optimizer, criterion, batch_size, args, init_lr=1e-7, end_lr=1., beta=0.98):
    """ from https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html """
    global CONFIG
    global train_ids
    # create train loader
    data_loader = setup_loader(False)
    num_iter = len(data_loader) - 1
    coeff = (end_lr / init_lr) ** (1 / num_iter)
    lr = init_lr
    for p in optimizer.param_groups:
        p["lr"] = lr
    best_loss = float('inf')
    avg_loss = 0.0
    losses = []
    log_lrs = []
    start = time.time()
    # train loop
    print(" > Training", flush=True)
    model.train()
    for i, (x, m, y) in enumerate(data_loader):
        if use_cuda:
            x, m, y = x.cuda(), m.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x, m)
        # y_hat = y_hat.transpose(1, 2)
        if type(model.mode) == int :
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        else:
            y = y.float()
        y = y.unsqueeze(-1)
        loss = criterion(y_hat, y)
        # compute smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(i + 1))
        # stop if the loss is exploding
        if i > 0 and smoothed_loss > 100 * best_loss:
            break
        # Record the best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do optimizer step
        loss.backward()
        optimizer.step()
        speed = (i + 1) / (time.time() - start)
        if i % CONFIG.print_step == 0:
            print(
                " | > Epoch: {}/{} -- Batch: {}/{} -- Loss: {:.3f}"
                " -- Speed: {:.2f} steps/sec -- lr: {}".format(
                    1, 1, i + 1, num_iter, avg_loss, speed, lr
                ), flush=True
            )
        # Update the lr for the next step
        lr *= coeff
        optimizer.param_groups[0]['lr'] = lr
    # plot results
    if args.rank == 0:
        plt.plot(log_lrs, losses)
        print(f"{VIS_PATH}/find_lr.png")
        plt.savefig(f"{VIS_PATH}/find_lr.png")
        plt.close


def main(args):
    global train_ids
    global test_ids

    # read meta data
    with open(f"{DATA_PATH}/dataset_ids.pkl", "rb") as f:
        train_ids = pickle.load(f)

    # pick validation set
    test_ids = train_ids[-10:]
    test_id = train_ids[4]
    train_ids = train_ids[:-10]

    # create the model
    assert np.prod(CONFIG.upsample_factors) == ap.hop_length, ap.hop_length
    model = Model(
        rnn_dims=512,
        fc_dims=512,
        mode=CONFIG.mode,
        mulaw=CONFIG.mulaw,
        pad=CONFIG.pad,
        upsample_factors=CONFIG.upsample_factors,
        feat_dims=80,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap.hop_length,
        sample_rate=ap.sample_rate,
    ).cuda()

    num_parameters = count_parameters(model)
    print(" > Number of model parameters: {}".format(num_parameters), flush=True)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.lr)

    # restore any checkpoint
    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            model.load_state_dict(checkpoint["model"])
            # TODO: fix resetting restored optimizer lr 
            # optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            model_dict = model.state_dict()
            # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v for k, v in checkpoint["model"].items() if k in model_dict
            }
            # 2. filter out different size layers
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if v.numel() == model_dict[k].numel()
            }
            # 3. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 4. load the new state dict
            model.load_state_dict(model_dict)
            print(
                " | > {} / {} layers are initialized".format(
                    len(pretrained_dict), len(model_dict)
                )
            )

    # DISTRIBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define train functions
    if CONFIG.mode == 'mold':
        criterion = discretized_mix_logistic_loss
    elif CONFIG.mode == 'gauss':
        criterion = gaussian_loss
    elif type(CONFIG.mode) is int:
        criterion = torch.nn.CrossEntropyLoss()
    model.train()

    # HIT IT!!!
    find_lr(
        model,
        optimizer,
        criterion,
        CONFIG.batch_size,
        args,
        init_lr=args.init_lr,
        end_lr=args.end_lr,
        beta=0.98
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for training."
    )
    parser.add_argument(
        "--restore_path", type=str, default=0, help="path for a model to fine-tune."
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="data path to overwrite config.json."
    )
    parser.add_argument(
        "--output_path", type=str, help="path for training outputs.", default=""
    )
    parser.add_argument(
        "--init_lr", type=float, help="path for training outputs.", default=1e-7
    )
    parser.add_argument(
        "--end_lr", type=float, help="path for training outputs.", default=1
    )
    # DISTRUBUTED
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="DISTRIBUTED: process rank for distributed training.",
    )
    parser.add_argument(
        "--group_id", type=str, default="", help="DISTRIBUTED: process group id."
    )

    args = parser.parse_args()
    CONFIG = load_config(args.config_path)

    if args.data_path != "":
        CONFIG.data_path = args.data_path
    DATA_PATH = CONFIG.data_path

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(
            args.rank,
            num_gpus,
            args.group_id,
            CONFIG.distributed["backend"],
            CONFIG.distributed["url"],
        )

    global ap
    ap = AudioProcessor(**CONFIG.audio)
    mode = CONFIG.mode

    # setup output paths and read configs
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != "":
        CONFIG.data_path = args.data_path

    if args.output_path == "":
        OUT_PATH = os.path.join(_, CONFIG.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == "":
        OUT_PATH = create_experiment_folder(OUT_PATH, CONFIG.model_name)


    if args.rank == 0:
        # set paths
        VIS_PATH = f"{OUT_PATH}/lr_find/"
        shutil.copyfile(args.config_path, os.path.join(OUT_PATH, "config.json"))

        # create paths
        os.makedirs(VIS_PATH, exist_ok=True)

    main(args)
    
