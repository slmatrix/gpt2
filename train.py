"""
@author : bilal siddiqui
@purpose: model train and test.
@invoker: executed as __main__
"""
import os
import sys
import time
import torch
import logging
import lightning as L

from pathlib import Path
from itertools import repeat
from argparse import ArgumentParser

from torch import nn
from torch.backends import cudnn

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from densenet import DenseNet

from utils import load_data
from utils import trainval, test
from utils import save_checkpoint


def accelerator_setup():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_bf16_supported(): args.bf16 = True


def multinode_setup():
    pass

if __name__ == "__main__":
    assert torch.cuda.is_available()

    #LOG INFO
    LOGDIR     = 'log/'
    DATADIR    = 'data/'
    MODELDIR   = 'model/'
    CHECKPOINT = None
    WORKERS    = 4
    MODPRINT   = 100

    #weights & biases

    #PyTorch Lightning
    ACCELERATOR   = "cuda"
    #PRECISION     = "bf16-mixed"
    PRECISION     = "16-mixed"
    DEVICE_CNT    = torch.cuda.device_count()
    PARALLEL_ALGO = "ddp"

    #adamW hyperparameters
    BETA_1        = 9e-1
    BETA_2        = 95e-2
    LR_MAX        = 6e-4
    LR_MIN        = 6e-5
    LR_DECAY      = 1e-1
    WEIGHT_DECAY  = 1e-1
    GRADIENT_CLIP = True

    #dataset
    DATASET      = "openwebtext"
    TEST_RATIO   = 1e2
    TRAIN_RATIO  = 90e-2
    VALID_RATIO  = 10e-2
    BATCH_SIZE   = 64
    WARMUP_ITERS = 2e3
    ITERATIONS   = 6e5    #iterations in 300epochs w/ 64batch

    #architecture hyperparameters

    parser = ArgumentParser(
        description='configure & train any DenseNet.')
    parser.add_argument('--growth', default=12, type=int,
        help='successive increase of kernels per block')
    parser.add_argument('--reduce', default=0.5, type=float,
        help='decreased kernels out of each transition block')
    #parser.add_argument('--depths', default=(3,3,3),
    #    nargs="+", help='num blocks per segment.')
    parser.add_argument('--depths', default=100,
        type=int, help='num blocks per segment')
    parser.add_argument('--widths', default=(16,32,64),
        nargs="+", help='block widths per segment')

    parser.add_argument('--dataset', default="cifar100",
        type=str, help='torch.datasets')
    parser.add_argument('--classes', default=100,
        type=int, help='class count; softmax output')
    parser.add_argument('--trainratio', default=TRAIN_RATIO,
        type=str, help='percentage of train/validate')
    parser.add_argument('--validratio', default=VALID_RATIO,
        type=str, help='percentage of train/validate')
    parser.add_argument('--testratio', default=TEST_RATIO,
        type=str, help='percentage of train/validate')

    parser.add_argument('--accelerator', default=ACCELERATOR,
        type=str, help='cuda, cpu, etc.')
    parser.add_argument('--precision', default=PRECISION,
        type=str, help='resolution in training. e.g. FP16.')
    parser.add_argument('--device_cnt', default=DEVICE_CNT,
        type=int, help='number of devices.')
    parser.add_argument('--parallel_algo',default=PARALLEL_ALGO,
        type=str, help='DistributedDataParallel, DeepSpeed,etc')

    parser.add_argument('--momentum', default=MOMENTUM,
        type=float, help='SGD momentum')
    parser.add_argument('--nesterov', dest="nesterov",
        default=NESTEROV, action='store_true', help="use neste")
    parser.add_argument('--weightdecay',
        default=WEIGHT_DECAY, type=float, help='')
    parser.add_argument('--batchsize',default=BATCH_SIZE,
        type=int, help='train samples per batch.')
    parser.add_argument('--iterations',
        default=ITERATIONS, type=int, help='')
    parser.add_argument('--learningrate', default=LR,
        type=float, help='initial SGD learning rate')
    parser.add_argument('--lr_steps', nargs="+",
        default=LR_STEPS, help="used with MultiStepLR")
    parser.add_argument('--lr_decay', default=LR_DECAY,
        type=float, help="scalar mult, used with MultiStepLR")

    parser.add_argument('--workers', default=WORKERS*DEVICE_CNT,
        type=int, help='num threads for data load.')
    parser.add_argument('--printfreq', default=MODPRINT,
        type=int, help='print stats after every <n> batch')
    parser.add_argument('--logdir', default=LOGDIR, type=str,
        help='path + name of logfile')
    parser.add_argument('--datadir', default=DATADIR, type=str,
        help='path to downloaded dataset.')
    parser.add_argument('--modeldir', default=MODELDIR,type=str,
        help='path for model saving.')
    args = parser.parse_args()

    fabric_params = {"accelerator":args.accelerator,
                     "devices"    :args.device_cnt,
                     "strategy"   :args.distributed_algo,
                     "precision"  :args.precision}
    fabric = L.Fabric(**fabric_params)
    fabric.launch()
    cudnn.benchmark = True

    model_params = {"image_dims"        :3,
                    "num_classes"       :args.classes,
                    "num_layers"        :args.depths,
                    "growth_rate"       :args.growth,
                    "compression_factor":args.reduce}
    model = DenseNet(**model_params)

    optim_params = {"params"      :model.parameters(),
                    "lr"          :args.learningrate,
                    "momentum"    :args.momentum,
                    "nesterov"    :args.nesterov,
                    "weight_decay":args.weightdecay}
    optimizer = SGD(**optim_params)

    sched_params = {"optimizer" :optimizer,
                    "gamma"     :args.lr_decay,
                    "milestones":args.lr_steps}
    scheduler = MultiStepLR(**sched_params)

    testloader               = load_data(args, is_train=False)
    trainloader, validloader = load_data(args, is_train=True)

    model, optimizer = fabric.setup(model, optimizer)
    testloader       = fabric.setup_dataloaders(testloader)
    validloader      = fabric.setup_dataloaders(validloader)
    trainloader      = fabric.setup_dataloaders(trainloader)

    #trainloader restarts after being exhausted. fabric expects
    #PyTorch's dataloader object; so make it generator here.
    #we save the __len__ because loop_loader returns a generator
    trainsize   = len(trainloader)
    trainloader = loop_loader(trainloader)

    logfile   = (f"{args.logdir}{args.dataset}_DenseNet_"
                 f"{args.growth}g_{args.reduce}r_{args.depths}"
                 f".txt")
    modelfile = (f"{args.modeldir}{args.dataset}_DenseNet_"
                 f"{args.growth}g_{args.reduce}r_"
                 f"{args.depths}.zip")

    logging.basicConfig(level=logging.INFO, format='')
    logger = logging.getLogger(logfile)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter('')
    logger.addHandler(fh)

    tic = time.time()
    val_stats = trainval(model       = model,
                         optimizer   = optimizer,
                         scheduler   = scheduler,
                         validloader = validloader,
                         trainloader = trainloader,
                         iterations  = args.iterations,
                         batchsize   = args.batchsize,
                         trainsize   = trainsize,
                         modelfile   = modelfile,
                         device      = device,
                         logger      = logger,
                         printfreq   = args.printfreq)
    tok = time.time()
    test_stats = test(dataloader = testloader,
                      model      = model,
                      logger     = logger,
                      device     = device)

    s= (f"--------------------------------------------------\n"
        f"stats    top1    top5     loss\n"
        f"valid: [{val_stats[0][0]:.2f}%]"
        f"[{val_stats[0][1]:.2f}%][[{val_stats[1]:.4f}]\n"
        f"test : [{test_stats[0][0]:.2f}%]"
        f"[{test_stats[0][1]:.2f}%][{test_stats[1]:.4f}]\n"
        f"time : {(tok-tic)/60:.2f}mins\n"
        f"--------------------------------------------------")
    logger.info(s)
