import torch
import lightning as L
import torch.nn.functional as F

from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import FakeData
from torchvision.datasets import ImageFolder

from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import CenterCrop
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip


@torch.no_grad()
def accuracy(result, answer, topk=1):
    r'''
    result (batch_size, class_cnt)
    answer (batch_size)
    '''
    #save the batch size before tensor mangling
    bz = answer.size(0)
    #ignore result values. its indices: (sz,cnt) -> (sz,topk)
    values, indices = result.topk(topk)
    #transpose the k best indice
    result = indices.t()  #(sz,topk) -> (topk, sz)
    
    #repeat same labels topk times to match result's shape
    answer = answer.view(1, -1)       #(sz) -> (1,sz)
    answer = answer.expand_as(result) #(1,sz) -> (topk,sz)

    correct = (result == answer)    #(topk,sz) of bool vals
    correct = correct.flatten()     #(topk*sz) of bool vals
    correct = correct.float()       #(topk*sz) of 1s or 0s
    correct = correct.sum()         #counts 1s (correct guesses)
    correct = correct.mul_(100/bz)  #convert into percentage

    return correct.item()


class AverageMeter(object):
    """
    @purpose: track average, sum, count, and most recent value.
    @params : N/A.
    @return : state. maintains mean, sum, count, and newest
              value.
    """
    def __init__(self):
        self.reset();

    def reset(self):
        self.val   = 0;
        self.avg   = 0;
        self.sum   = 0;
        self.count = 0;

    def update(self, val, n=1):
        self.val    = val;
        self.sum   += val * n;
        self.count += n;
        self.avg    = self.sum / self.count;


class CachedDataset(torch.utils.data.Dataset):
    def __init__():
        self.path = path
        self.preload_images = preload_images

        self.data = json.load(open(path, 'r'))
        self.keys = list(self.data.keys())

        if self.preload_images:
            self.images = []
            for k in self.keys:
                self.images.append(Image.open(k).convert("RBG"))

    def __len__(self,):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.preload_images:
            image = self.images[idx]
        else:
            image = Image.open(self.keys[idx]).convert("RGB")


def save_checkpoint(loss,
                    acc,
                    epoch,
                    fabric,
                    model,
                    optimizer,
                    scheduler,
                    pathname):
    """
    @purpose: save whole model, and its hyperparameters.
    @params:
      - epoch    : epochs trained.
      - acc      : best top-1 accuracy so far.
      - loss     : best loss so far.
      - model    : model parameters.
      - optimizer: optimizer parameters.
      - scheduler: scheduler parameters.
      - pathname : saving directory path, suffixed with the
                   hyperparameters.
    #return: pickle serialized object, saved on disk.
    """
    state = {'loss'        : loss                  ,
             'acc'         : acc                   ,
             'epoch'       : epoch                 ,
             'model'       : model.state_dict()    ,
             'optimizer'   : optimizer.state_dict(),
             'scheduler'   : scheduler}
    extension = "" if ".zip" in pathname else "_best.zip"
    fabric.save(state, pathname + extension)


def transform(args,
              train:bool):
    if "cifar" in args.dataset.lower():
        if args.dataset.lower() == "cifar10":
            std  = [0.247,0.244,0.262]
            mean = [0.491,0.482,0.447]
        else:
            std  = [0.268,0.257,0.276]
            mean = [0.507,0.487,0.441]
        if train:
            transform = Compose([ToTensor(),
                                 RandomHorizontalFlip(),
                                 RandomCrop(32, padding=4),
                                 Normalize(std=std,mean=mean)])
        else:
            transform = Compose([ToTensor(),
                                 Normalize(std=std,mean=mean)])
    elif "imagenet1k" in args.dataset.lower():
        std  = [0.229,0.224,0.225]
        mean = [0.485,0.456,0.406]
        if train:
            transform = Compose([ToTensor(),
                         RandomHorizontalFlip(),
                         RandomResizedCrop(224,antialias=True),
                         Normalize(std=std,mean=mean)])
        else:
            transform = Compose([ToTensor(),
                                 Resize(256),
                                 CenterCrop(224),
                                 Normalize(std=std,mean=mean)])
    elif args.dataset.lower() == "fake":
        std  = [0.229,0.224,0.225]
        mean = [0.485,0.456,0.406]
        if train:
            transform = Compose([ToTensor(),
                         RandomHorizontalFlip(),
                         RandomResizedCrop(224,antialias=True),
                         Normalize(std=std,mean=mean)])
        else:
            transform = Compose([ToTensor(),
                                 Resize(256),
                                 CenterCrop(224),
                                 Normalize(std=std,mean=mean)])
    else:
        assert False

    return transform


def load_data(args,
              is_train:bool):
    if args.dataset.lower() == "cifar10":
        if is_train:
            dataindex = CIFAR10(root=args.datadir,
                                train=True,
                                download=True,
                                transform=transform(args,True))
            trainindex, validindex = random_split(dataindex, \
                [args.trainratio, args.validratio])
        else:
            testindex = CIFAR10(root=args.datadir,
                                train=False,
                                download=True,
                                transform=transform(args,False))
    elif args.dataset.lower() == "cifar100":
        if is_train:
            dataindex = CIFAR100(root=args.datadir,
                                 train=True,
                                 download=True,
                                 transform=transform(args,True))
            trainindex, validindex = random_split(dataindex, \
                [args.trainratio, args.validratio])
        else:
            testindex= CIFAR100(root=args.datadir,
                                train=False,
                                download=True,
                                transform=transform(args,False))
    elif args.dataset.lower() == "imagenet1k":
        if is_train:
            trainindex = ImageFolder(args.datadir + "train")
            validindex = ImageFolder(args.datadir + "val")
        else:
            testindex = ImageFolder(args.datadir + "val")
            #testindex = ImageFolder(args.datadir + "test")
    elif args.dataset.lower() == "fake":
        if is_train:
            trainindex = FakeData(1281167, (3, 224, 224), 1000)
            validindex = FakeData(50000,   (3, 224, 224), 1000)
        else:
            testindex = FakeData(100000, (3, 224, 224), 1000)
    else:
        assert False
 
    if is_train:
        trainindex.transform = transform(args,True)
        validindex.transform = transform(args,False)
        dataloader = (DataLoader(trainindex,
                                 shuffle=True,
                                 batch_size=args.batchsize,
                                 num_workers=args.workers,
                                 persistent_workers=True,
                                 pin_memory=True),
                      DataLoader(validindex,
                                 shuffle=False,
                                 batch_size=args.batchsize,
                                 num_workers=args.workers,
                                 persistent_workers=True,
                                 pin_memory=True))
    else:
        testindex.transform = transform(args,False)
        dataloader = DataLoader(testindex,
                                shuffle=False,
                                batch_size=args.batchsize,
                                num_workers=args.workers,
                                persistent_workers=True,
                                pin_memory=True)

    return dataloader


def trainval(model      ,
             fabric     ,
             optimizer  ,
             scheduler  ,
             validloader,
             trainloader,
             batchsize  ,
             trainsize  ,
             iterations ,
             modelfile  ,
             device     ,
             logger     ,
             printfreq):
    """
    @purpose: optimize weights over train data.
    @params :
      - trainloader: trainset image tensors.
      - model      : initialized model.
      - criterion  : loss fn.
      - optimizer  : model update algo.
    """
    best_acc  = float("-inf")
    best_loss = float("inf")
    stalled   = 0
    top1      = AverageMeter()
    top5      = AverageMeter()
    losses    = AverageMeter()

    model.train()
    logger.info("Train:")
    for batchidx in range(1, iterations+1):
        #zero gradient buffers batch
        optimizer.zero_grad()
        #send batch to compute device [CPU, TPU, etc]
        images, targets = next(iter(trainloader))

        #get predictions
        #results = model(images.half())
        results = model(images)
        #calc loss using predictions & truth
        loss    = F.cross_entropy(results, targets)
        #backprop loss gradient through model
        fabric.backward(loss)
        #update the model weights
        optimizer.step()
        #reduce learning rate (if scheduler conditions are met)
        scheduler.step()

        t1 = accuracy(results, targets, 1)
        t5 = accuracy(results, targets, 5)
        top1.update(t1, images.size(0))
        top5.update(t5, images.size(0))
        losses.update(loss.item(), images.size(0))

        if batchidx%printfreq==0:
            epoch = batchidx//trainsize
            s = (f'\titer [{epoch}][{batchidx}/{iterations}]'
                 f'\tloss [{losses.val:.4f} ({losses.avg:.4f})]'
                 f'\ttop1 [{top1.val:.2f} ({top1.avg:.2f})]'
                 f'\ttop5 [{top5.val:.2f} ({top5.avg:.2f})]')
            logger.info(s)

        if batchidx%trainsize==0:
            vaccuracy, vloss = test(dataloader = validloader,
                                    model      = model,
                                    criterion  = criterion,
                                    logger     = logger,
                                    device     = device)
            logger.info("Valid:")
            s = (f'\tloss ({vloss:.4f})'
                 f'\ttop1 ({vaccuracy[0]:.2f})'
                 f'\ttop5 ({vaccuracy[1]:.2f})')
            logger.info(s)
            logger.info("Train:")

            isbest  = vloss < best_loss
            if isbest:
                stalled  = 0
                best_acc = vaccuracy
                save_checkpoint(vloss,
                                vaccuracy,
                                batchidx//trainsize,
                                fabric,
                                model,
                                optimizer,
                                scheduler,
                                modelfile)
    return vaccuracy, vloss


@torch.inference_mode()
def test(dataloader,
         model,
         logger,
         device):
    """
    @purpose: loss and accuracy on testset. gradients frozen.
    @params :
      - (torch.utils) testloader : testset image tensors.
      - (torch.nn)    model      : initialized model.
      - (torch.nn)    criterion  : loss fn.
      - (torch.optim) optimizer  : model update algo.
      - (int)         epoch      :
    """
    top1   = AverageMeter()
    top5   = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for idx, (images, targets) in enumerate(dataloader):
        results = model(images)
        loss    = F.cross_entropy(results, targets)

        t1 = accuracy(results, targets, 1)
        t5 = accuracy(results, targets, 5)
        top1.update(t1, images.size(0))
        top5.update(t5, images.size(0))
        losses.update(loss.item(), images.size(0))
    model.train()
    return (top1.avg,top5.avg), losses.avg
