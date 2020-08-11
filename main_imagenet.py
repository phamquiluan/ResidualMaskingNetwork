import argparse
import os
import sys
import random
import shutil
import glob
import time
import datetime
import warnings
import logging
import subprocess


IS_SM = False

# =================================================
# setting logger
if os.getcwd() != "/opt/ml/code":
    logging_format = "\033[02m \033[36m[%(asctime)s] [%(levelname)s]\033[0m %(message)s \033[02m <%(name)s, %(funcName)s(): %(lineno)d>\033[0m"
else:
    IS_SM = True
    logging_format = "[%(asctime)s] [%(levelname)s] %(message)s <%(name)s, %(funcName)s(): %(lineno)d>"

datefmt = "%b-%d %H:%M"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    datefmt=datefmt,
    filename="log",
    filemode="w",
)

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter(logging_format, datefmt=datefmt)
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)
# =================================================


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.datasets as datasets

# import torchvision.models as models
import models

from tqdm import tqdm
from natsort import natsorted

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "-d", "--data", metavar="DIR", help="path to dataset", default="/home/"
)
parser.add_argument(
    "-a", "--arch", metavar="ARCH", default="cbam_resnet34", choices=model_names
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

# TODO: edit below for resuming
# parser.add_argument('--resume', default="/opt/ml/checkpoints/checkpoints/", type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=1, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    default=True,
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

if IS_SM:
    imagenet_one_large_zip_path = "/opt/ml/input/data/train/imagenet_one_large_file.zip"

    if not os.path.exists("/opt/ml/input/data/train/imagenet/"):
        logger.info("START UNZIPPING {}".format(imagenet_one_large_zip_path))
        from zipfile import ZipFile

        with ZipFile(imagenet_one_large_zip_path, "r") as zipref:
            zipref.extractall(os.path.dirname(imagenet_one_large_zip_path))
        logger.info("UNZIPPING DONE", os.listdir("/opt/ml/input/data/train"))

    if not os.path.exists("/opt/ml/checkpoints/checkpoints"):
        os.makedirs("/opt/ml/checkpoints/checkpoints", exist_ok=True)


best_acc1 = 0
st = datetime.datetime.now()


def main():
    args = parser.parse_args()

    if IS_SM:
        args.data = "/opt/ml/input/data/train/imagenet/"
    else:
        # FIXME: you should re-specify your path to the imagenet here
        args.data = "/mnt/data/luan/imagenet/"

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # FIXME: current set false for fair
        model = models.__dict__[args.arch](pretrained=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](in_channels=3, num_classes=1000)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint /opt/ml/checkpoints/checkpoints/
    checkpoint_dir = "/opt/ml/checkpoints/checkpoints/"
    checkpoint_list = os.listdir(checkpoint_dir)

    logger.info("Checking checkpoints dir..{}".format(checkpoint_dir))
    logger.info(os.listdir(checkpoint_dir))

    # choose the latest checkpoint
    latest_path_parent = ""
    latest_path = ""
    latest_epoch = -1
    for checkpoint_path in natsorted(
        glob.glob("/opt/ml/checkpoints/checkpoints/*.pth")
    ):
        checkpoint_name = os.path.basename(checkpoint_path)
        logger.info("Found checkpoints {}".format(checkpoint_name))
        epoch_num = int(os.path.splitext(checkpoint_name)[0].split("_")[1])

        if epoch_num > latest_epoch:
            latest_path_parent = latest_path
            latest_path = checkpoint_path
            latest_epoch = epoch_num

    print("> latest epoch is checkpoint '{}'".format(latest_path))

    if latest_path_parent:
        print("=> loading checkpoint '{}'".format(latest_path_parent))
        if args.gpu is None:
            checkpoint = torch.load(latest_path_parent)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(latest_path_parent, map_location=loc)

        args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        if args.gpu is not None:
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                latest_path_parent, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(latest_path_parent))

    # cudnn.benchmark = False

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                state={
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                epoch=epoch,
            )


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":.2f")
    top5 = AverageMeter("Acc@5", ":.2f")
    progress = ProgressMeter(
        len(train_loader), [losses, top1, top5], prefix="Epoch {} ".format(epoch)
    )

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":.2f")
    top5 = AverageMeter("Acc@5", ":.2f")
    progress = ProgressMeter(len(val_loader), [losses, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(state, is_best, epoch):
    checkpoint_name = "resmaskingnet_{}.pth".format(epoch)
    checkpoint_path = "/opt/ml/checkpoints/checkpoints/{}".format(checkpoint_name)
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, "/opt/ml/checkpoints/best_0.pth")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters  # type of meter, avaragemeter :))
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = " - ".join(entries) + " consume {}".format(
            datetime.datetime.now() - st
        )
        # print(" ".join(entries))
        print(message)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    st = datetime.datetime.now()
    try:
        main()
    except Exception as e:
        logger.fatal(e, exc_info=True)

    logger.info("DONE! ESTIMATED RUN TIME: {}".format(datetime.datetime.now() - st))
