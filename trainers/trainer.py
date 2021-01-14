"""this class build and run a trainer by a configuration"""
import os
import sys
import shutil
import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.radam import RAdam
from utils.metrics.segment_metrics import eval_metrics
from utils.metrics.metrics import accuracy


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class TeeTrainer(Trainer):
    """for classification task"""

    def __init__(self, model, train_set, val_set, configs):
        super().__init__()
        print("Start trainer..")
        # load config
        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"]
        self._weight_decay = self._configs["weight_decay"]
        self._distributed = self._configs["distributed"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]

        # load dataloader and model
        self._train_set = train_set
        self._val_set = val_set
        self._model = model(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
        )
        self._model.load_state_dict(torch.load("saved/checkpoints/mixed.test")["net"])

        print(self._configs)
        self._model = self._model.to(self._device)

        if self._distributed == 1:
            torch.distributed.init_process_group(backend="nccl")
            self._model = nn.parallel.DistributedDataParallel(
                self._model, find_unused_parameters=True
            )
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=lambda x: np.random.seed(x),
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
        else:
            self._train_loader = DataLoader(
                self._train_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=True,
                # worker_init_fn=lambda x: np.random.seed(x)
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
                # worker_init_fn=lambda x: np.random.seed(x)
            )

        # define loss function (criterion) and optimizer
        # class_weights = torch.FloatTensor(np.array([0.3, 0.7])).to(self._device)
        self._criterion = nn.CrossEntropyLoss().to(self._device)

        self._optimizer = RAdam(
            params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        self._scheduler = ReduceLROnPlateau(
            self._optimizer, patience=self._configs["plateau_patience"], verbose=True
        )

        # training info
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        log_dir = os.path.join(
            self._configs["cwd"],
            self._configs["log_dir"],
            "{}_{}".format(self._configs["model_name"], str(self._start_time)),
        )

        self._writer = SummaryWriter(log_dir)
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []
        self._best_loss = 1e9
        self._best_acc = 0
        self._plateau_count = 0
        self._current_epoch_num = 0

    def reset(self):
        """reset trainer"""
        pass

    def _train(self):
        self._model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, (images, targets) in tqdm(
            enumerate(self._train_loader), total=len(self._train_loader), leave=False
        ):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output, measure accuracy and record loss
            outputs = self._model(images)

            loss = self._criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]
            # acc = eval_metrics(targets, outputs, 2)[0]

            train_loss += loss.item()
            train_acc += acc.item()

            # compute gradient and do SGD step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # log
            if i == 0:
                grid = torchvision.utils.make_grid(images)
                self._writer.add_image("images", grid, 0)
                # self._writer.add_graph(self._model, images)
                # self._writer.close()

            if self._configs["little"] == 1:
                mask = torch.squeeze(outputs, 0)
                mask = mask.detach().cpu().numpy() * 255
                mask = np.transpose(mask, (1, 2, 0)).astype(np.uint8)
                cv2.imwrite(
                    os.path.join("debug", "e{}.png".format(self._current_epoch_num)),
                    mask[..., 1],
                )

        i += 1
        self._train_loss.append(train_loss / i)
        self._train_acc.append(train_acc / i)

    def _val(self):
        self._model.eval()
        val_loss = 0.0
        val_acc = 0.0

        os.system("rm -rf debug/*")
        for i, (images, targets) in tqdm(
            enumerate(self._val_loader), total=len(self._val_loader), leave=False
        ):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # compute output, measure accuracy and record loss
            outputs = self._model(images)

            loss = self._criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]
            # acc = eval_metrics(targets, outputs, 2)[0]

            val_loss += loss.item()
            val_acc += acc.item()

            # debug time
            outputs = torch.squeeze(outputs, dim=0)
            outputs = torch.argmax(outputs, dim=0)
            tmp_image = torch.squeeze(images, dim=0)
            print(tmp_image.shape)
            tmp_image = tmp_image.cpu().numpy()
            cv2.imwrite("debug/{}/{}.png".format(outputs, i), tmp_image)

        i += 1
        self._val_loss.append(val_loss / i)
        self._val_acc.append(val_acc / i)

    def train(self):
        """make a training job"""
        while not self._is_stop():
            self._train()
            self._val()

            self._update_training_state()
            self._logging()
            self._increase_epoch_num()

        self._writer.close()  # be careful with this line of code

    def _update_training_state(self):
        if self._val_acc[-1] > self._best_acc:
            self._save_weights()
            self._plateau_count = 0
            self._best_acc = self._val_acc[-1]
            self._best_loss = self._val_loss[-1]
        else:
            self._plateau_count += 1
        self._scheduler.step(self._val_loss[-1])

    def _logging(self):
        # TODO: save message to log file, tensorboard then
        consume_time = str(datetime.datetime.now() - self._start_time)

        message = "\nE{:03d}  {:.3f}/{:.3f}/{:.3f} {:.3f}/{:.3f}/{:.3f} | p{:02d}  Time {}\n".format(
            self._current_epoch_num,
            self._train_loss[-1],
            self._val_loss[-1],
            self._best_loss,
            self._train_acc[-1],
            self._val_acc[-1],
            self._best_acc,
            self._plateau_count,
            consume_time[:-7],
        )

        self._writer.add_scalar(
            "Accuracy/train", self._train_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Accuracy/val", self._val_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/train", self._train_loss[-1], self._current_epoch_num
        )
        self._writer.add_scalar("Loss/val", self._val_loss[-1], self._current_epoch_num)

        print(message)

    def _is_stop(self):
        """check stop condition"""
        return (
            self._plateau_count > self._max_plateau_count
            or self._current_epoch_num > self._max_epoch_num
        )

    def _increase_epoch_num(self):
        self._current_epoch_num += 1

    def _store_trainer(self):
        """store config, training info and traning result to file"""
        pass

    def _save_weights(self):
        """save checkpoint"""
        if self._distributed == 0:
            state_dict = self._model.state_dict()
        else:
            state_dict = self._model.module.state_dict()
        state = {
            **self._configs,
            "net": state_dict,
            "best_loss": self._best_loss,
            "best_acc": self._best_acc,
        }

        checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(state, os.path.join(checkpoint_dir, self._configs["model_name"]))
