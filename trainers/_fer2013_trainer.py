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
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.radam import RAdam

# from torch.optim import Adam as RAdam
# from torch.optim import SGD as RAdam

from utils.metrics.segment_metrics import eval_metrics
from utils.metrics.metrics import accuracy
from utils.generals import make_batch


EMO_DICT = {0: "ne", 1: "an", 2: "di", 3: "fe", 4: "ha", 5: "sa", 6: "su"}


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class FER2013Trainer(Trainer):
    """for classification task"""

    def __init__(self, model, train_set, val_set, test_set, configs):
        super().__init__()
        print("Start trainer..")
        print(configs)

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
        self._test_set = test_set
        self._model = model(
            in_channels=configs["in_channels"],
            num_classes=configs["num_classes"],
        )

        self._model.fc = nn.Linear(512, 7)
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

            self._test_loader = DataLoader(
                self._test_set,
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
            )
            self._val_loader = DataLoader(
                self._val_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            self._test_loader = DataLoader(
                self._test_set,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )

        # define loss function (criterion) and optimizer
        class_weights = [
            1.02660468,
            9.40661861,
            1.00104606,
            0.56843877,
            0.84912748,
            1.29337298,
            0.82603942,
        ]
        class_weights = torch.FloatTensor(np.array(class_weights))

        if self._configs["weighted_loss"] == 0:
            self._criterion = nn.CrossEntropyLoss().to(self._device)
        else:
            self._criterion = nn.CrossEntropyLoss(class_weights).to(self._device)

        self._optimizer = RAdam(
            params=self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            patience=self._configs["plateau_patience"],
            min_lr=1e-6,
            verbose=True,
        )

        # training info
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        log_dir = os.path.join(
            self._configs["cwd"],
            self._configs["log_dir"],
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )

        self._writer = SummaryWriter(log_dir)
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []
        self._best_loss = 1e9
        self._best_acc = 0
        self._test_acc = 0.0
        self._plateau_count = 0
        self._current_epoch_num = 0

        # for checkpoints
        self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )

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

        i += 1
        self._train_loss.append(train_loss / i)
        self._train_acc.append(train_acc / i)

    def _val(self):
        self._model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._val_loader), total=len(self._val_loader), leave=False
            ):
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                # compute output, measure accuracy and record loss
                outputs = self._model(images)

                loss = self._criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]

                val_loss += loss.item()
                val_acc += acc.item()

            i += 1
            self._val_loss.append(val_loss / i)
            self._val_acc.append(val_acc / i)

    def _calc_acc_on_private_test(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                # TODO: implement augment when predict
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                outputs = self._model(images)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        return test_acc

    def _calc_acc_on_private_test_with_tta(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
        )

        for idx in len(self._test_set):
            image, label = self._test_set[idx]

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                # TODO: implement augment when predict
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                outputs = self._model(images)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        return test_acc

    def _calc_acc_on_private_test(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):

                # TODO: implement augment when predict
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                outputs = self._model(images)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        return test_acc

    def train(self):
        """make a training job"""
        print(self._model)
        while not self._is_stop():
            self._increase_epoch_num()
            self._train()
            self._val()

            self._update_training_state()
            self._logging()

        # training stop
        try:
            state = torch.load(self._checkpoint_path)
            if self._distributed:
                self._model.module.load_state_dict(state["net"])
            else:
                self._model.load_state_dict(state["net"])
            # self._test_acc = self._calc_acc_on_private_test()
            test_acc = self._calc_acc_on_private_test_with_tta()
            self._save_weights()
        except Exception as e:
            print("Testing error when training stop")
            print(e)

        self._writer.add_text(
            "Summary", "Converged after {} epochs".format(self._current_epoch_num)
        )
        self._writer.add_text(
            "Summary",
            "Best validation accuracy: {:.3f}".format(self._current_epoch_num),
        )
        self._writer.add_text(
            "Summary", "Private test accuracy: {:.3f}".format(self._test_acc)
        )
        self._writer.close()

    def _update_training_state(self):
        if self._val_acc[-1] > self._best_acc:
            self._save_weights()
            self._plateau_count = 0
            self._best_acc = self._val_acc[-1]
            self._best_loss = self._val_loss[-1]
        else:
            self._plateau_count += 1

        self._scheduler.step(100 - self._val_acc[-1])

    def _logging(self):
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
            "Accuracy/Train", self._train_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Accuracy/Val", self._val_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Train", self._train_loss[-1], self._current_epoch_num
        )
        self._writer.add_scalar("Loss/Val", self._val_loss[-1], self._current_epoch_num)

        print(message)

    def _is_stop(self):
        """check stop condition"""
        return (
            self._plateau_count > self._max_plateau_count
            or self._current_epoch_num > self._max_epoch_num
        )

    def _increase_epoch_num(self):
        self._current_epoch_num += 1

    def _save_weights(self, test_acc=0.0):
        if self._distributed == 0:
            state_dict = self._model.state_dict()
        else:
            state_dict = self._model.module.state_dict()

        state = {
            **self._configs,
            "net": state_dict,
            "best_loss": self._best_loss,
            "best_acc": self._best_acc,
            "train_losses": self._train_loss,
            "val_loss": self._val_loss,
            "train_acc": self._train_acc,
            "val_acc": self._val_acc,
            "test_acc": self._test_acc,
        }

        torch.save(state, self._checkpoint_path)
