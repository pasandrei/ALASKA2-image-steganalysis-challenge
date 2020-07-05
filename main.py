import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler

from configs.system_device import device

from dataset import ALASKA2Dataset

from src.evaluate import evaluate, benchmark_inference_loop
from src.test import test_

from models.MobileNetV2 import mobilenet_v2
from efficientnet_pytorch import EfficientNet

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.logger import Logger, BenchLogger
from src.unfreezers import MobileNet_Unfreezer, EfficientNet_Unfreezer

from src.train import load_checkpoint, train_loop
from src.misc import generate_mean_std, construct_dataset

try:
    from apex import amp
except ImportError:
    pass


def make_parser():
    parser = ArgumentParser(description="Train Fall Detector")
    parser.add_argument('--data', '-d', type=str, default='dataset', required=False,
                        help='path to test and training data files')
    parser.add_argument('--epochs', '-e', type=int, default=25,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '--bs', type=int, default=64,
                        help='number of examples for each iteration')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true', help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, help='manually set random seed for torch')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--save', action='store_true', help='save model checkpoints')
    parser.add_argument('--mode', type=str, default='training',
                        choices=['training', 'evaluation', 'benchmark-training',
                                 'benchmark-inference', 'testing'])

    # Hyperparameters
    parser.add_argument('--learning-rate', '--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum argument for SGD optimizer')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0.0001,
                        help='momentum argument for SGD optimizer')

    parser.add_argument('--backbone', type=str, default='mobilenetv2',
                        choices=['mobilenetv2', 'efficientnet-b0'])
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    return parser


def train(train_loop_func, args, logger):
    args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(10000)

    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)

    # Setup data, defaults

    train, test = construct_dataset(args.data)
    random.shuffle(train)

    train = train[:len(train) // 5]
    split_position = int(len(train) * 0.90)

    train_dataset = ALASKA2Dataset(train[:split_position], root_dir=args.data, augmented=True)
    val_dataset = ALASKA2Dataset(train[split_position:], root_dir=args.data, augmented=False)
    test_dataset = ALASKA2Dataset(test, root_dir=args.data, augmented=False)

    print(len(train_dataset))
    print(len(val_dataset))

    nr_images_in_train = len(train_dataset)
    # nr_images_in_train = 256 * 20

    train_sampler = SubsetRandomSampler([i for i in range(nr_images_in_train)])
    # train_sampler = WeightedRandomSampler(train_dataset.get_weights(), nr_images_in_train)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, drop_last=False,
                                num_workers=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False,
                                 num_workers=4, shuffle=False)

    # model = mobilenet_v2(pretrained=True)
    #
    # model.classifier = nn.Sequential(
    #     nn.Dropout(0.2),
    #     nn.Linear(model.last_channel, 4),
    # )

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)

    # cnt = 0
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # return

    mean, std = generate_mean_std(amp=args.amp)

    model = model.to(device)
    # args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
    #                             momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=1,
    #                                                        verbose=True)

    start_epoch = 1
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            model, optimizer, scheduler, start_epoch = load_checkpoint(
                args.checkpoint, model, optimizer, scheduler)

            start_epoch += 1  # this is because the epoch saved is the previous epoch
        else:
            print('Provided checkpoint is not path to a file')
            return

    if args.mode == 'evaluation':
        acc = evaluate(model, val_dataloader, args, mean, std)

        print('Model precision {} mAP'.format(acc))
        return
    elif args.mode == 'testing':
        test_(model, test_dataloader, args, mean, std)
        return
    elif args.mode == 'benchmark-inference':
        benchmark_inference_loop(model, val_dataloader, args, mean, std, logger)
        return

    loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.CrossEntropyLoss()

    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # for parameters in model.classifier.parameters():
    #     parameters.requires_grad = True

    # backbone_freezer = MobileNet_Unfreezer([1, 1, 2, 3, 4, 5, 8, 10, 12])
    # backbone_freezer = EfficientNet_Unfreezer([1, 2, 5, 8, 10, 14, 18],
    #                                           [0.97, 0.75, 0.6, 0.4, 0.2, 0.1, 0.0])

    for epoch in range(start_epoch, args.epochs + 1):
        print("-----------------------")
        print("Epoch {} of {}".format(epoch, args.epochs))

        # backbone_freezer.step(epoch, model)

        print("Total number of parameters trained this epoch: ",
              sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if
                  p.requires_grad))

        # for param in model.parameters():
        #     print(param.requires_grad)

        avg_loss = train_loop_func(model, loss_function, optimizer, train_dataloader, None, args,
                                   mean, std, epoch)

        # logger.update_epoch_time(epoch, end_epoch_time)
        print("saving model...")
        obj = {'epoch': epoch,
               'model': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict()}

        torch.save(obj, f'./saved/{args.backbone}_epoch_{epoch}.pt')

        if epoch % 1 == 0:
            print("Ancepe evaluarea")
            evaluate(model, val_dataloader, args, mean, std, loss_function)
            # test_(model, test_dataloader, args, mean, std)

        scheduler.step()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    os.makedirs('./saved', exist_ok=True)

    if args.mode == 'benchmark-training':
        train_loop_func = benchmark_train_loop
        logger = BenchLogger('Training benchmark')
        args.epochs = 1
    elif args.mode == 'benchmark-inference':
        train_loop_func = benchmark_inference_loop
        logger = BenchLogger('Inference benchmark')
        args.epochs = 1
    else:
        train_loop_func = train_loop
        logger = Logger('Training logger', print_freq=1)

    train(train_loop_func, args, logger)
