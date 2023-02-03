import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pprint
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from datasets.cifar10 import CIFAR10_LT
from datasets.cifar100 import CIFAR100_LT
from datasets.imagenet import ImageNet_LT

from models import resnet
from models import resnet_cifar

from utils import config, update_config, create_logger
from utils import AverageMeter, ProgressMeter
from utils import accuracy, calibration

from methods import mixup_data, mixup_criterion


def parse_args():
    parser = argparse.ArgumentParser(description='MiSLAS training (Stage-1)')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()
    logger, model_dir = create_logger(config, args.cfg)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + str(config))

    if config.deterministic:
        seed = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, logger))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config, logger, model_dir)


def main_worker(gpu, ngpus_per_node, config, logger, model_dir):
    global best_acc1
    config.gpu = gpu
#     start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)

    if config.dataset == 'cifar10' or config.dataset == 'cifar100':
        model = getattr(resnet_cifar, config.backbone)()
        classifier1 = getattr(resnet_cifar, 'Classifier')(feat_in=config.feat_size, num_classes=config.num_classes)
        classifier2 = getattr(resnet_cifar, 'Classifier')(feat_in=config.feat_size, num_classes=config.num_classes)
    
    elif config.dataset == 'imagenet' or config.dataset == 'ina2018':
        model = getattr(resnet, config.backbone)()
        classifier1 = getattr(resnet, 'Classifier')(feat_in=config.feat_size, num_classes=config.num_classes)
        classifier2 = getattr(resnet, 'Classifier')(feat_in=config.feat_size, num_classes=config.num_classes)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            classifier1.cuda(config.gpu)
            classifier2.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
            classifier1 = torch.nn.parallel.DistributedDataParallel(classifier1, device_ids=[config.gpu])
            classifier2 = torch.nn.parallel.DistributedDataParallel(classifier2, device_ids=[config.gpu])
        else:
            model.cuda()
            classifier1.cuda()
            classifier2.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier1 = torch.nn.parallel.DistributedDataParallel(classifier1)
            classifier2 = torch.nn.parallel.DistributedDataParallel(classifier2)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        classifier1 = classifier1.cuda(config.gpu)
        classifier2 = classifier2.cuda(config.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        classifier1 = torch.nn.DataParallel(classifier1).cuda()
        classifier2 = torch.nn.DataParallel(classifier2).cuda()

    # optionally resume from a checkpoint
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            if config.gpu is None:
                checkpoint = torch.load(config.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(config.gpu)
                checkpoint = torch.load(config.resume, map_location=loc)
            # config.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if config.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(config.gpu)
            model.load_state_dict(checkpoint['state_dict_model'])
            classifier1.load_state_dict(checkpoint['state_dict_classifier1'])
            classifier2.load_state_dict(checkpoint['state_dict_classifier2'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    # Data loading code
    if config.dataset == 'cifar10':
        dataset = CIFAR10_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                             batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'cifar100':
        dataset = CIFAR100_LT(config.distributed, root=config.data_path, imb_factor=config.imb_factor,
                              batch_size=config.batch_size, num_works=config.workers)

    elif config.dataset == 'imagenet':
        dataset = ImageNet_LT(config.distributed, root=config.data_path,
                              batch_size=config.batch_size, num_works=config.workers)

    train_loader = dataset.train_instance
    balance_loader = dataset.train_balance
    val_loader = dataset.eval
    if config.distributed:
        train_sampler = dataset.dist_sampler

    if config.balance_ratio is None or config.balance_ratio == 1.0:
        pass
    elif config.balance_ratio == 0.0:
        balance_loader = dataset.train_instance
    else:
        balance_loader = dataset.get_weighted_loader(weighted_alpha=config.balance_ratio)

    cls_num_list = train_loader.dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)

    optimizer = torch.optim.SGD([{"params": model.parameters()},
                                {"params": classifier1.parameters()},
                                {"params": classifier2.parameters()}], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    for epoch in range(config.num_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)

        # define loss function (criterion) and optimizer
        if config.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=None).cuda(config.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, balance_loader, model, classifier1, classifier2, criterion, optimizer, epoch, config, logger)

        # evaluate on validation set
        is_best = validate(val_loader, model, classifier1, classifier2, criterion, config, logger)

        # save checkpoint
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed
                                                      and config.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_model': model.state_dict(),
                'state_dict_classifier1': classifier1.state_dict(),
                'state_dict_classifier2': classifier2.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, model_dir)


def train(train_loader, balance_loader, model, classifier1, classifier2, criterion, optimizer, epoch, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    classifier1.train()
    classifier2.train()

    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)

    fit_num = 0
    tot_num = 0
    back_images = torch.Tensor([]).cuda(config.gpu)
    back_masks = torch.Tensor([]).cuda(config.gpu)

    balance_loader_iter = iter(balance_loader)
    
    end = time.time()
    for i, (input1, target1) in enumerate(train_loader):
        if i > end_steps:
            break

        input2, target2 = next(balance_loader_iter)
        input2 = input2[:input1.shape[0]]
        target2 = target2[:target1.shape[0]]

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input1 = input1.cuda(config.gpu, non_blocking=True)
            target1 = target1.cuda(config.gpu, non_blocking=True)
            input2 = input2.cuda(config.gpu, non_blocking=True)
            target2 = target2.cuda(config.gpu, non_blocking=True)
        
        # separate background
        if config.gpu is not None:
            target_layer = model.last_layer
            fc_layer = classifier1.weight
        else:
            target_layer = model.module.last_layer
            fc_layer = classifier1.module.weight
        mask, logit = get_background_mask(model, classifier1, input1, target1,
                                          target_layer=target_layer, fc_layer=fc_layer)
        prob = F.softmax(logit, dim=1)
        fit = (prob[target1>=0, target1] >= config.fit_thresh)
        back_images = torch.cat([back_images, input1[fit]], dim=0)[-config.bank_size:]
        back_masks = torch.cat([back_masks, mask[fit]], dim=0)[-config.bank_size:]
        fit_num += sum(fit).item()
        tot_num += len(fit)

        if back_images.shape[0] >= input1.shape[0] and epoch >= config.start_aug and epoch < config.num_epochs - config.end_aug:
            perm = np.random.permutation(back_images.shape[0])
            aug_images, aug_masks = back_images[perm][:input1.shape[0]], back_masks[perm][:input1.shape[0]]
            # generate mixed sample
            lam = np.random.uniform(config.a, config.b)
            input2 = lam * aug_masks * aug_images + input2 * (1. - lam * aug_masks)
            # compute output
            feat2 = model(input2)
            output2 = classifier2(feat2)
            loss2 = criterion(output2, target2)
        else:
            loss2 = 0
        
        if config.mixup is True and epoch < config.num_epochs - config.end_aug:
            input1, targets_a, targets_b, lam = mixup_data(input1, target1, alpha=config.alpha)
            feat = model(input1)
            output1 = classifier1(feat)
            loss = mixup_criterion(criterion, output1, targets_a, targets_b, lam)
        else:
            feat = model(input1)
            output1 = classifier1(feat)
            loss = criterion(output1, target1)

        loss = loss + loss2

        acc1, acc5 = accuracy(output1, target1, topk=(1, 5))
        losses.update(loss.item(), input1.size(0))
        top1.update(acc1[0], input1.size(0))
        top5.update(acc5[0], input1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i, logger)
    
    logger.info('Fit num: {0}/{1}'.format(fit_num, tot_num))


feat_map_global = None
grad_map_global = None

def _hook_a(module, input, output):
    global feat_map_global
    feat_map_global[output.device.index] = output

def _hook_g(module, grad_in, grad_out):
    global grad_map_global
    grad_map_global[grad_out[0].device.index] = grad_out[0]

def get_background_mask(model, classifier, input, target, target_layer, fc_layer, mode='GradCAM'):
    hook_a = target_layer.register_forward_hook(_hook_a)
    hook_g = target_layer.register_full_backward_hook(_hook_g)

    training_mode = model.training
    model.eval()
    classifier.eval()

    global feat_map_global
    global grad_map_global
    feat_map_global = {}
    grad_map_global = {}

    feat = model(input)
    output = classifier(feat)
    loss = output[target>=0, target].sum()
    model.zero_grad()
    classifier.zero_grad()
    loss.backward(retain_graph=False)

    hook_a.remove()
    hook_g.remove()

    if isinstance(model, torch.nn.DataParallel):
        feat_map = []
        grad_map = []
        for i in model.device_ids:
            if i in feat_map_global.keys():
                feat_map.append(feat_map_global[i].cuda(config.gpu))
                grad_map.append(grad_map_global[i].cuda(config.gpu))
        feat_map = torch.cat(feat_map)
        grad_map = torch.cat(grad_map)
    else:
        device_id = input.device.index
        feat_map = feat_map_global[device_id]
        grad_map = grad_map_global[device_id]

    with torch.no_grad():
        if mode == 'CAM':
            weights = fc_layer[target].unsqueeze(-1).unsqueeze(-1)
            cam = (weights * feat_map).sum(dim=1, keepdim=True)
        elif mode == 'GradCAM':
            weights = grad_map.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feat_map).sum(dim=1, keepdim=True)
            cam = F.relu(cam, inplace=True)
    
    def _normalize(x):
        x.sub_(x.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        x.div_(x.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
    _normalize(cam)

    input_h, input_w = input.shape[-2], input.shape[-1]
    resized_cam = F.interpolate(cam, size=(input_h, input_w), mode='bicubic', align_corners=False)
    resized_cam = resized_cam.clamp(0, 1)
    mask = (1 - resized_cam) ** 2

    model.train(training_mode)
    classifier.train(training_mode)
    return mask, output.detach()


class AccMeter:
    def __init__(self):
        self.top1 = AverageMeter('Acc@1', ':6.3f')
        self.top5 = AverageMeter('Acc@5', ':6.3f')

        self.class_num = torch.zeros(config.num_classes).cuda(config.gpu)
        self.correct = torch.zeros(config.num_classes).cuda(config.gpu)
        
        self.confidence = np.array([])
        self.pred_class = np.array([])
        self.true_class = np.array([])

    def update(self, output, target, is_prob=False):
        if not is_prob:
            output = torch.softmax(output, dim=1)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.top1.update(acc1[0], target.size(0))
        self.top5.update(acc5[0], target.size(0))

        _, predicted = output.max(1)
        target_one_hot = F.one_hot(target, config.num_classes)
        predict_one_hot = F.one_hot(predicted, config.num_classes)
        self.class_num = self.class_num + target_one_hot.sum(dim=0).to(torch.float)
        self.correct = self.correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

        confidence_part, pred_class_part = torch.max(output, dim=1)
        self.confidence = np.append(self.confidence, confidence_part.cpu().numpy())
        self.pred_class = np.append(self.pred_class, pred_class_part.cpu().numpy())
        self.true_class = np.append(self.true_class, target.cpu().numpy())

    def get_shot_acc(self):
        acc_classes = self.correct / self.class_num
        acc_classes = torch.cat([acc_classes, acc_classes[:1]]) # for SVHN
        head_acc = acc_classes[config.head_class_idx[0]:config.head_class_idx[1]].mean() * 100
        med_acc = acc_classes[config.med_class_idx[0]:config.med_class_idx[1]].mean() * 100
        tail_acc = acc_classes[config.tail_class_idx[0]:config.tail_class_idx[1]].mean() * 100
        return head_acc, med_acc, tail_acc

    def get_cal(self):
        cal = calibration(self.true_class, self.pred_class, self.confidence, num_bins=15)
        return cal


best_acc1 = defaultdict(float)

def validate(val_loader, model, classifier1, classifier2, criterion, config, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_meter = {
        'classifier1': AccMeter(),
        'classifier2': AccMeter(),
        'ensemble': AccMeter()}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, acc_meter['classifier1'].top1, acc_meter['classifier1'].top5],
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()
    classifier1.eval()
    classifier2.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if config.gpu is not None:
                input = input.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            feat = model(input)
            output1 = classifier1(feat)
            output2 = classifier2(feat)
            output_ensemble = (output1 + output2) / 2

            # measure accuracy and record loss
            acc_meter['classifier1'].update(output1, target)
            acc_meter['classifier2'].update(output2, target)
            acc_meter['ensemble'].update(output_ensemble, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i, logger)

        global best_acc1
        is_classifier1_best = False
        is_classifier2_best = False

        for name in acc_meter.keys():
            entry = acc_meter[name]

            acc1, acc5 = entry.top1.avg, entry.top5.avg
            head_acc, med_acc, tail_acc = entry.get_shot_acc()
            
            # remember best acc@1
            is_best = acc1 > best_acc1[name]
            if is_best:
                best_acc1[name] = acc1
                if name == 'classifier1':
                    is_classifier1_best = True
                elif name == 'classifier2':
                    is_classifier2_best = True
            
            logger.info(('* ({name})  Acc@1 {acc1:.3f}  HAcc {head_acc:.3f}  MAcc {med_acc:.3f}  TAcc {tail_acc:.3f}  '
                         '(Best Acc@1 {best_acc1:.3f}).').format(
                             name=name, acc1=acc1, acc5=acc5, head_acc=head_acc, med_acc=med_acc, tail_acc=tail_acc,
                             best_acc1=best_acc1[name]))
    
    return is_classifier1_best


def save_checkpoint(state, is_best, model_dir):
    filename = model_dir + '/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate"""
    if config.cos:
        lr_min = 0
        lr_max = config.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / config.num_epochs * 3.1415926535))
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = config.lr * epoch / 5
        elif epoch > 180:
            lr = config.lr * 0.01
        elif epoch > 160:
            lr = config.lr * 0.1
        else:
            lr = config.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
