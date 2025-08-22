import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import sys
from yacs.config import CfgNode as CfgNode

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
from torchsummary import summary
import torchvision
from torchvision.models import resnet34

from models.Simodel import *
from models.cifar10_models import resnet34 as cifar10_resnet34
import datasets

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='cifar10', type=str,
                    help='name of training dataset'),
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--save-dir', default='victim_out',
                    help='path to store output model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=['cnn32', 'resnet32', 'resnet34','swin_base','mvit_base','convnext_base',\
                             'densenet121','wide_resnet_50','vit_s_16_21k','vit_b_16_dino','vit_t_16_21k',\
                                'vit_b_16_21k','vgg16','resnet152','vit_b_16_21k_MoCo','efficientnet_b0',\
                                    'vit_b_16_1k','resnet101','resnet18','resnet50','vit_l_16_21k',\
                                        'vit_l_16_1k','mobilenet_v2','vit_b_16_clip'],
                    help='model architecture: (default: resnet34)')
parser.add_argument('-lp', '--linear-probing', destination='linear-probing', action='store_true')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-step', default=30, type=int,
                    help='learning rate decay step')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='learning rate decay factor')
parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--sched', type=str, default='steplr', 
                    help='learning rate scheduler')
parser.add_argument('--optim', type=str, default='sgd',
                    help='optimizer')
parser.add_argument('--n-channels', default=3, type=int,
                    help='Number of channels in dataset.')
parser.add_argument('--n-classes', default=10, type=int,
                    help='Number of classes in dataset.')
parser.add_argument('--pretrained_path', type=str, 
                    help='Pretrained weights for initializing the  model')
parser.add_argument('--img_size', type=int, default=32,
                    help='Image Dimensions to be used for training and validation')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

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

def activate_linear_probing(model, linear_keyword):
    """
    Freeze all parameters except the ones specified by linear_keyword
    """
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False

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
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
   
    # create model
    if args.pretrained:
        pretrained_state = torch.load(args.pretrained_path) 
        
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
    print("=> creating model '{}'".format(args.arch))
    
    if args.arch == 'efficientnet_b0':
        # use this model definition for imagenet1k trained weights
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(in_features=1280, out_features=args.n_classes,bias=True)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'classifier.1')
        
    elif args.arch == 'vgg16':
        # use this model definition for imagenet1k trained weights
        from torchvision.models import vgg16
        model = vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(in_features=4096, out_features=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'classifier.6')
    
    elif args.arch == 'mobilenet_v2':
        # use this model definition for imagenet1k trained weights
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(in_features=1280, out_features=args.n_classes,bias=True)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'classifier.1')  

    elif args.arch == 'resnet18':
        import torchvision.models as models
        model = models.resnet18(weights="DEFAULT")  
        model.fc = nn.Linear(in_features=512, out_features=args.n_classes,bias=True)   
        print(model)  
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 
    
    elif args.arch == 'resnet34':
        import torchvision.models as models
        model = models.resnet34(weights="DEFAULT")  
        model.fc = nn.Linear(in_features=512, out_features=args.n_classes,bias=True)   
        print(model)  
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 
    
    elif args.arch == 'resnet50':
        import torchvision.models as models
        model = models.resnet50(weights="DEFAULT")  
        model.fc = nn.Linear(in_features=2048, out_features=args.n_classes,bias=True)   
        print(model)  
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 

    elif args.arch == 'resnet101':
        import torchvision.models as models
        model = models.resnet101(weights="DEFAULT")  
        model.fc = nn.Linear(in_features=2048, out_features=args.n_classes,bias=True)   
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 

    elif args.arch == 'resnet152':
        import torchvision.models as models
        model = models.resnet152(weights="DEFAULT")  
        model.fc = nn.Linear(in_features=2048, out_features=args.n_classes,bias=True)   
        print(model)  
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 

    elif args.arch == 'wide_resnet_50':
        import timm
        model = timm.create_model('wide_resnet50_2', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'fc') 

    elif args.arch == 'densenet121':
        import timm
        model = timm.create_model('densenet121', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'classifier') 

    elif args.arch == 'convnext_base':
        import timm
        model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head.fc') 

    elif args.arch == 'swin_base':
        import timm
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head.fc') 

    elif args.arch == 'mvit_base':
        import timm
        model = timm.create_model('mvitv2_base_cls.fb_inw21k', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head.fc') 

    elif args.arch == 'vit_t_16_21k':
        import timm
        model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head') 

    elif args.arch == 'vit_s_16_21k':
        import timm
        model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True,num_classes=args.n_classes)
        print(model)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head') 
                
    elif args.arch =='vit_b_16_21k':
        import timm
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head') 
    
    elif args.arch =='vit_l_16_21k':
        import timm
        model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        linear_keyword = 'head'
        if args.lp is True:
            activate_linear_probing(model, 'head') 

    elif args.arch =='vit_b_16_21k_1k':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head')

    elif args.arch =='vit_l_16_21k_1k':
        import timm
        model = timm.create_model('vit_large_patch16_224', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head') 

    elif args.arch =='vit_b_16_clip':
        import timm
        model = timm.create_model('vit_base_patch16_clip_224.laion2b_ft_in1k', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head') 

    elif args.arch =='vit_b_16_dino':
        import timm
        model = timm.create_model('vit_base_patch16_224.dino', pretrained=True,num_classes=args.n_classes)
        print(model.state_dict().keys())
        if args.lp is True:
            activate_linear_probing(model, 'head')


    if args.pretrained :
        print("Load pretrained model for initializing the victim")
        model_state = model.state_dict()
        print("Model Dict Keys : -")
        print(model_state.keys())
        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
        model_state.update(pretrained_state)
        print("Pretrained Dict Keys : -")
        print(pretrained_state.keys())
        model.load_state_dict(model_state, strict=True)

    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise AssertionError('invalid choice of optimizer')
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.sched == 'steplr':
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif args.sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
    else:
        raise AssertionError("invalid choice of learning rate scheduler")
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    victim_datasets = datasets.__dict__.keys()
    if args.data not in victim_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(victim_datasets))
    dataset = datasets.__dict__[args.data]

    # load model family
    modelfamily = datasets.dataset_to_modelfamily[args.data]
    print('modelfamily: ', modelfamily)

    # set up a dummy config to pass to dataset loading function
    dummy_cfg = CfgNode()
    dummy_cfg.VICTIM = CfgNode()
    dummy_cfg.VICTIM.DATA_ROOT = args.data_root

    # load dataset splits
    train_transform =  datasets.modelfamily_to_transforms[modelfamily]['train']
    val_transform =  datasets.modelfamily_to_transforms[modelfamily]['test']
    resize_transform = transforms.Resize((args.img_size,args.img_size))
    train_transform = transforms.Compose([train_transform,  # Add the existing transformations
    resize_transform,
    ])
    val_transform = transforms.Compose([
        val_transform,  # Add the existing transformations
        resize_transform,
    ])

    train_dataset = dataset(dummy_cfg, train=True, transform=train_transform)
    val_dataset = dataset(dummy_cfg, train=False, transform=val_transform)    


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        # validate(val_loader, model, criterion, args)
        validate(train_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args.save_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        images = data[0]
        target = data[1]
        # print(target.min(), target.max())
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    progress.display(i + 1)

def validate(val_loader, model, criterion, args):

    # print(len(val_loader))
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(loader):
                images = data[0]
                target = data[1]
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    full_filename = os.path.join(out_dir, filename)
    torch.save(state, full_filename)
    if is_best:
        shutil.copyfile(full_filename, os.path.join( out_dir, 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
