# ------------------------------------------------------------------------
# company:zyyl tech shenzhen
# ------------------------------------------------------------------------
# author:kunlei hong
# email:hongkl2014hust@163.com
# ------------------------------------------------------------------------


import argparse
import os
import json
import time
import math
import shutil
import errno
import random

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from ISDA import ISDALoss
from models.resnest import resnest101
from models.full_layer import Full_layer

import autoaugment

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

training_configurations = {
    'resnest': {
        'epochs': 100,
        'batch_size': 32,
        'initial_learning_rate': 0.01,
        'changing_lr': [20, 60],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


class Zyyl_dataset(Dataset):
    def __init__(self, args, flag, image_size=512):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.flag = flag
        self.image_size = image_size
        assert self.flag in [
            'train', 'valid'], 'Only train and valid supported'
        self.thing_classes = eval(self.thing_classes)
        self.img_label = self.get_imgs_labels()
        self.transformer_train, self.transformer_valid = self.get_transformer()

    def __getitem__(self, item):
        image, lbl = self.img_label[item]
        label = self.thing_classes.index(str(lbl))
        img = Image.open(os.path.join(
            '/home/hkl/Projects/Post_treatment_data/data4/%s' % lbl, image)).convert('RGB')
        transformer = eval('self.transformer_%s' % self.flag)
        img = transformer(img)
        pos_x = math.ceil(float(image.split('_')[-2])*100)
        pos_y = math.ceil(float(image.split('_')[-1].replace('.png', ''))*100)
        return img, label, torch.LongTensor([pos_x, pos_y])

    def __len__(self):
        return len(self.img_label)

    def get_imgs_labels(self):
        csv_path = eval('self.{}_path'.format(self.flag))
        pd_data = pd.read_csv(csv_path, names=['img_path', 'label'])
        return list(zip(pd_data['img_path'], pd_data['label']))

    def get_transformer(self):
        normalize = transforms.Normalize(
            std=(0.485, 0.456, 0.406), mean=(0.229, 0.224, 0.225))

        if self.augment == 1:
            print('Normal Augmentation!')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                normalize
            ])

        elif self.augment == 2:
            print('Strong Augmentation!')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=(-5, 5)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])

        elif self.augment == 3:
            print('affare and transform..')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToPILImage(),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAffine(
                    degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.8, 1.5)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize
            ])

        elif self.augment == 4:
            print('Auto Augment')
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToPILImage(),
                autoaugment.ImageNetPolicy(),
                transforms.ToTensor(),
                normalize
            ])

        elif self.augment == 5:
            print('add PNG to JPG augmentation')
            trans1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                normalize])
            trans2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=1),
                normalize])
            trans3 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomRotation(degrees=(-5, 5)),
                normalize])
            trans4 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                normalize])
            trans5 = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                PNG2JPG(40, 90),
                transforms.ToTensor(),
                normalize])

            transform_train = transforms.RandomChoice(
                [trans1, trans1, trans1, trans1, trans1, trans1, trans2, trans3, trans4, trans5])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            normalize
        ])
        return transform_train, transform_test


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""

    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate'] \
                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))


class PNG2JPG(torch.nn.Module):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        super().__init__()

    def forward(self, img):
        return self._png2jpg(img)

    def _png2jpg(self, img):
        num = random.randint(self.low, self.high)
        with BytesIO() as f:
            img.save(f, format='JPEG', quality=num)
            f.seek(0)
            img_jpg = Image.open(f)
            img_jpg.load()
        return img_jpg


def train(train_loader, model, fc, criterion, optimizer, epoch, record_file, loss_file, embedding):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    train_batches_num = len(train_loader)
    ratio = args.lambda_0 * \
        (epoch / (training_configurations[args.model]['epochs']))
    model.train()
    fc.train()
    embedding.train()
    end = time.time()
    for i, (x, target, pos) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        pos = pos.cuda()
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        # compute output
        loss, output = criterion(model, fc, input_var,
                                 target_var, ratio, embedding, pos)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                          epoch, i + 1, train_batches_num, batch_time=batch_time,
                          loss=losses, top1=top1))

            print(string)
            # print(weights)
            fd.write(string + '\n')
            fd.close()

    fd = open(loss_file, 'a+')
    string = ('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                  epoch, i + 1, train_batches_num, batch_time=batch_time,
                  loss=losses, top1=top1))

    print(string)
    fd.write(string + '\n')
    fd.close()


def validate(val_loader, model, fc, criterion, epoch, record_file, loss_file, embedding):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()
    fc.eval()
    embedding.eval()

    end = time.time()
    for i, (input, target, pos) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        pos = pos.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var)
            output = fc(features)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                          epoch, (i + 1), train_batches_num, batch_time=batch_time,
                          loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(loss_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                  epoch, (i + 1), train_batches_num, batch_time=batch_time,
                  loss=losses, top1=top1))
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)

    return top1.ave


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_args_parser():

    parser = argparse.ArgumentParser(
        description='ISDA for zyyl', add_help=False)

    #  args for data
    parser.add_argument('--train_path', default=r'', type=str,
                        help='train data path to be trained')
    parser.add_argument('--valid_path', default=r'', type=str,
                        help='valid data path to be valided')
    parser.add_argument('--thing_classes', default="['21','23','25','902','405','1201','6','3','7','907','801','12','401','4','402','407','701','1305','11','501','8','5','0']", type=str,
                        help='train labels,assume as a list')
    parser.add_argument('--augment', type=int, choices=[1, 2, 3, 4, 5],
                        help='augmentation no.')

    # args for model
    parser.add_argument('--model', default='resnest', type=str,
                        help='deep networks to be trained')

    # args for train
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        help='print frequency (default: 10)')

    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--name', default='zyyl_pretrain', type=str,
                        help='name of experiment')
    parser.add_argument('--no', default='1', type=str,
                        help='index of the experiment (for recording convenience)')
    parser.add_argument('--lambda_0', default=0.5, type=float,
                        help='hyper-patameter_\lambda for ISDA')

    # Cosine learning rate
    parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                        help='whether to use cosine learning rate')
    parser.set_defaults(cos_lr=False)

    parser.add_argument('--pos', dest='pos',
                        action='store_true', help='embedding position')

    return parser


def model_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.kaiming_normal_(
                param, mode='fan_in', nonlinearity='relu')


def main(args):
    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []
    global class_num
    class_num = 23
    record_path = './ISDA_test/' \
                  + '_' + str(args.model) \
                  + '_' + str(args.name) \
                  + '/' + 'no_' + str(args.no)
    record_file = record_path + '/training_process.txt'
    accuracy_file = record_path + '/accuracy_epoch.txt'
    loss_file = record_path + '/loss_epoch.txt'
    check_point = os.path.join(record_path, args.checkpoint)
    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    # data
    train_loader = torch.utils.data.DataLoader(
        Zyyl_dataset(args, flag='train'), batch_size=training_configurations[args.model]['batch_size'], shuffle=True,
        num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        Zyyl_dataset(args, flag='valid'), batch_size=training_configurations[args.model]['batch_size'], shuffle=True,
        num_workers=4, drop_last=True)

    # model
    if args.model == 'resnest':
        model = resnest101()
    fc = Full_layer(int(model.num_classes), class_num)
    if args.pos:
        embedding = nn.Embedding(100, int(model.num_classes) // 2)
    # init
    model_init(fc)
    if args.pos:
        model_init(embedding)
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    isda_criterion = ISDALoss(int(model.num_classes), class_num).cuda()
    # isda_criterion = ISDALoss(int(model.num_classes), class_num)
    ce_criterion = nn.CrossEntropyLoss().cuda()
    # ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                 {'params': fc.parameters()}],
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])
    model = torch.nn.DataParallel(model).cuda()
    fc = torch.nn.DataParallel(fc).cuda()
    if args.pos:
        embedding = torch.nn.DataParallel(embedding).cuda()
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        isda_criterion = checkpoint['isda_criterion']
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        adjust_learning_rate(optimizer, epoch + 1)
        # train for one epoch
        train(train_loader, model, fc, isda_criterion,
              optimizer, epoch, record_file, loss_file, embedding)

        # evaluate on validation set
        prec1 = validate(val_loader, model, fc,
                         ce_criterion, epoch, record_file, loss_file, embedding)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'isda_criterion': isda_criterion,
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))

    print('Best accuracy: ', best_prec1)
    print('Average accuracy', sum(val_acc[len(val_acc) - 10:]) / 10)
    np.savetxt(accuracy_file, np.array(val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'ISDA for zyyl', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
