#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable
'''

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.common.initializer import initializer
from mindspore.ops import stop_gradient
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net
from mindspore import save_checkpoint
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


import os
import time
import argparse
import numpy as np

#import torch.backends.cudnn as cudnn

from data.config import cfg
from pyramidbox import build_net
from layers.modules import MultiBoxLoss
from data.widerface import WIDERDetection, detection_collate


parser = argparse.ArgumentParser(
    description='Pyramidbox face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet',
                    default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
#暂时设置为不用cuda
parser.add_argument('--cuda',
                    default=False, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not args.multigpu:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''
————更改为context.set_context————
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')'''


if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


#train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train')
train_dataset_generator = WIDERDetection()
train_dataset = ds.GeneratorDataset(train_dataset_generator, cfg.FACE.TRAIN_FILE, mode='train')
'''mindspore 没有loader
train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)'''

#val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val')
val_dataset_generator = WIDERDetection()
val_dataset = ds.GeneratorDataset(val_dataset_generator, cfg.FACE.VAL_FILE, mode='val')

'''
val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)'''

min_loss = np.inf


def train():
    iteration = 0
    start_epoch = 0
    step_index = 0
    per_epoch_size = len(train_dataset) // args.batch_size

    pyramidbox = build_net('train', cfg.NUM_CLASSES)
    net = pyramidbox
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)
        iteration = start_epoch * per_epoch_size
    else:
        #vgg_weights = torch.load(args.save_folder + args.basenet)
        vgg_weights = load_checkpoint(args.save_folder + args.basenet)
        print('Load base network....')
        #net.vgg.load_state_dict(vgg_weights)
        load_param_into_net(net.vgg, vgg_weights)
    
    '''找不到相关接口
    if args.cuda:
        if args.multigpu:
            net = torch.nn.DataParallel(pyramidbox)
        net = net.cuda()
        cudnn.benckmark = True'''

    if not args.resume:
        print('Initializing weights...')
        pyramidbox.extras.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_topdown.apply(pyramidbox.weights_init)
        pyramidbox.lfpn_later.apply(pyramidbox.weights_init)
        pyramidbox.cpm.apply(pyramidbox.weights_init)
        pyramidbox.loc_layers.apply(pyramidbox.weights_init)
        pyramidbox.conf_layers.apply(pyramidbox.weights_init)


    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = nn.SGD(net.parameters(),learning_rate=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #criterion1 = MultiBoxLoss(cfg, args.cuda)
    #criterion2 = MultiBoxLoss(cfg, args.cuda, use_head_loss=True)
    print('Loading wider dataset...')
    print('Using the specified args:')
    print(args)
    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        #for batch_idx, (images, face_targets, head_targets) in enumerate(train_loader):
        for batch_idx, (images, face_targets, head_targets) in train_dataset.create_dict_iterator():
            '''if args.cuda:
                images = Variable(images.cuda())
                face_targets = [Variable(ann.cuda(), volatile=True)
                                for ann in face_targets]
                head_targets = [Variable(ann.cuda(), volatile=True)
                                for ann in head_targets]
            else:
                images = Variable(images)
                face_targets = [Variable(ann, volatile=True)
                                for ann in face_targets]
                head_targets = [Variable(ann, volatile=True)
                                for ann in head_targets]'''

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            #out = net(images)
            # backprop
            '''
            optimizer.zero_grad()
            face_loss_l, face_loss_c = criterion1(out, face_targets)
            head_loss_l, head_loss_c = criterion2(out, head_targets)
            loss = face_loss_l + face_loss_c + head_loss_l + head_loss_c
            losses += loss.data[0]
            loss.backward()            
            optimizer.step()
            t1 = time.time()
            face_loss = (face_loss_l + face_loss_c).data[0]
            head_loss = (head_loss_l + head_loss_c).data[0]'''

            crit = loss_fn()
            net_with_criterion = MyWithLossCell(net, crit)

            loss = net_with_criterion(images, face_targets, head_targets)
            losses += loss.data[0]       

            # 定义训练网络
            train_net = nn.TrainOneStepCell(net_with_criterion, optimizer)
            # 设置网络为训练模式
            train_net.set_train()            

            if iteration % 10 == 0:
                loss_ = losses / (batch_idx + 1)
                print('Timer: {:.4f} sec.'.format(t1 - t0))
                print('epoch ' + repr(epoch) + ' iter ' +
                      repr(iteration) + ' || Loss:%.4f' % (loss_))
                #print('->> face Loss: {:.4f} || head loss : {:.4f}'.format(face_loss, head_loss))
                print('->> lr: {}'.format(optimizer.parameters[0]['lr']))

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'pyramidbox_' + repr(iteration) + '.ckpt'
                #torch.save(pyramidbox.state_dict(),os.path.join(args.save_folder, file))
                save_checkpoint(pyramidbox, os.path.join(args.save_folder, file))
            iteration += 1

        val(epoch, net, pyramidbox, criterion1, criterion2)


class MyWithLossCell(Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, images, face_targets, head_targets):
        out = self._backbone(images)
        return self._loss_fn(out, face_targets, head_targets)

    @property
    def backbone_network(self):
        return self._backbone

def loss_fn(out, face_targets, head_targets):
    criterion1 = MultiBoxLoss(cfg, args.cuda)
    criterion2 = MultiBoxLoss(cfg, args.cuda, use_head_loss=True)

    face_loss_l, face_loss_c = criterion1(out, face_targets)
    head_loss_l, head_loss_c = criterion2(out, head_targets)

    loss = face_loss_l + face_loss_c + head_loss_l + head_loss_c

    return loss

def val(epoch,
        net,
        pyramidbox,
        criterion1,
        criterion2):
    net.eval()
    face_losses = 0
    head_losses = 0
    step = 0
    t1 = time.time()
    
    #for batch_idx, (images, face_targets, head_targets) in enumerate(val_loader):
    for batch_idx, (images, face_targets, head_targets) in val_dataset.create_dict_iterator():
        '''if args.cuda:
            images = Variable(images.cuda())
            face_targets = [Variable(ann.cuda(), volatile=True)
                            for ann in face_targets]
            head_targets = [Variable(ann.cuda(), volatile=True)
                            for ann in head_targets]
        else:
            images = Variable(images)
            face_targets = [Variable(ann, volatile=True)
                            for ann in face_targets]
            head_targets = [Variable(ann, volatile=True)
                            for ann in head_targets]'''

        out = net(images)
        face_loss_l, face_loss_c = criterion1(out, face_targets)
        head_loss_l, head_loss_c = criterion2(out, head_targets)

        face_losses += (face_loss_l + face_loss_c).data[0]
        head_losses += (head_loss_l + head_loss_c).data[0]
        step += 1

    tloss = face_losses / step

    t2 = time.time()
    print('test Timer:{:.4f} .sec'.format(t2 - t1))
    print('epoch ' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        #torch.save(pyramidbox.state_dict(), os.path.join(args.save_folder, 'pyramidbox.pth'))
        save_checkpoint(pyramidbox, os.path.join(args.save_folder, 'pyramidbox.ckpt'))
        min_loss = tloss

    states = [{
        'epoch': epoch,
        'weight': pyramidbox.state_dict(),
    }]
    #torch.save(states, os.path.join(args.save_folder, 'pyramidbox_checkpoint.pth'))
    save_checkpoint(states, os.path.join(args.save_folder, 'pyramidbox_checkpoint.ckpt'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    #for param_group in optimizer.param_groups:
    for param in optimizer.parameters:
        param['lr'] = lr


if __name__ == '__main__':
    train()
