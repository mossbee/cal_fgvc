import os
import config as config

import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from models import WSDAN_CAL
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment, VerificationMetrics, compute_cosine_similarity
from datasets import get_trainval_datasets
import math


# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
top1_container = AverageMeter(name='top1')
top5_container = AverageMeter(name='top5')

raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

verification_metric = VerificationMetrics()

best_acc = 0.0

def main():
    torch.backends.cudnn.benchmark = True

    ##################################
    # Logging setting
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    train_dataset, verification_dataset = get_trainval_datasets(config.tag, config.image_size)

    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN_CAL(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).cuda()

    if config.ckpt and os.path.isfile(config.ckpt):
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt, weights_only=False)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch']) # start from the beginning

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict, strict=False)
        logging.info('Network loaded from {}'.format(config.ckpt))
        print('Network loaded from {} @ {} epoch'.format(config.ckpt, start_epoch))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].cuda()
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.cuda()

    learning_rate = config.learning_rate
    print('begin with', learning_rate, 'learning rate')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    train_loader, verification_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True, drop_last=True), \
                                    DataLoader(verification_dataset, batch_size=config.batch_size * 2, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)

    callback_monitor = 'val_verification_accuracy'
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Verification pairs: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(verification_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']
        print('current lr =', optimizer.param_groups[0]['lr'])

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))
        train(epoch=epoch,
              logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)

        verify_evaluate(logs=logs,
                 data_loader=verification_loader,
                 net=net,
                 pbar=pbar,
                 epoch=epoch)

        torch.cuda.synchronize()
        callback.on_epoch_end(logs, net, feature_center=feature_center)
        if (epoch + 1) % 5 == 0:
            save_model(net, logs, 'model_last.pth')
        pbar.close()

def verify_evaluate(**kwargs):
    """Verification evaluation using feature embeddings and cosine similarity"""
    global best_acc
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    epoch = kwargs['epoch']

    # Reset metrics
    verification_metric.reset()

    # Begin verification evaluation
    start_time = time.time()
    net.eval()
    
    with torch.no_grad():
        for i, (img1, img2, labels) in enumerate(data_loader):
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.cuda()

            # Extract features for both images
            _, _, feature1, _ = net(img1)
            _, _, feature2, _ = net(img2)

            # Compute cosine similarity
            similarities = compute_cosine_similarity(feature1, feature2)
            
            # Update metrics
            verification_metric.update(similarities, labels)

    # Compute all verification metrics
    metrics_dict = verification_metric.compute_metrics()
    end_time = time.time()

    # Log results
    logs['val_verification_accuracy'] = metrics_dict['accuracy']
    logs['val_verification_eer'] = metrics_dict['eer']
    logs['val_verification_auc'] = metrics_dict['auc_roc']
    logs['val_verification_far'] = metrics_dict['far']
    logs['val_verification_frr'] = metrics_dict['frr']

    batch_info = 'Verify Acc {:.2f}%, EER {:.2f}%, AUC {:.2f}%, FAR {:.2f}%, FRR {:.2f}%'.format(
        metrics_dict['accuracy'], metrics_dict['eer'], metrics_dict['auc_roc'],
        metrics_dict['far'], metrics_dict['frr'])

    pbar.set_postfix_str(batch_info)

    if metrics_dict['accuracy'] > best_acc:
        best_acc = metrics_dict['accuracy']
        save_model(net, logs, 'model_best_verify.pth')

    if epoch % 10 == 0:
        save_model(net, logs, 'model_epoch%d.pth' % epoch)

    print(batch_info + f', Best {best_acc:.2f}%')

    # Write log for this epoch
    logging.info('Verify: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')

def adjust_learning(optimizer, epoch, iter):
    """Decay the learning rate based on schedule"""
    base_lr = config.learning_rate
    base_rate = 0.9
    base_duration = 2.0
    lr = base_lr * pow(base_rate, (epoch + iter) / base_duration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(**kwargs):
    # Retrieve training configuration
    epoch = kwargs['epoch']
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    batch_len = len(data_loader)
    for i, (X, y) in enumerate(data_loader):
        float_iter = float(i) / batch_len
        adjust_learning(optimizer, epoch, float_iter)
        now_lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        # obtain data for training
        X = X.cuda()
        y = y.cuda()

        y_pred_raw, y_pred_aux, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))
        aug_images = torch.cat([crop_images, drop_images], dim=0)
        y_aug = torch.cat([y, y], dim=0)

        # crop images forward
        y_pred_aug, y_pred_aux_aug, _, _ = net(aug_images)

        y_pred_aux = torch.cat([y_pred_aux, y_pred_aux_aug], dim=0)
        y_aux = torch.cat([y, y_aug], dim=0)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. + \
                     cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_aug, y_aug)
            epoch_drop_acc = drop_metric(y_pred_aux, y_aux)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Aug Acc ({:.2f}, {:.2f}), Aux Acc ({:.2f}, {:.2f}), lr {:.5f}'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1], now_lr)

        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))

def save_model(net, logs, ckpt_name):
    torch.save({'logs': logs, 'state_dict': net.state_dict()}, config.save_dir + ckpt_name)

if __name__ == '__main__':
    main()

