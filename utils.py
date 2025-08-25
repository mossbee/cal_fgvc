import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()

        return self.corrects * 100. / self.num_samples


class VerificationMetrics(Metric):
    def __init__(self):
        self.name = 'verification'
        self.reset()
    
    def reset(self):
        self.similarities = []
        self.labels = []
    
    def update(self, similarities, labels):
        """Update with batch of similarities and labels
        Args:
            similarities: cosine similarities between pairs
            labels: 0 for different identity, 1 for same identity
        """
        if isinstance(similarities, torch.Tensor):
            similarities = similarities.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        self.similarities.extend(similarities.flatten())
        self.labels.extend(labels.flatten())
    
    def compute_metrics(self):
        """Compute all verification metrics"""
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(labels, similarities)
        
        # AUC ROC
        auc_roc = metrics.auc(fpr, tpr)
        
        # EER (Equal Error Rate)
        fnr = 1 - tpr
        eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = interp1d(fpr, fnr)(eer_threshold)
        
        # Best threshold for accuracy
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        predictions = (similarities >= best_threshold).astype(int)
        accuracy = metrics.accuracy_score(labels, predictions)
        
        # FAR and FRR at best threshold
        tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Accept Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Reject Rate
        
        return {
            'accuracy': accuracy * 100,
            'eer': eer * 100,
            'auc_roc': auc_roc * 100,
            'far': far * 100,
            'frr': frr * 100,
            'best_threshold': best_threshold
        }


def compute_cosine_similarity(features1, features2):
    """Compute cosine similarity between two feature vectors"""
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    return F.cosine_similarity(features1, features2)

##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_transform_ndtwin():
    return transforms.Compose([transforms.ColorJitter(brightness=0.126, saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

