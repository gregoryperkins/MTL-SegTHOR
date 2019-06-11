import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb


class BCELoss(nn.Module):
    def __init__(self, eps=1e-7, if_mean=True):
        super(BCELoss, self).__init__()
        self.eps = eps
        self.if_mean = if_mean

    def forward(self, inputs, target):
        logit = inputs.clamp(self.eps, 1. - self.eps)
        loss = -(target.float() * torch.log(logit) +
                 (1 - target.float()) * torch.log(1 - logit))
        if self.if_mean:
            return loss.mean()
        else:
            return loss


class DependentLoss(nn.Module):
    '''
    Attributes:
        alpha: a metric to indicate the global probability
        binary_loss: the binary classification loss for each class
        
    Functions:
        forward:
            attr:
                inputs: the sigmoid probability with shape (batch_size, n_class)
                target: the label with shape (batch_size, n_class)
            return:
                count_loss: the dependent loss for each class
                count_p: the dependent probability for each class
    '''

    def __init__(self, alpha=None):
        super(DependentLoss, self).__init__()
        self.alpha = alpha
        self.binary_loss = BCELoss(if_mean=False)

    def forward(self, inputs, target):
        n_class = inputs.size(1)
        batch_size = inputs.size(0)
        count_loss = 0
        count_p = []

        if self.alpha is not None:
            for class_index in range(n_class):
                cur_p = []
                for condition_index in range(n_class):
                    alpha_condition_batch = self.alpha[
                        condition_index, class_index] * inputs[:,
                                                               condition_index]
                    cur_p.append(alpha_condition_batch)
                cur_p = torch.stack(cur_p, 1).sum(1) / n_class
                count_p.append(cur_p)
            count_p = torch.stack(count_p, 1)
        else:
            count_p = inputs

        for class_index in range(n_class):
            cur_loss = self.binary_loss(count_p[:, class_index],
                                        target[:, class_index])
            count_loss += cur_loss.mean()
        return count_loss, count_p


class MultiLabelLoss(nn.Module):
    """
    Weighted BCELoss. This loss was used for comparation.
    reference 
    @inproceedings{segthor_tao2019,
    author = {Tao He, Jixiang Guo, Jianyong Wang, Xiuyuan Xu, Zhang Yi},
    title = {Multi-task Learning for the Segmentation of Thoracic Organs at Risk in CT images},
    booktile = {Proceedings of the 2019 Challenge on Segmentation of THoracic 
    Organs at Risk in CT Images (SegTHOR2019) },
    volume = {2349},
    year = {2019},
    }
    Args:
        alpha: the weight for current class (alpha in the paper)
    Funs:
        forward: the forward computing of bceloss
            Returns:
            count_loss: the loss
            inputs: the probability for each class
    
    """

    def __init__(self, alpha=None):
        super(MultiLabelLoss, self).__init__()
        self.alpha = alpha
        self.binary_loss = BCELoss(if_mean=False)

    def forward(self, inputs, target):
        n_class = inputs.size(1)
        count_loss = 0
        for class_index in range(n_class):
            cur_loss = self.binary_loss(inputs[:, class_index],
                                        target[:, class_index])
            count_loss += cur_loss.mean()
        if self.alpha is not None:
            count_loss = count_loss * self.alpha
        return count_loss, inputs


class SoftDiceLoss(nn.Module):
    """
    The Dice Loss function
    """
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels):
        numerator = 2 * torch.sum(labels * probs, 2)
        denominator = torch.sum(labels + probs**2, 2) + self.smooth
        return 1 - torch.mean(numerator / denominator)


class CombinedLoss(nn.Module):
    """
    The combined loss for multi-task learning.
    if if_closs is True, the multi-task learning is used; otherwise, the dice loss is used. 
    if alpha=None, the c_loss_fun is the weighted BCELoss; otherwise, the c_loss_fun is the DependentLoss.
    Args:
        alpha: the weight
        if_closs: the flag whether use multi-task learning
        s_loss_fun: the segmentation loss (SoftDiceLoss)
        c_loss_fun: the multi-label classification loss (DependentLoss or MultiLabelLoss)
    Functions:
        Args:
            s_logit: network output for segmentation
            c_logit: network output for classification
            s_label: 
            
    """
    def __init__(self, alpha=None, if_closs=1):
        super(CombinedLoss, self).__init__()
        self.closs_flag = if_closs
        self.s_loss_fun = SoftDiceLoss()
        if alpha is not None:
            self.c_loss_fun = DependentLoss(alpha)
        else:
            self.c_loss_fun = MultiLabelLoss()

    def forward(self, s_logit, c_logit, s_label, c_label):
        probs = F.softmax(s_logit, 1)
        batch_size, n_class = probs.size(0), probs.size(1)
        labels = s_label.view(batch_size, n_class, -1).float()
        probs = probs.view(batch_size, n_class, -1)
        s_loss = self.s_loss_fun(probs, labels)
        c_loss, c_p = self.c_loss_fun(c_logit, c_label)
        total_loss = s_loss + self.closs_flag * c_loss
        return total_loss, c_loss, s_loss, c_p
    