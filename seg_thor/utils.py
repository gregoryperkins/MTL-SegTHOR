import os
import torch
import pdb
import numpy as np
from sklearn import metrics as sm


def get_threshold(predict, label, min_tpr=0.995):
    n_class = predict.shape[1]
    adaptive_thresholds = np.zeros((n_class,))
    for class_index in range(n_class):
        fpr, tpr, thresholds = sm.roc_curve(
            label[:, class_index],
            predict[:, class_index],
            pos_label=1,
            drop_intermediate=False)
        for c_tpr, threshold in zip(tpr, thresholds):
            if c_tpr >= min_tpr:
                adaptive_thresholds[class_index] = threshold
                break
    return adaptive_thresholds


def setgpu(gpus):
    if gpus=='all':
        gpus = '0,1,2,3'
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def aic_fundus_lesion_segmentation(ground_truth, prediction):
    ground_truth = ground_truth.ravel()
    prediction = prediction.ravel()
    num_class = 5
    try:
        ret = np.zeros((num_class,))
        for i in range(num_class):
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = float(2 * (
                    (mask1 * (ground_truth == prediction)).sum())) / (
                        mask1.sum() + mask2.sum())
            else:
                ret[i] = 0  #float('nan')
    except Exception as e:
        print("ERROR msg:")
        return None
    
    return [np.mean(ret)] + ret.tolist()


def metric(predict, label, thresholds=None):
    precision = []
    recall = []
    for j in range(4):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        if not thresholds is None:
            cur_threshold = thresholds[j]
        else:
            cur_threshold = 0.5
        for i in range(predict.shape[0]):
            if (predict[i, j] > cur_threshold and label[i, j] == 0):
                fp += 1
            elif (predict[i, j] > cur_threshold and label[i, j] == 1):
                tp += 1
            elif (predict[i, j] < cur_threshold and label[i, j] == 0):
                tn += 1
            else:
                fn += 1
        cur_precision = float(tp) / (tp + fp + 10e-7)
        cur_recall = float(tp) / (tp + fn + 10e-7)
        precision.append(cur_precision)
        recall.append(cur_recall)
    return precision, recall


def segmentation_metrics(outputs, labels, class_idxs=[1, 2, 3, 4]):
    TPVFs, dices, PPVs, FPVFs = [], [], [], []
    for class_idx in class_idxs:
        mask_o = (outputs == class_idx)
        mask_y = (labels == class_idx)
        union = (mask_o * mask_y).sum()
        inter = mask_o.sum() + mask_y.sum()
        v_y = mask_y.sum()
        v_o = mask_o.sum()
        v_a = np.prod(labels.shape)

        if v_y == 0:
            TPVFs.append(0)
        else:
            TPVFs.append(round(float(union) / v_y, 5))
        if inter == 0:
            dices.append(0)
        else:
            dices.append(round(float(2 * union) / inter, 5))
        if v_o == 0:
            PPVs.append(0)
        else:
            PPVs.append(round(float(union) / v_o, 5))
        if (v_a - v_y) == 0:
            FPVFs.append(0)
        else:
            FPVFs.append(round(float(v_o - union) / (v_a - v_y), 5))
    return TPVFs, dices, PPVs, FPVFs

