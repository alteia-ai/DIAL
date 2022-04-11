import numpy as np
from sklearn.metrics import f1_score as sk_f1


def accuracy(input, target, ignore_indx=None):
    """Computes the total accuracy"""
    if ignore_indx is None:
        return 100 * float(np.count_nonzero(input == target)) / target.size
    else:
        mask = input == target
        mask[np.where(target == ignore_indx)] = False
        total = np.sum(np.where(target != ignore_indx, 1, 0))
        return 100 * np.sum(mask) / total


def IoU(pred, gt, n_classes, all_iou=False, ignore_indx=None):
    """Computes the IoU by class and returns mean-IoU"""
    # print("IoU")
    iou = []
    for i in range(n_classes):
        if i == ignore_indx:
            continue
        if np.sum(gt == i) == 0:
            iou.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        iou.append(TP / (TP + FP + FN))
    # nanmean: if a class is not present in the image, it's a NaN
    result = [np.nanmean(iou), iou] if all_iou else np.nanmean(iou)
    return result


def f1_score(pred, gt, n_classes, all=False, ignore_indx=None):
    f1 = []
    for i in range(n_classes):
        if i == ignore_indx:
            continue
        if np.sum(gt == i) == 0:
            f1.append(np.NaN)
            continue
        TP = np.sum(np.logical_and(pred == i, gt == i))
        FP = np.sum(np.logical_and(pred == i, gt != i))
        FN = np.sum(np.logical_and(pred != i, gt == i))
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        result = 2 * (prec * recall) / (prec + recall)
        f1.append(result)
    result = [np.nanmean(f1), f1] if all else np.nanmean(f1)
    if all:
        flat_pred = pred.reshape(-1)
        flat_gt = gt.reshape(-1)
        if ignore_indx is not None:
            flat_pred = flat_pred[np.where(flat_gt != ignore_indx)]
            flat_gt = flat_gt[np.where(flat_gt != ignore_indx)]
        f1_weighted = sk_f1(flat_gt, flat_pred, average="weighted")
        result.append(f1_weighted)
    return result

