import numpy as np
from sklearn import metrics
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve, roc_curve
import torch
import torch.nn as nn
from copy import deepcopy


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy


def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    #print(prob.size(), label.size())
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def calculate_metrics_for_train(label, output, pos_label=1):
    try:
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
    except:
        prob = output

    # Accuracy
    try:
        _, prediction = torch.max(output, 1)
        correct = (prediction == label).sum().item()
        accuracy = correct / prediction.size(0)
    except:
        accuracy = -1.0

    # Average Precision
    y_true = label.cpu().detach().numpy()
    y_pred = prob.cpu().detach().numpy()
    ap = metrics.average_precision_score(y_true, y_pred)

    # AUC and EER
    try:
        fpr, tpr, thresholds = metrics.roc_curve(label.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=pos_label)
    except:
        # for the case when we only have one sample
        return None, None, accuracy, ap

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # for the case when all the samples within a batch is fake/real
        auc, eer = None, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return auc, eer, accuracy, ap


# ------------ compute average metrics of batches---------------------
class Metrics_batch():
    def __init__(self):
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.aucs = []
        self.eers = []
        self.aps = []

        self.correct = 0
        self.total = 0
        self.losses = []

    def update(self, label, output):
        acc = self._update_acc(label, output)
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
        #label = 1-label
        #prob = torch.softmax(output, dim=1)[:, 1]
        auc, eer = self._update_auc(label, prob)
        ap = self._update_ap(label, prob)

        return acc, auc, eer, ap

    def _update_auc(self, lab, prob):
        fpr, tpr, thresholds = metrics.roc_curve(lab.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=1)
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            return -1, -1

        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(auc)

        # return auc

        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        self.eers.append(eer)

        return auc, eer

    def _update_acc(self, lab, output):
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == lab).sum().item()
        accuracy = correct / prediction.size(0)
        # self.accs.append(accuracy)
        self.correct = self.correct+correct
        self.total = self.total+lab.size(0)
        return accuracy

    def _update_ap(self, label, prob):
        y_true = label.cpu().detach().numpy()
        y_pred = prob.cpu().detach().numpy()
        ap = metrics.average_precision_score(y_true,y_pred)
        self.aps.append(ap)

        return np.mean(ap)

    def get_mean_metrics(self):
        mean_acc, std_acc = self.correct/self.total, 0
        mean_auc, std_auc = self._mean_auc()
        mean_err, std_err = np.mean(self.eers), np.std(self.eers)
        mean_ap, std_ap = np.mean(self.aps), np.std(self.aps)
        
        return {'acc':mean_acc, 'auc':mean_auc, 'eer':mean_err, 'ap':mean_ap}

    def _mean_auc(self):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        return mean_auc, std_auc

    def clear(self):
        self.tprs.clear()
        self.aucs.clear()
        # self.accs.clear()
        self.correct=0
        self.total=0
        self.eers.clear()
        self.aps.clear()
        self.losses.clear()


# ------------ compute average metrics of all data ---------------------
class Metrics_all():
    def __init__(self):
        self.probs = []
        self.labels = []
        self.correct = 0
        self.total = 0

    def store(self, label, output):
        prob = torch.softmax(output, dim=1)[:, 1]
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == label).sum().item()
        self.correct += correct
        self.total += label.size(0)
        self.labels.append(label.squeeze().cpu().numpy())
        self.probs.append(prob.squeeze().cpu().numpy())

    def get_metrics(self):
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred,pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap}

    def clear(self):
        self.probs.clear()
        self.labels.clear()
        self.correct = 0
        self.total = 0


# only used to record a series of scalar value
class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0
    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num
    def average(self):
        if self.num == 0:
            return None
        return self.sum/self.num
    def clear(self):
        self.sum = 0
        self.num = 0






def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_video_metrics(image, pred, label):
    """
        Calculate video-level auc for the frame-level methods.
    """
    result_dict = {}
    new_label = []
    new_pred = []
    result_dict_hash = {}
    for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

        s = item[0]
        if '\\' in s:
            parts = s.split('\\')
        else:
            parts = s.split('/')
        a = parts[-2]
        b = parts[-1]

        if a not in result_dict:
            result_dict[a] = []
        if a not in result_dict_hash:
            result_dict_hash[a] = item[2]

        if result_dict_hash[a] != item[2]:
            a_new = f'{a}_new'
            if a_new not in result_dict:
                result_dict[a_new] = []
            result_dict[a_new].append(item)
        else:
            result_dict[a].append(item)
    image_arr = list(result_dict.values())

    for video in image_arr:
        pred_sum = 0
        label_sum = 0
        leng = 0
        for frame in video:
            pred_sum += float(frame[1])
            label_sum += int(frame[2])
            leng += 1
        new_pred.append(pred_sum / leng)
        new_label.append(int(label_sum / leng))
    fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
    v_auc = metrics.auc(fpr, tpr)
    fnr = 1 - tpr
    v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return v_auc, v_eer


def get_test_metrics(y_pred, y_true, img_names, pos_label=1, thresh=0.5):
    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > thresh).astype(int)
    acc = np.sum(prediction_class == y_true) / len(y_true)

    # f1
    precision_fake = precision_score(y_true, prediction_class, pos_label=1)
    recall_fake = recall_score(y_true, prediction_class, pos_label=1)
    precision_real = precision_score(y_true, prediction_class, pos_label=0)
    recall_real = recall_score(y_true, prediction_class, pos_label=0)
    try:
        f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake)
    except:
        f1_fake = "null"
    try:
        f1_real = 2 * precision_real * recall_real / (precision_real + recall_real)
    except:
        f1_real = "null"
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc
    
    # pred cnt
    pred_real = np.count_nonzero(prediction_class==0)
    pred_fake = np.count_nonzero(prediction_class==1)

    return {'acc': acc, 
    'precision_fake':precision_fake, 'recall_fake': recall_fake, 'f1_fake': f1_fake, 
    'precision_real':precision_real, 'recall_real': recall_real, 'f1_real': f1_real, 
    'pred_real': pred_real, 'pred_fake': pred_fake, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}

