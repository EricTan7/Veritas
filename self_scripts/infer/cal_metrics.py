import base64
from openai import OpenAI
import io
import os
import json
import argparse
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score


def get_metrics(data):
    """
        Calculate metrics for one dataset
        Input: data: {"images": "xxx.png", "result": "xxx", "label": 0}
        Return: result_dict: {"acc", "precision_fake", "recall_fake", "f1_fake", "precision_real", "recall_real", "f1_real"}
    """
    pred, label = [], []
    failed = 0
    for item in data:
        output = item["result"]
        try:
            pattern_answer = r'<answer>\s*(.*?)\s*</answer>'
            answer = re.search(pattern_answer, output)
            answer = answer.group(1).lower().strip()
            if "real" in answer:
                pred_cur = 0
            elif "fake" in answer:
                pred_cur = 1
            else:
                failed += 1
                continue
        except:
            failed += 1
            continue
        pred.append(pred_cur)
        label.append(item["label"])

    pred, label = np.array(pred), np.array(label)
    acc = np.sum(pred == label) / len(label)
    precision_fake = precision_score(label, pred, pos_label=1)
    recall_fake = recall_score(label, pred, pos_label=1)
    precision_real = precision_score(label, pred, pos_label=0)
    recall_real = recall_score(label, pred, pos_label=0)
    
    try:
        f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake)
    except:
        f1_fake = 0.0
    try:
        f1_real = 2 * precision_real * recall_real / (precision_real + recall_real)
    except:
        f1_real = 0.0


    acc, precision_fake, recall_fake, f1_fake, precision_real, recall_real, f1_real = round(acc, 4), round(precision_fake, 4),round(recall_fake, 4),round(f1_fake, 4),round(precision_real, 4),round(recall_real, 4),round(f1_real, 4)
    print(acc, precision_fake, recall_fake, f1_fake, precision_real, recall_real, f1_real)
    result_dict = {
        "acc": acc,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real
    }
    
    return result_dict

