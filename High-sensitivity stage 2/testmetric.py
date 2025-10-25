import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.nn import functional as F


# -----------------------------
#  自动二值化加载函数
# -----------------------------
def load_binary_image(img_path, is_pred: bool = True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {img_path}")

    # 二值化
    img = np.where(img > 127, 1, 0).astype(np.uint8)

    if not is_pred:
        tensor = torch.from_numpy(img).unsqueeze(0).long()  # (1,H,W)
    else:
        tensor = torch.from_numpy(img).unsqueeze(0).long()
        tensor = F.one_hot(tensor, num_classes=2).permute(0, 3, 1, 2).float()
        tensor = F.softmax(tensor, dim=1)
    return tensor


# -----------------------------
# 混淆矩阵指标计算
# -----------------------------
def confusion_matrix_metrics(pred, gt):
    """
    输入: pred (1,2,H,W), gt (1,H,W)
    返回: dict(TP, FP, TN, FN, precision, recall, F1, OA)
    """
    pred_label = pred.argmax(1).squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()

    TP = np.logical_and(pred_label == 1, gt == 1).sum()
    TN = np.logical_and(pred_label == 0, gt == 0).sum()
    FP = np.logical_and(pred_label == 1, gt == 0).sum()
    FN = np.logical_and(pred_label == 0, gt == 1).sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    F1 = 2 * precision * recall / (precision + recall + 1e-6)
    OA = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    return dict(TP=TP, FP=FP, TN=TN, FN=FN, precision=precision, recall=recall, F1=F1, OA=OA)


# -----------------------------
#  二分类指标汇总函数
# -----------------------------
def compare_binary_folders(pred_folder, gt_folder, output_csv="binary_metrics.csv"):
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    pred_files = sorted([f for ext in img_extensions for f in glob(os.path.join(pred_folder, ext))])
    gt_files = sorted([f for ext in img_extensions for f in glob(os.path.join(gt_folder, ext))])

    if len(pred_files) != len(gt_files):
        raise ValueError(f"预测图与标签图数量不匹配: {len(pred_files)} vs {len(gt_files)}")

    results = []
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

    for pred_path, gt_path in zip(pred_files, gt_files):
        fname = os.path.basename(pred_path)

        pred = load_binary_image(pred_path, is_pred=True)
        gt = load_binary_image(gt_path, is_pred=False)

        if torch.cuda.is_available():
            pred = pred.cuda()
            gt = gt.cuda()

        # 计算每张图的指标
        metrics = confusion_matrix_metrics(pred, gt)

        results.append({
            "filename": fname,
            "F1": round(metrics["F1"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "OA": round(metrics["OA"], 4)
        })

        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_TN += metrics["TN"]
        total_FN += metrics["FN"]

        print(f"{fname} - F1: {metrics['F1']:.4f}, OA: {metrics['OA']:.4f}")

    # 计算总体 MFI / OA
    total_precision = total_TP / (total_TP + total_FP + 1e-6)
    total_recall = total_TP / (total_TP + total_FN + 1e-6)
    total_F1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-6)
    total_OA = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN + 1e-6)

    summary = {
        "MFI (Mean F1 Index)": round(total_F1, 4),
        "Overall Accuracy (OA)": round(total_OA, 4)
    }

    # 保存结果
    df = pd.DataFrame(results)
    df.loc["Average"] = ["-", df["F1"].mean(), df["Precision"].mean(), df["Recall"].mean(), df["OA"].mean()]
    df.to_csv(output_csv, index=False)

    print(f"\n 平均结果：MFI={summary['MFI (Mean F1 Index)']:.4f}, OA={summary['Overall Accuracy (OA)']:.4f}")
    print(f"结果已保存到 {output_csv}")

    return summary


# -----------------------------
# 
# -----------------------------
if __name__ == "__main__":
    PRED_FOLDER = "path"
    GT_FOLDER = "path"

    summary = compare_binary_folders(PRED_FOLDER, GT_FOLDER, output_csv="metrics_with_MFI_OA.csv")

