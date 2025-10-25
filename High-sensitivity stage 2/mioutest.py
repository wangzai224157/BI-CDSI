import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.nn import functional as F
# -----------------------------
# ✅ 二分类 DiceLoss
# -----------------------------
class DiceLoss:
    def __init__(self,
                 axis: int = 1,
                 smooth: float = 1e-6,
                 reduction: str = "mean",
                 square_in_union: bool = False):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union
    def __call__(self, pred, targ):
        targ = self._one_hot(targ, classes=2)
        assert pred.shape == targ.shape, "预测与标签形状不匹配"
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred**2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        return loss.mean() if self.reduction == "mean" else loss.sum()
    @staticmethod
    def _one_hot(x, classes: int = 2, axis: int = 1):
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)
    def activation(self, x):
        return F.softmax(x, dim=self.axis)
# -----------------------------
# ✅ IoU 计算
# -----------------------------
def compute_iou(pred, gt):
    pred_label = pred.argmax(1)
    gt = gt.squeeze(0)
    intersection = torch.sum(torch.mul(pred_label, gt))
    union = torch.sum(torch.add(pred_label, gt)) - intersection
    return float(intersection) / float(union) if union != 0 else 0.0
# -----------------------------
# ✅ 图像加载函数（自动修正标签）
# -----------------------------
def load_binary_image(img_path, is_pred: bool = True):
    """
    加载图像并转为二分类任务所需张量
    自动修正标签值，防止出现 255 或非法像素值
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {img_path}")
    # 自动二值化 + 修正
    if not is_pred:
        if img.max() > 1: # 防止存在 255 或其他灰度值
            img = np.where(img > 127, 1, 0).astype(np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).long()
    else:
        # 预测图同样阈值化，确保仅 {0,1}
        if img.max() > 1:
            img = np.where(img > 127, 1, 0).astype(np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).long()
        tensor = F.one_hot(tensor, num_classes=2).permute(0, 3, 1, 2).float()
        tensor = F.softmax(tensor, dim=1)
    return tensor
# -----------------------------
# ✅ 文件夹批量对比
# -----------------------------
def compare_binary_folders(pred_folder, gt_folder, output_csv="binary_comparison.csv"):
    dice_loss_fn = DiceLoss(reduction="mean")
    ce_loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([0.9, 1.1]).cuda() if torch.cuda.is_available() else torch.FloatTensor([0.9, 1.1])
    )
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    pred_files = sorted([f for ext in img_extensions for f in glob(os.path.join(pred_folder, ext))])
    gt_files = sorted([f for ext in img_extensions for f in glob(os.path.join(gt_folder, ext))])
    if len(pred_files) != len(gt_files):
        raise ValueError(f"预测图与标签图数量不匹配: {len(pred_files)} vs {len(gt_files)}")
    results = []
    for pred_path, gt_path in zip(pred_files, gt_files):
        fname = os.path.basename(pred_path)
        try:
            pred = load_binary_image(pred_path, is_pred=True)
            gt = load_binary_image(gt_path, is_pred=False)
            if torch.cuda.is_available():
                pred = pred.cuda()
                gt = gt.cuda()
            dice_loss = dice_loss_fn(pred, gt)
            dice_score = 1 - dice_loss.item()
            logits = torch.log(pred + 1e-8)
            ce_loss = ce_loss_fn(logits, gt.squeeze(1)).item()
            iou = compute_iou(pred, gt)
            results.append({
                "filename": fname,
                "dice_score": round(dice_score, 4),
                "ce_loss": round(ce_loss, 4),
                "iou": round(iou, 4)
            })
            print(f"处理完成: {fname} - Dice: {dice_score:.4f}, CE: {ce_loss:.4f}, IoU: {iou:.4f}")
        except Exception as e:
            print(f"处理 {fname} 时出错: {str(e)}")
            continue
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ 二分类对比结果已保存至 {output_csv}")
# -----------------------------
# ✅ 主程序入口
# -----------------------------
if __name__ == "__main__":
    PREDICTION_FOLDER = "/mnt/sda/zhouying/2.48/SWCNN-main_multi/same"
    GROUND_TRUTH_FOLDER = "/mnt/sda/zhouying/2.48/SWCNN-main_multi/ground_truth"
    compare_binary_folders(
        pred_folder=PREDICTION_FOLDER,
        gt_folder=GROUND_TRUTH_FOLDER,
        output_csv="binary_results.csv"
    )

