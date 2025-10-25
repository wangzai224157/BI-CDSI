import os
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.nn import functional as F


# 二分类专用Dice损失（复用原始逻辑，num_classes固定为2）
class DiceLoss:
    def __init__(self,
                 axis: int = 1,
                 smooth: float = 1e-6,
                 reduction: str = "mean",  # 对比场景用mean更合理
                 square_in_union: bool = False):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def __call__(self, pred, targ):
        # 对标签进行独热编码（二分类，num_classes=2）
        targ = self._one_hot(targ, classes=2)
        assert pred.shape == targ.shape, "预测与标签形状不匹配"
        pred = self.activation(pred)  # 应用softmax
        
        # 计算交并集（忽略批次维度，只对H、W求和）
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred**2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        
        # 损失聚合
        return loss.mean() if self.reduction == "mean" else loss.sum()

    @staticmethod
    def _one_hot(x, classes: int = 2, axis: int = 1):
        # 二分类独热编码（仅支持0和1的标签）
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)

    def activation(self, x):
        return F.softmax(x, dim=self.axis)


# 二分类专用IoU计算（复用原始逻辑）
def compute_iou(pred, gt):
    # pred: 模型输出概率图 (1, 2, H, W)；gt: 标签 (1, H, W)
    pred_label = pred.argmax(1)  # 从概率图取预测类别（0或1）
    gt = gt.squeeze(0)  # 去除批次维度 (H, W)
    
    # 计算交集和并集
    intersection = torch.sum(torch.mul(pred_label, gt))
    union = torch.sum(torch.add(pred_label, gt)) - intersection
    
    # 避免除零错误
    return float(intersection) / float(union) if union != 0 else 0.0


# 二分类图像加载（强制标签为0/1）
def load_binary_image(img_path, is_pred: bool = True):
    """
    加载图像并转为二分类任务所需张量
    is_pred: True=预测图（转为概率分布），False=标签图（强制0/1）
    """
    # 读取灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {img_path}")
    
    # 处理标签图：强制所有非0值为1（确保只有0和1）
    if not is_pred:
        img = np.where(img > 0, 1, 0).astype(np.uint8)  # 核心：二值化
        tensor = torch.from_numpy(img).unsqueeze(0).long()  # 形状: (1, H, W)
    
    # 处理预测图：转为概率分布（模拟模型输出）
    else:
        # 先转为标签格式（0/1），再转为独热编码+softmax（模拟模型输出）
        img = np.where(img > 0, 1, 0).astype(np.uint8)
        tensor = torch.from_numpy(img).unsqueeze(0).long()  # (1, H, W)
        tensor = F.one_hot(tensor, num_classes=2).permute(0, 3, 1, 2).float()  # (1, 2, H, W)
        tensor = F.softmax(tensor, dim=1)  # 转为概率
    
    return tensor


# 对比两个文件夹（二分类专用）
def compare_binary_folders(pred_folder, gt_folder, output_csv="binary_comparison.csv"):
    # 初始化损失函数
    dice_loss_fn = DiceLoss(reduction="mean")
    ce_loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([0.9, 1.1]).cuda() if torch.cuda.is_available() else torch.FloatTensor([0.9, 1.1])
    )

    # 获取所有图像路径（按文件名排序，确保对应）
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    pred_files = sorted([f for ext in img_extensions for f in glob(os.path.join(pred_folder, ext))])
    gt_files = sorted([f for ext in img_extensions for f in glob(os.path.join(gt_folder, ext))])

    # 检查文件数量
    if len(pred_files) != len(gt_files):
        raise ValueError(f"预测图与标签图数量不匹配: {len(pred_files)} vs {len(gt_files)}")

    # 存储结果
    results = []

    for pred_path, gt_path in zip(pred_files, gt_files):
        fname = os.path.basename(pred_path)
        gt_fname = os.path.basename(gt_path)
        if fname != gt_fname:
            print(f"警告: 文件名不匹配 - {fname} vs {gt_fname}")

        try:
            # 加载图像（预测图和标签图）
            pred = load_binary_image(pred_path, is_pred=True)
            gt = load_binary_image(gt_path, is_pred=False)

            # 转移到GPU（如果可用）
            if torch.cuda.is_available():
                pred = pred.cuda()
                gt = gt.cuda()

            # 计算Dice系数（1 - Dice损失）
            dice_loss = dice_loss_fn(pred, gt)
            dice_score = 1 - dice_loss.item()

            # 计算交叉熵损失（需将概率转为logits）
            logits = torch.log(pred + 1e-8)  # 避免log(0)
            ce_loss = ce_loss_fn(logits, gt.squeeze(1)).item()  # gt去除通道维度

            # 计算IoU
            iou = compute_iou(pred, gt)

            # 保存结果
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

    # 保存到CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"二分类对比结果已保存至 {output_csv}")


if __name__ == "__main__":
    # 替换为你的文件夹路径
    PREDICTION_FOLDER = "/mnt/sda/zhouying/2.48/SWCNN-main_multi/same"  # 预测图文件夹
    GROUND_TRUTH_FOLDER = "/mnt/sda/zhouying/2.48/SWCNN-main_multi/ground_truth"  # 标签文件夹
    
    # 运行对比
    compare_binary_folders(
        pred_folder=PREDICTION_FOLDER,
        gt_folder=GROUND_TRUTH_FOLDER,
        output_csv="binary_results.csv"
    )
