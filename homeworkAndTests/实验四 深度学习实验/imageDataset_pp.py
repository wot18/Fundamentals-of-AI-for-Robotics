import os
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
from paddle.vision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置随机种子，保证结果可复现
paddle.seed(42)
np.random.seed(42)

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = []
        self.class_to_idx = {}
        
        # 读取txt文件中的图像路径
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        # 获取所有类别
        classes = set()
        for line in lines:
            class_name = line.strip().split('/')[0]
            classes.add(class_name)
        
        # 为每个类别分配索引
        self.classes = sorted(list(classes))
        for i, cls in enumerate(self.classes):
            self.class_to_idx[cls] = i
        
        # 处理图像路径和标签
        for line in lines:
            img_path = line.strip()
            class_name = img_path.split('/')[0]
            class_idx = self.class_to_idx[class_name]
            self.img_labels.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        # 构建完整的图像路径
        full_img_path = os.path.join(self.root_dir, img_path)
        
        try:
            # 读取图像并转换为RGB格式
            image = Image.open(full_img_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {full_img_path}: {e}")
            # 如果图像加载失败，返回一个随机图像和标签
            return paddle.randn([3, 128, 128]), label