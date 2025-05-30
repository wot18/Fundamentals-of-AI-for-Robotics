import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
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
            return torch.randn(3, 128, 128), label

# 定义基于ResNet的神经网络模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetClassifier, self).__init__()
        # 加载预训练的ResNet模型作为特征提取器
        # 由于没有标准的ResNet-30，我们使用ResNet-34作为替代
        self.feature_extractor = models.resnet34(pretrained=pretrained)
        
        # 获取ResNet最后一层的输出特征数
        feature_size = self.feature_extractor.fc.in_features
        
        # 移除ResNet的全连接层
        self.feature_extractor.fc = nn.Identity()
        
        # 添加自定义的三层全连接网络
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # 使用ResNet提取特征
        features = self.feature_extractor(x)
        # 通过全连接层进行分类
        return self.classifier(features)

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accs

# 定义测试函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return test_loss, test_acc, all_preds, all_labels

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    #plt.savefig('confusion_matrix.png')
    plt.show()

# 绘制训练过程
def plot_training_process(train_losses, train_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    #plt.savefig('training_process.png')
    plt.show()

# 主函数
def main():
    # 设置参数
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 定义训练数据变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((144, 144)),  # 稍大一些的尺寸，便于随机裁剪
        transforms.RandomCrop(128),    # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),         # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    # 定义测试数据变换（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 缩放图像为128x128
        transforms.ToTensor(),         # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    
    # 创建数据集
    train_dataset = ImageDataset(
        txt_file='train.txt',
        root_dir='dataset',
        transform=train_transform  # 使用带数据增强的变换
    )
    
    test_dataset = ImageDataset(
        txt_file='test.txt',
        root_dir='dataset',
        transform=test_transform  # 使用不带数据增强的变换
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 获取类别数量
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    # 创建基于ResNet的模型
    model = ResNetClassifier(num_classes=num_classes, pretrained=True).to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("\nTraining model...")
    train_losses, train_accs = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # 绘制训练过程
    plot_training_process(train_losses, train_accs)
    
    # 评估模型
    print("\nEvaluating model...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(model, test_loader, criterion, device)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, train_dataset.classes)
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
    
    # 保存模型
    torch.save(model.state_dict(), 'resnet_classifier_model.pth')
    print("Model saved as 'resnet_classifier_model.pth'")

if __name__ == "__main__":
    main()