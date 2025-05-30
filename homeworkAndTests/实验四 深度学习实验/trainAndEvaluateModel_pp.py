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

# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            # 统计损失和准确率
            running_loss += loss.item() * inputs.shape[0]
            predicted = paddle.argmax(outputs, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
        
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accs

# 定义测试函数
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with paddle.no_grad():
        for inputs, labels in test_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            test_loss += loss.item() * inputs.shape[0]
            predicted = paddle.argmax(outputs, axis=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和真实标签
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return test_loss, test_acc, all_preds, all_labels