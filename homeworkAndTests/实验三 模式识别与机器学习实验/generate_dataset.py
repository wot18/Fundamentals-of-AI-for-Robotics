import os
import random

def generate_dataset_files(dataset_path, train_ratio=0.8):
    # 获取所有类别
    categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    train_lines = []
    test_lines = []
    
    # 遍历每个类别文件夹
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        # 获取该类别下的所有图片
        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        # 随机打乱图片顺序
        random.shuffle(images)
        
        # 计算训练集数量
        train_size = int(len(images) * train_ratio)
        
        # 分配到训练集和测试集
        for i, image in enumerate(images):
            line = f'{category}/{image}\n'
            if i < train_size:
                train_lines.append(line)
            else:
                test_lines.append(line)
    
    # 写入文件
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

if __name__ == '__main__':
    dataset_path = 'dataset'
    generate_dataset_files(dataset_path)