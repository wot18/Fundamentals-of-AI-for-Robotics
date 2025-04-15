from flask import Flask, render_template, jsonify
import numpy as np
from sklearn.datasets import make_blobs
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# 全局变量存储聚类状态
class KMeansState:
    def __init__(self):
        self.X = None
        self.centers = None
        self.labels = None
        self.iteration = 0
        self.k = 3
        self.finished = False

state = KMeansState()

def plot_to_base64():
    plt.figure(figsize=(8, 6))
    if state.labels is None:
        plt.scatter(state.X[:, 0], state.X[:, 1], c='black', s=50)
    else:
        plt.scatter(state.X[:, 0], state.X[:, 1], c=state.labels, cmap='viridis', s=50)
    
    if state.centers is not None:
        plt.scatter(state.centers[:, 0], state.centers[:, 1], c='red', 
                   marker='*', s=200, label='Cluster Centers')
    
    plt.title(f'Iteration {state.iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # 将图像转换为base64字符串
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode()

@app.route('/')
def home():
    # 初始化数据
    state.X, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.60, random_state=0)
    state.iteration = 0
    state.finished = False
    
    # 随机初始化聚类中心
    n_samples = state.X.shape[0]
    centers_idx = np.random.choice(n_samples, state.k, replace=False)
    state.centers = state.X[centers_idx]
    state.labels = None
    
    return render_template('index.html', image=plot_to_base64())

@app.route('/iterate')
def iterate():
    if state.finished:
        return jsonify({'image': plot_to_base64(), 'finished': True})
    
    # 计算每个点到聚类中心的距离
    distances = np.sqrt(((state.X - state.centers[:, np.newaxis])**2).sum(axis=2))
    state.labels = np.argmin(distances, axis=0)
    
    # 更新聚类中心
    new_centers = np.array([state.X[state.labels == i].mean(axis=0) for i in range(state.k)])
    
    # 检查是否收敛
    if np.all(state.centers == new_centers):
        state.finished = True
    
    state.centers = new_centers
    state.iteration += 1
    
    return jsonify({
        'image': plot_to_base64(),
        'finished': state.finished
    })

if __name__ == '__main__':
    app.run(debug=True)