import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 设置散点形状
maker = ['s', 'o', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['r', 'green', 'blue', 'cyan', 'blue', 'lime', 'violet', 'm', 'peru', 'olivedrab', 'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }

def plot_embedding_2D(data, label, title):
    #x_min, x_max = np.min(data, 0), np.max(data, 0)
    #data = (data - x_min) / (x_max - x_min)

    for index in range(3):
        data_i = data[label==index]
        print(data_i.shape)
        plt.scatter(data_i[:,0], data_i[:,1], cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
        plt.xticks([])
        plt.yticks([])
    plt.title(title, fontsize=32, fontweight='normal', pad=20)

def vis_2d(data, label, title, save_path):
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)
    #norm = np.linalg.norm(result_2D, ord=2, axis=1, keepdims=True)
    #result_2D = result_2D / norm
    fig = plt.figure(figsize=(10,10))
    plot_embedding_2D(result_2D, label, title)
    plt.savefig(save_path)
    plt.show()