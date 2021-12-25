import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import matplotlib.pyplot as plt

from Bio.Cluster import kcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import random

from numpy.ma.core import sort

random.seed(0)
np.random.seed(0)

prefix = 'data/'
outputprefix = 'processed_data/'
path = prefix + '{}.csv'
img_path = 'image/' + '{}.png'
output_path = outputprefix + '{}.csv'

def read():
    '读入数据，截取最后4个季度的季报，删除异常点'
    info = pd.read_csv(path.format('pt_info'), index_col=['ts_code'])
    info = info.groupby(info.index)
    def join(x: pd.DataFrame):
        x = x[-4:]
        if len(x) != 4:
            return None
        x.drop(['end_date'], axis=1, inplace=True)
        col = x.columns
        x = x.to_numpy()
        x = x.mean(axis=0)
        x = np.expand_dims(x, axis=0)
        return pd.DataFrame(x, index=None, columns=col)
    info = info.apply(join)
    info.index = info.index.map(lambda x : x[0])
    return info

def show(x, label, output_file = None):
    rng = np.random.RandomState(0)
    pca = PCA(n_components=2)
    pca.fit(x)
    y = pca.transform(x)
    plt.ylim((-5, 5))
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    # sizes = x * rng.rand(50)  # 随机产生50个用于改变散点面积的数值
    plt.scatter(y[:, 0], y[:, 1], marker='.', c=label, s=1)
    if output_file is not None:
        plt.title(output_file)
        plt.savefig(img_path.format(output_file.replace(':', ' ')), dpi=600)

def pretreat1(x: np.ndarray):
    '标准化'
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def cluster1(df: pd.DataFrame, max_size: int = 10):
    '每一类样本数不超过max_size，聚类方式为kmeans+cosine'
    now = 0
    data = df.to_numpy()
    data = pretreat2(data)
    res = np.zeros(len(data), dtype=int)
    id = np.arange(len(data), dtype=int)

    def cluster_recursion(x, id):
        nonlocal now, res
        if len(x) > max_size:
            label = kcluster(x, nclusters=2, dist='u', npass=100)[0]
            order = np.argsort(label)
            x, id = x[order], id[order]
            count0 = len(label[label == 0])
            
            x1, id1 = x[:count0], id[:count0]
            x2, id2 = x[count0:], id[count0:]

            '二分聚类，归于0类的保持类标签不变，归于1类划到新类'
            now += 1
            res[id2] = now
            cluster_recursion(x1, id1)
            cluster_recursion(x2, id2)
    
    cluster_recursion(data, id)
    show(data, res, output_file = 'cluster1-' + str(max_size))
    res = pd.DataFrame(res, index=df.index)
    res.to_csv(output_path.format('cluster1-' + str(max_size)))
    return res

def cluster1_diff_size(info: pd.DataFrame, sizes: list):
    '使用kmeans聚类，sizes为不同类最大样本数的列表'
    for size in sizes:
        print('cluster with max_size %d' % size)
        cluster1(info, size)
        
def pretreat2(x: np.ndarray):
    def logize(x):
        '我们认为1亿和10亿；10亿和100亿是比较接近的，所以标准化前先取对数'
        x_sgn = 1 if x >= 0 else -1
        x_abs = max(abs(x), 1)
        return np.math.log(x_abs) * x_sgn
    logize_v = vectorize(logize)
    '此处对所有值都取对数'
    is_number = (np.sum(np.abs(x) >= 100, axis=0) / len(x)) > -1
    y = x.copy()
    y.T[is_number] = logize_v(x.T[is_number])
    x = y
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return x

def dist_for_cluster2(x: np.ndarray, k = 0.25):
    '综合考虑欧氏距离和余弦相似度，计算两只股票的距离'
    '距离1使用欧氏距离，距离2使用余弦相似度，两者结合，只有两者都比较接近时，距离较小'
    'd1和d2结合的公式的灵感来自于角点响应函数：R=det(M)-k(trace(M)^2)=(L1*L2)-k(L1+L2)^2'
    '此处d = d1 * d2 + k(d1 + d2)^2'
    'd1为欧氏距离，归一化到[0,2]，d2为余弦相似度，范围也为[0,2]'
    d1 = pairwise_distances(x, x, metric = 'euclidean')
    d1 = 2 * d1 / np.max(d1)
    d2 = pairwise_distances(x, x, metric = 'cosine')
    dist = d1 * d2 + k * (d1 + d2) ** 2
    return dist

def cluster2(df: pd.DataFrame, k = 5, dist = None):
    '使用层次聚类+自定义距离，k为类个数'
    data = df.to_numpy()
    data = pretreat2(data)
    if dist is None:
        dist = dist_for_cluster2(data)

    model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    label = model.fit_predict(dist)
    show(data, label, output_file = 'cluster2-' + str(k))
    res = pd.DataFrame(label, index=df.index)
    res.to_csv(output_path.format('cluster2-' + str(k)))
    return res
    
def cluster2_diff_k(info: pd.DataFrame, nums: list):
    '使用层次聚类，nums为聚类个数'
    data = info.to_numpy()
    data = pretreat2(data)
    dist = dist_for_cluster2(data)
    for k in nums:
        print('cluster with k %d' % k)
        cluster2(info, k, dist = dist)

def factor_selection(df: pd.DataFrame, from_file = False):
    '筛选因子，不断在相关系数最小的2个因子里删除，如果已经筛选过并保存到文件，更改from_file=True'
    if from_file:
        with open(outputprefix + 'factors.txt', 'r') as f:
            factors = f.readline()
            return df[eval(factors)]
    df.drop('comp_type', axis=1, inplace=True)
    pre_select = df[['total_revenue', 'free_cashflow', 'roe']]
    df.drop(['total_revenue', 'free_cashflow', 'roe'], axis=1, inplace=True)
    df = pd.concat([pre_select, df], axis=1)
    x = df.to_numpy()
    x = pretreat2(x)
    y = pd.DataFrame(x, index=df.index, columns=df.columns)
    corr_df = np.abs(y.corr())
    corr_np = corr_df.to_numpy()

    size = len(corr_np)
    corr_l = list()
    removed = dict()
    for i in range(size):
        for j in range(i + 1, size):
            corr_l.append((i, j, corr_np[i][j]))
    corr_l.sort(key = lambda x: x[2], reverse=True)
    for i, j, corr in corr_l:
        if corr <= 0.8: break
        if i in removed or j in removed: continue
        removed[j] = 1
    rest = []
    mp = {}
    for i in range(size):
        if i not in removed:
            mp[len(rest)] = i
            rest.append(df.columns[i])

    size = len(rest)
    new_corr = [[0.0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            new_corr[i][j] = corr_np[mp[i]][mp[j]]
    new_corr = np.array(new_corr)
    from itertools import combinations
    best = None
    min_now = 1e9
    for comb in combinations(range(3, size), 5):
        comb = [0, 1, 2] + list(comb)
        i, j = zip(*combinations(comb, 2))
        s = np.sum(new_corr[i, j])
        if s < min_now:
            min_now = s
            best = comb

    factors = [rest[i] for i in best]
    with open(outputprefix + 'factors.txt', 'w') as f:
        f.write(str(factors))
    df = df[factors]
    return df

if __name__ == '__main__':
    info = read()

    # info = info[['total_revenue', 'free_cashflow', 'roe', 'total_cogs', \
    #     'total_cur_assets', 'fix_assets_total' , 'bps', 'ebt_yoy']]
    
    info = factor_selection(info, from_file=True)
    
    info.to_csv(output_path.format('test'))
    
    # x = info.to_numpy()
    # x = pretreat2(x)
    # y = pd.DataFrame(x, index=info.index, columns=info.columns)
    # print(y.corr())
    # pca = PCA(n_components=0.9)
    # pca.fit(x)
    # print(pca.explained_variance_ratio_)

    '#! 特别注意，聚类一传入的是聚类后每一类最大样本数size，聚类二传入的是聚类后类的个数k'
    cluster1_diff_size(info, sizes = [5, 10, 20, 50, 100, 200, 500])
    cluster2_diff_k(info, [5, 10, 20, 50, 100, 200, 500])
    