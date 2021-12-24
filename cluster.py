import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd
import matplotlib.pyplot as plt

from Bio.Cluster import kcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import random

random.seed(0)
np.random.seed(0)

prefix = 'data/'
outputprefix = 'processed_data/'
path = prefix + '{}.csv'
img_path = outputprefix + '{}.png'
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

def cluster1(df: pd.DataFrame, max_size: int = 10):
    '每一类样本数不超过max_size，聚类方式为kmeans+cosine'
    now = 0
    data = df.to_numpy()
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
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
    res = pd.DataFrame(res, index=df.index)
    return res

def cluster1_diff_size(info: pd.DataFrame, sizes: list):
    '使用kmeans聚类，sizes为不同类最大样本数的列表'
    for size in sizes:
        print('cluster with size %d' % size)
        label = cluster1(info, size)
        label.to_csv(output_path.format('cluster1-' + str(size)))

def dist_for_cluster2(x, k = 0.5):
    '综合考虑欧氏距离和余弦相似度，计算两只股票的距离'
    def logize(x):
        '我们认为1亿和10亿；10亿和100亿是比较接近的，所以标准化前先取对数'
        x_sgn = 1 if x >= 0 else -1
        x_abs = max(abs(x), 1)
        return np.math.log(x_abs) * x_sgn
    logize_v = vectorize(logize)
    '需要先判断那些是有量纲的。这里的判断依据是多数值大于1万的是有量纲的。'
    is_number = (np.sum(np.abs(x) >= 1e4, axis=0) / len(x)) > 0.5
    x.T[is_number] = logize_v(x.T[is_number])
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    
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
    if dist is None:
        data = df.to_numpy()
        dist = dist_for_cluster2(data)
    model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    label = model.fit_predict(dist)
    res = pd.DataFrame(label, index=df.index)
    return res

def cluster2_diff_k(info: pd.DataFrame, nums: list):
    '使用层次聚类，nums为聚类个数'
    data = info.to_numpy()
    dist = dist_for_cluster2(data)
    for k in nums:
        print('cluster with size %d' % k)
        label = cluster2(info, k, dist = dist)
        label.to_csv(output_path.format('cluster2-' + str(k)))


if __name__ == '__main__':
    info = read()
    '''
    利润表：营业总收入，营业总成本
    资产负债表：流动资产合计，固定资产合计
    现金流量表：自由现金流量
    财务指标：每股净资产，净资产收益率，利润总额同比增长
    '''
    info = info[['total_revenue', 'total_cogs', 'total_cur_assets', 'fix_assets_total',
                'free_cashflow', 'bps', 'roe', 'ebt_yoy']]
    info.to_csv(output_path.format('test'))
    # cluster1_diff_size(info, sizes = [5, 10, 20, 50, 100, 200, 500])

    cluster2_diff_k(info, [5, 10, 20, 50, 100, 200, 500])
    
    