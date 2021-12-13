import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation

from Bio.Cluster import kcluster, clustercentroids

prefix = 'data/'
outputprefix = 'processed_data/'
path = prefix + '{}.csv'
img_path = outputprefix + '{}.png'
output_path = outputprefix + '{}.csv'

def read():
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

def cluster(df: pd.DataFrame):
    '聚类方法：kmeans+余弦相似度，每一类不超过10个样本'
    data = df.to_numpy()
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    now = 0
    res = np.zeros(len(data), dtype=int)
    id = np.arange(len(data), dtype=int)

    def cluster_recursion(x, id):
        nonlocal now, res
        if len(x) > 10:
            label, _, _ = kcluster(x, nclusters=2, dist='u', npass=100)
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
    # info = info[['total_revenue', 'bps']]
    info.to_csv(output_path.format('test'))

    label = cluster(info)
    label.to_csv(output_path.format('cluster'))
    