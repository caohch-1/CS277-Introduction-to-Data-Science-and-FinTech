from os import read
import tushare as ts
import pandas as pd
import numpy as np
import time
prefix = 'data/'
path = prefix + '{}.csv'

'数据来源于tushare，https://www.tushare.pro/document/2，需先获取权限，设置秘钥'
'代码和数据在https://github.com/Chocolita/SSLT'
'秘钥已改变，请下载github的数据集，将from_csv设为True进行数据预处理'
ts.set_token('90f542ce349ae3c72715530fdb411ddec58c505cd1bede27cfe43525')
pro = ts.pro_api()

def read_symbol(from_csv = False):
    print('reading symbol...')
    if from_csv:
        return pd.read_csv(path.format('symbol'), index_col=['ts_code'])
    sse_stock = pro.stock_basic(exchange='SSE', list_status='L', \
                    fields='ts_code,symbol,name,area,industry,list_date')
    szse_stock = pro.stock_basic(exchange='SZSE', list_status='L', \
                        fields='ts_code,symbol,name,area,industry,list_date')
    symbol = pd.concat([sse_stock, szse_stock], axis=0)
    symbol.set_index(['ts_code'], inplace=True)
    symbol.to_csv(path.format('symbol'), index=1)
    return symbol

def read_price(st = '20140101', end = '20211031', from_csv = False):
    print('reading pirce...')
    if from_csv:
        return pd.read_csv(path.format('price_all'), index_col=['ts_code'])
    trade_date = pro.trade_cal(exchange='SSE', is_open='1', 
                        start_date=st, end_date=end, fields='cal_date')
    def get_daily(date):
        for _ in range(100):
            try: df = pro.daily(trade_date=date)
            except: time.sleep(1)
            else: return df
    price_all = []
    vis = {}
    for date in trade_date['cal_date']:
        if int(int(date)/100) not in vis:
            print('reading price in %s' % date)
            vis[int(int(date)/100)] = 1
        price = get_daily(date)
        price_all.append(price)
    price = pd.concat(price_all, axis=0)
    price.set_index(['ts_code'], inplace=True)
    price.to_csv(path.format('price_all'), index=1)
    return price

def add_price(st = '20201101', end = '20211031'):
    df1 = read_price(from_csv=True)
    df2 = read_price(st=st, end=end, from_csv=False)
    df = pd.concat([df1, df2], axis=0)
    df.to_csv(path.format('price_all'), index=1)
    return df

def pretreat_price(df: pd.DataFrame = None):
    print('pretreating price...')
    if df is None:
        return pd.read_csv(path.format('price'), index_col=['ts_code'])
    df = df[['trade_date', 'close']]
    df = df.pivot_table(index=['ts_code'], columns=['trade_date'], values=['close'])
    df = df.xs('close', axis = 1, drop_level = True)
    
    df.index = df.index.map(lambda x : x[:6])
    index_val =  df.index.map(lambda x : int(x))
    
    '保留上证和深证股票'
    df = df[((index_val / 1000).astype(np.int) == 600) | \
            ((index_val / 1000).astype(np.int) == 688) | \
            ((index_val / 10000).astype(np.int) == 00) | \
            ((index_val / 10000).astype(np.int) == 30)]

    '新股不做考虑，这里使用的标准是最后300个交易日至少有270个值'
    last = df.T.tail(300).T
    df = df[last.isnull().sum(axis=1) <= 30]
    df.to_csv(path.format('price'), index=1)
    return df

def read_report(_type = None, st = '20140101', end = '20211031', from_csv = False):
    '''
    指标说明     https://waditu.com/document/2?doc_id=16
    财务指标     api = pro.fina_indicator
    利润表       api = pro.income
    资产负债表   api = pro.balancesheet
    现金流量表   api = pro.cashflow
    '''
    print('reading %s...' % _type)
    if from_csv:
        return pd.read_csv(path.format(_type), index_col=['ts_code', 'end_date'])
    if _type == '财务指标': api = pro.fina_indicator
    elif _type == '利润表': api = pro.income
    elif _type == '资产负债表': api = pro.balancesheet
    elif _type == '现金流量表': api = pro.cashflow
    else: raise NameError
    def get_one(code):
        for _ in range(100):
            try: df = api(ts_code=code, start_date=st, end_date=end)
            except: time.sleep(1)
            else: return df
    report_all = []
    symbol = read_symbol(from_csv=True)
    for code in symbol.index:
        print('reading %s in %s' % (_type, code))
        report = get_one(code)
        report.sort_values('end_date', inplace=True)
        report_all.append(report)
        
    report = pd.concat(report_all, axis=0)
    report.set_index(['ts_code', 'end_date'], inplace=True)
    report = report[~report.index.duplicated()]
    report.to_csv(path.format(_type), index=1)
    return report

def join_report(from_csv = False):
    if from_csv:
        return pd.read_csv(path.format('info'), index_col=['ts_code', 'end_date'])
    def __delete_one_indicator(df: pd.DataFrame, column):
        try:
            df.drop(column, axis=1, inplace=True)
        finally:
            return df
    def __delete_muti_indicator(df: pd.DataFrame, columns):
        for col in columns:
            df = __delete_one_indicator(df, col)
        return df
    def __treat(name):
        report = read_report(name, from_csv = True)
        report.index = report.index.map(lambda x : (x[0][:6], x[1]))
        report = __delete_muti_indicator(report, ['ann_date', 'f_ann_date', \
                'report_type', 'update_flag', 'end_type'])
        '删除提供信息的股票数不到总数90%的因子'
        report = report.T[report.isnull().sum(axis=0) < len(report) * (1 - 0.9)].T
        return report
    
    indicator = __treat('财务指标')
    income = __treat('利润表')
    balancesheet = __treat('资产负债表')
    cashflow = __treat('现金流量表')
    info = pd.concat([indicator, income, balancesheet, cashflow], axis=1)
    info: pd.DataFrame
    info = info.T[~info.columns.duplicated()].T
    '删除丢失至少30%信息的股票（约400支）'
    info = info[info.isnull().sum(axis=1) < len(info.columns) * 0.3]
    '对缺失值使用中位数填充'
    info.fillna(value=info.median(axis=0), inplace=True)
    info.to_csv(path.format('info'), index=1)
    return info

def match(price: pd.DataFrame, info: pd.DataFrame):
    print('matching price and info...')

    stock = set(info.index.droplevel('end_date'))
    idx = price.index.map(lambda x : x in stock)
    price = price[idx]

    stock = set(price.index)
    idx = info.index.map(lambda x : x[0] in stock)
    info = info[idx]
    
    price.to_csv(path.format('pt_price'), index=1)
    info.to_csv(path.format('pt_info'), index=1)

if __name__ == '__main__':
    # price = read_price(from_csv=True)
    # price = pretreat_price(df=price)
    
    price = pretreat_price(df=None)
    
    # read_report('财务指标', from_csv = False)
    # read_report('利润表', from_csv = False)
    # read_report('资产负债表', from_csv = False)
    # read_report('现金流量表', from_csv = False)
    
    # join_report(from_csv=False)
    
    info = join_report(from_csv=True)
    match(price, info)
