import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import *
# Basic rule is sell all stock positions at the end of the day and save all money as current fund. At the beginning of next day, follow given strategies and buy stock or not.

# Benchmark strategy
# default strategy: buy stocks at start, always keep them until the end of simulation
def invest_by_default(real:list)-> list: 
    balance = []
    position = 100
    balance.append(position)
    for i in range(1, len(real)):
        position = position * real[i]/real[i-1]
        balance.append(position)
    return balance
    # balance describes the current fund of each end of days.

# Benchmark strategy
# Theoretical maximum strategy: Investors can see into the future and always know the true future stock price, so he can always get the maximum profit.
def invest_by_maximum(real:list)-> list:
    balance = []
    position = 100
    # buy_price = real[0]
    balance.append(position)
    for i in range(1, len(real)):
        if real[i] > real[i-1]:
            position = position * real[i]/real[i-1]
            # buy_price = real[i]
        balance.append(position)
    print('maximum:', balance[-1]/100)
    return balance

# Baseline strategy
def invest_by_baseline(real:list, baseline:list)-> list:
    balance = []
    position = 100
    # buy_price = real[0]
    balance.append(position)
    for i in range(1, len(baseline)):
        if baseline[i] > baseline[i-1]:
            position = position * real[i]/real[i-1]
            # buy_price = real[i]
        balance.append(position)
    print('baseline:', balance[-1]/100)
    return balance

# def draw_test():
#     x = range(40) 
#     default = invest_by_default()
#     god = invest_by_GOD()
#     baseline = invest_by_baseline()
#     ourMethod = invest_by_ourMethod()
#     strategy = invest_by_strategy()
#     strategy1 = invest_by_strategy1()

#     plt.figure(figsize=(10,5))
#     plt.title("AVG ROR : average rate of return")
#     plt.xlabel("Time(Days)")
#     # plt.plot(x, default, '-', label='default')
#     plt.plot(x, baseline, '-', label='baseline')
#     plt.plot(x, strategy, '-', label='ourmethod')
#     plt.plot(x, strategy1, '-', label='ourmethod+period')
#     plt.plot(x, ourMethod, '-', label='baseline+period')
#     plt.plot(x, god, '-',label='Theoretical maximum')
#     plt.legend()
#     plt.grid()
#     plt.show()

# strategy
# Here, invest is the percentage of our funds we use to buy stock, and the change percentage is computed by dividing the predicted change in the market tomorrow by the price today. The default value of p is 25% when not take periodic information into consideration.

def invest_by_strategy(real:list, predicted:list, percent=0.25)-> list:
    balance = []
    position = 100
    # buy_price = real[0]
    balance.append(position)
    # percent = 0.25
    for i in range(1, len(predicted)):
        change_percent = (predicted[i] - predicted[i-1])/predicted[i-1]
        if change_percent > 0.01:
            position = position * real[i]/real[i-1]
            # buy_price = real[i]
        # elif change_percent > 0.005:
        #     position = 0.5*position * real[i]/real[i-1] + 0.5*position
        elif change_percent > -0.025:
            position = percent*position * real[i]/real[i-1] + (1-percent)*position
        balance.append(position)
    print('strategy:', balance[-1]/100)
    return balance

def calculate_pvalue(stock_id:int, start_date=st, end_date=end, target_date:str)-> float:
    periodinfo = pd.read_csv('./data/PeriodicOLR.csv')
    price = pd.read_csv('./data/processed_price.csv')
    ids = periodinfo['name'].to_list()
    p = 0.25
    # start, end data => percentage of test days in period

    if stock_id not in ids:
        return p
    else:
        period = periodinfo.loc[periodinfo['name'] == stock_id, 'period'].to_list()
        # period format is [-1 1 -1 .......]

        # index = ids.index(stock_id)
        # start_date = periodinfo['start'][index]
        # end_date = periodinfo['end'][index]
        start = price.loc[price['date'] == start_date, 'close'].to_list()[0]
        end = price.loc[price['date'] == end_date, 'close'].to_list()[0]
        
        # date format: '2020-01-01'
        start_month = int(start[5:7])
        end_month = int(end[5:7])
        target_month = int(target_date[5:7]) 
        # print(start_month, end_month, target_month)
        p = p + (period[target_month]+ period[target_month+1])*0.05
        return p
        
    
