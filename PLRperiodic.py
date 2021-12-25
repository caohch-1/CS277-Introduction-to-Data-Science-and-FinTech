import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import csv
import random
import re

from pandas.io.parsers import read_csv


def main():
    file = pd.read_csv('./data/price.csv')
    temp = file.dropna(inplace=False)
    temp.to_csv('./processed_data/drop.csv')
    data = open('./processed_data/drop.csv')
    exampleReader = csv.reader(data)  # 读取csv文件
    exampleData = list(exampleReader)  # csv数据转换为列表
    num_row = len(exampleData)
    length_yuan = len(exampleData[0])

    res = []

    for i in range(1, num_row):
        # 14-20, total 7
        num_year = 0
        curr_year = 0

        Derivative = []
        for q in range(0, 7):
            Derivative.append([])
        price_per_year = []
        for q in range(0, 7):
            price_per_year.append([])

        year = 2014
        curr_year = 2014
        curr_month = 12
        tag = 0
        for j in range(3, length_yuan):
            str = exampleData[0][j]
            pattern = re.search(r'\d{4}(.*?)\d{2}', str)
            year = int(pattern.group()[0:4])
            month = int(pattern.group()[4:6])
            price = float(exampleData[i][j])
            if year == 2021:
                break
            if year != curr_year:
                curr_year += 1
            if month == curr_month:
                continue
            curr_month = month
            price_per_year[curr_year-2014].append(price)

        for k in range(0, 7):
            for j in range(0, 11):
                if price_per_year[k][j+1] - price_per_year[k][j] >= 0:
                    Derivative[k].append(1)
                else:
                    Derivative[k].append(-1)
        #print(Derivative)

        #Method 1
        connect = []
        count = 0
        for m in range(0, 11):
            if Derivative[-1][m] == Derivative[-2][m]:
                count += 1
        if count >= 6:
            ts = exampleData[i][2]
            tup = [ts]
            for m in range(0, 11):
                tup.append(Derivative[-1][m])
            res.append(tup)
            continue

        connect = []
        count = 0
        for k in range(0, 7):
            for j in range(0, 7):
                #count = 0
                if k==j:
                    continue
                for m in range(0, 11):
                    if Derivative[k][m] == Derivative[j][m]:
                        count += 1
                # if count >= 8:
                #     connect.append([k+2014, j+2014])
        if count >= 30:
            ts = exampleData[i][2]
            tup = [ts]
            for m in range(0, 11):
                tup.append(Derivative[-1][m])
            res.append(tup)

    # for item in res:
    #     print(item)

    plr_arr = np.array(res)
    plr_df = pd.DataFrame(plr_arr)
    plr_df.to_csv('./Processed_data/PeriodicPLR.csv')





if __name__ == '__main__':
    main()
