import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import csv
import random
import re

from pandas.io.parsers import read_csv


def limit_update(arr:list, data:list, tag):
    #if tag = 0, min   else max
    arr.append(data)
    #first price, then month
    res = sorted(arr, key = lambda arr:(arr[0], arr[1]))

    #save max
    if tag == 0:
        arr.remove(arr[0])
    #save min
    else:
        arr.remove(arr[3])




def main():
    file = pd.read_csv('./data/price.csv')
    temp = file.dropna(inplace=False)
    temp.to_csv('./processed_data/drop.csv')
    data = open('./processed_data/drop.csv')
    exampleReader = csv.reader(data)  # 读取csv文件
    exampleData = list(exampleReader)  # csv数据转换为列表
    num_row = len(exampleData)
    length_yuan = len(exampleData[0])

    x = list()
    y1 = list()
    y2 = list()

    for i in range(2, length_yuan):
        if i - 15 * int(i/15) != 0:
            continue
        x.append(exampleData[0][i])

    #10, 3、4月
    data = 11
    for i in range(2, length_yuan):
        if i - 15 * int(i/15) != 0:
            continue
        y1.append(float(exampleData[data][i]))
        y2.append(float(exampleData[data+1][i]))



    # plt.figure(figsize=(60, 10))
    # plt.xticks(rotation=45)
    # plt.tick_params(labelsize=5.5)
    # plt.subplot(2,1,1)
    # plt.xticks(rotation=45)
    # plt.tick_params(labelsize=5.5)
    # plt.plot(x, y1)  # 绘制x,y的折线图
    # plt.subplot(2,1,2)
    # plt.xticks(rotation=45)
    # plt.tick_params(labelsize=5.5)
    # plt.plot(x, y2)  # 绘制x,y的折线图
    # plt.show()

    high = []
    low = []
    for i in range(0, 12):
        high.append([])
        low.append([])

    limit_max = []
    limit_min = []

    max_list = []
    min_list = []

    for i in range(1, num_row):
        #14-20, total 7
        num_year = 0
        curr_year = 0
        max_price = []
        min_price = []
        limit_max = []
        limit_min = []
        year = 2014
        curr_year = 2014
        tag = 0
        test_max = -999
        for j in range(2, length_yuan):
            if j - 10 * int(j/10) != 0:
                continue
            str = exampleData[0][j]
            pattern = re.search(r'\d{4}(.*?)\d{2}', str)
            year = int(pattern.group()[0:4])
            month = int(pattern.group()[4:6])
            price = float(exampleData[i][j])
            #when into a new year, reset
            if year != curr_year:
                num_year += 1
                tag = 0

                curr_year = year
                # print(curr_year)
                # print(max_price)
                # print(min_price)

                # save the current result, then empty the array
                #
                if max_price[2][1] == max_price[1][1]:
                    limit_max.append(max_price[2][1])
                elif max_price[2][1] == max_price[0][1]:
                    limit_max.append(max_price[2][1])
                elif max_price[1][1] == max_price[0][1]:
                    limit_max.append(max_price[1][1])
                else:
                    limit_max.append(0)

                if min_price[2][1] == min_price[1][1]:
                    limit_min.append(min_price[2][1])
                elif min_price[2][1] == min_price[0][1]:
                    limit_min.append(min_price[2][1])
                elif min_price[1][1] == min_price[0][1]:
                    limit_min.append(min_price[1][1])
                else:
                    limit_min.append(0)
                max_price = []
                min_price = []
                if year == 2021:
                    break
                #evaluate


            if tag < 3:
                max_price.append([price, month])
                min_price.append([price, month])
            else:
                #save the max/min 3 data in a year
                max_price.append([price, month])
                min_price.append([price, month])
                # first price, then month
                max_price=sorted(max_price,key=(lambda x:x[0]),reverse=False)
                min_price = sorted(min_price, key=(lambda x: x[0]), reverse=False)
                # save max
                max_price.remove(max_price[0])
                min_price.remove(min_price[3])
                #print(max_price)
            tag += 1




        if abs(limit_max[5] - limit_max[6]) <= 3 and limit_max[5] != 0 and limit_max[6] != 0:
            tup = [int(exampleData[i][1]), int((limit_max[5] + limit_max[6])/2)]
            max_list.append(tup)
        if abs(limit_min[5] - limit_min[6]) <= 3 and limit_min[5] != 0 and limit_min[6] != 0:
            tup = [int(exampleData[i][1]), int((limit_min[5] + limit_min[6])/2)]
            min_list.append(tup)
        limit_max.sort()
        limit_min.sort()
        # print(limit_max)
        # print(limit_min)
        tag_max = -5
        count = 0
        for k in range(len(limit_max)):
            if count >= 3 and tag_max != 0:
                # print("found")
                # print(limit_max)
                #ts_code
                tup = [int(exampleData[i][1]), int((limit_max[k-1] + limit_max[k-2] + limit_max[k])/3)]
                max_list.append(tup)
                break
            if abs(tag_max - limit_max[k]) >= 4:
                tag_max = limit_max[k]
                count = 1
            elif abs(tag_max - limit_max[k]) < 4:
                count += 1

        tag_min = -5
        count = 0
        for k in range(len(limit_min)):
            if count >= 3 and tag_min != 0:
                print("found")
                print(limit_min)
                # ts_code
                tup = [int(exampleData[i][1]), int((limit_min[k-1] + limit_min[k-2] + limit_min[k])/3)]
                min_list.append(tup)
                break
            if abs(tag_min - limit_min[k]) >= 3:
                tag_min = limit_min[k]
                count = 1
            elif abs(tag_min - limit_min[k]) < 3:
                count += 1


        # if i == 100:
        #     break
    max_list=sorted(max_list,key=(lambda x:x[0]),reverse=False)
    min_list = sorted(min_list, key=(lambda x: x[0]), reverse=False)
    print(max_list)
    print(min_list)
    max_arr = np.array(max_list)
    max_df = pd.DataFrame(max_arr)
    max_df.to_csv('./Processed_data/PeriodicMax.csv')
    min_arr = np.array(min_list)
    min_df = pd.DataFrame(min_arr)
    min_df.to_csv('./Processed_data/PeriodicMin.csv')






if __name__ == '__main__':
    main()

