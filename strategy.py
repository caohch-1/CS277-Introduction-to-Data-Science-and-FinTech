import numpy as np
import matplotlib.pyplot as plt

# Benchmark strategy
# default strategy: buy stocks at start, always keep them

def invest_by_default(real:list)-> list: 
    balance = []
    position = 100
    balance.append(position)
    for i in range(1, len(real)):
        position = position * real[i]/real[i-1]
        balance.append(position)
    return balance