# SSLT

The code for project "Standing on the Shoulders of Long-Term: Short-Term Stock Prediction Based on Financial Report and Quotations" in CS277, ShanghaiTech.

## Introduction

With the help of the development of deep learning technology, stock price prediction is becoming one of the most popular research directions. A lot of efforts are invested into short-term or long-term data in the previous research, but no one has tried to combine short-term and long-term data together to design an investment strategy. We propose a pipeline that utilizes both long-term and short-term data to output an investing strategy. We also conducted a simple simulated trading experiment, and the results show that the rate of return (ROR) of our method reached 12.56%.

### **What task does our code (method) solve?**

We propose a pipeline that utilizes the combination of the long-term (i.e., financial report ) and short-term (i.e., quotations) data to generate profitable investment strategies. Our approach uses long-term data to cluster stocks and analyze their periodicity, and then exploit short-term data to predict future prices. By taking the advantage of two types of data, our approach can be better than traditional methods that rely only on fundamental quantitative analysis techniques or historical stock prices.

## Code Structure

```
|--SSLT/
|   |--data/
|      |--price.csv 		/* Files contain stock price data
|      |--pt_info.csv 		/* Files contain financial report data
|      |--pt_price.csv 		/* Files contain preprocessed data
|      |--symbol.csv 		/* Files contain stock code, name and section
|      |--*表.csv 			/* Financial reports
|      |--财务指标.csv      /* Financial indicator
|   |--processed_data/ 		/* Directory for processed data
|   |--cluster.py 			/* Files for clustering stocks
|   |--PLRperiodic.py 		/* Files for PLR Periodicity Evaluation
|   |--predictor.py 		/*Files for price prediction based on clustering
|   |--read.py 				/*Files for getting raw dataset and preprocessing
|   |--strategy.py      /*Files for generate strategy and do evaluate simulation
```

We have upload the data sample. If you want to get the complete experiment results， please download the data from https://pan.baidu.com/s/1Au9-eJruJcWJ77tz-mV9BA with extraction code: 8jx3

### Clustering

Here we provide the code for clustering the stocks based on the financial report data.

- The code contains two parts:
  - The K-means Clustering + Cosine Distance
  - The Hierarchical Clustering + Self-defined Distance

- The Input
  - The financial report data
- The Output
  - Stock cluster categories
- Usage

```
python cluster.py
```



### Periodicity Analyzer

Based on the observation that the price of some stocks tends to change periodically, we use the stocks price data to build a periodicity analyzer to judge whether a stock has periodicity or not. Periodicity analyzer handle the price data and evaluate whether a stock is periodic or not. It based on two criteria: whether the price of this stock shows a similar trend in two years, and whether the price has a similar trend for most of the period in the past.

- The Input
  - The stock price data
- The Output
  - A set which record the periodic stock ID and the derivative of the price curve.
- Usage

```
python PLRperiodic.py
```



### Short-term Analyzer

We use a simple Long short-term memory based deep learning model as the backbone of short-term analysis. Its input is the closing price of the past twenty trading days, and its output is the predicted closing price of the next trading day. Based on the clustering results obtained from the Long-term Analysis,we train an LSTM model for each cluster. Long-term Analysis ensures that the stocks in each cluster have similar characteristics and potential connections, which can help the model improve its predicting performance.

- The Input
  - The stock price data
  - The clustering result from Long-term Analyzer
- The Output
  - Models for predicting stock prices
- Usage

```
python predictor.py
```



### Strategy Generator

We use the prediction result from models and the periodic information to generate investment strategies. The strategies are based on the Regression idea. We classify the predicted change into three categories: positive, negative, and partial. The positive and negative categories are only based on the predicted change, if the predicted change is significant enough, we will only consider the shift direction of the change and we will buy as many sharesas possible with our current funds. 
If the predicted change is not that significant, i.e. partial, we will adjust how certain we are about the prediction results based on periodic information. In this situation we will buy shares with a certain ratio of the current funds.

- The Input
  - The predicted stock prices from models above
  - The periodicity result from Periodicity Analyzer
- The Output
  - A strategy for investing based on price prediction and the periodicity result
- Usage

```
python strategy.py
```



## Conclusion

We design and implement a stock price forecasting system that utilizes both long-term and short-term data and further generate investment strategies. Experiments on the dataset containing 3192 stock information shows that our pipeline can achieve a ROR of 12.56%. We have also conducted analysis and experiments including clustering algorithms and periodicity search methods. Limited by time and computing resources, we cannot further explore how to embed more complex deep learning techniques into our pipeline to achieve better performance, but our results prove that the combination of long-term data and short-term data to predict stock prices and generate investment strategies is feasible and effective.
