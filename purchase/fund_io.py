import pandas as pd


date_interest=pd.read_csv(r'../analyed_data/date_interest.csv',index_col='report_date',parse_dates=['report_date'])
columns=date_interest.columns
# 因为有10个属性,所以需要每一个分开来进行预测
for i in range(len(columns)):
   print(date_interest.iloc[:,i].describe())
