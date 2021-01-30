# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:29:40 2021

@author: leeso
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\1장')
customer_master = pd.read_csv('customer_master.csv') #[5000,8]
item_master = pd.read_csv('item_master.csv') #[5,3]
transaction_1 = pd.read_csv('transaction_1.csv') #[5000,4]
transaction_2 = pd.read_csv('transaction_2.csv') #[1786,4]
transaction_detail_1 = pd.read_csv('transaction_detail_1.csv') #[5000,4]
transaction_detail_2 = pd.read_csv('transaction_detail_2.csv') #[2144,4]

transaction=pd.concat([transaction_1,transaction_2],ignore_index=True) #[6786,4]
transaction_detail=pd.concat([transaction_detail_1,transaction_detail_2],ignore_index=True) #[7144,4]


transaction_detail.duplicated('transaction_id').sum()
transaction.duplicated('transaction_id').sum()

merge_data_1=pd.merge(transaction_detail,transaction,on='transaction_id',how='left') #[7144,7]
merge_data_2=pd.merge(merge_data_1,item_master,on='item_id',how='left') #[7144,9]
merge_data_3=pd.merge(merge_data_2,customer_master,on='customer_id',how='left') #[7144,16]

merge_data_3['detail_price']= merge_data_3['quantity']*merge_data_3['item_price']
transaction['price'].sum()
merge_data_3['detail_price'].sum() # 검산과정 = TRUE
merge_data_3.drop_duplicates('transaction_id')['price'].sum()


# 데이터 분석

merge_data_3.isnull().sum()
merge_data_3[['quantity','price','item_price','age','detail_price']].describe()


merge_data_3.dtypes
merge_data_3['payment_date'] = pd.to_datetime(merge_data_3['payment_date'])
merge_data_3['payment_date'].min()
merge_data_3['payment_date'].max()

merge_data_3['payment_month']=merge_data_3['payment_date'].dt.strftime('%Y%m') #대문자 %Y는 2019 / %y는 19


merge_data_3.groupby("payment_month").sum()['detail_price']


merge_data_3.groupby(['item_name','payment_month']).sum()[['quantity','detail_price']]
pivot_graph = pd.pivot_table(merge_data_3,index='payment_month',columns='item_name',values='detail_price',aggfunc='sum')

plt.plot(list(pivot_graph.index), pivot_graph['PC-A'],label='PC-A')
plt.plot(list(pivot_graph.index), pivot_graph['PC-B'],label='PC-B')
plt.plot(list(pivot_graph.index), pivot_graph['PC-C'],label='PC-C')
plt.plot(list(pivot_graph.index), pivot_graph['PC-D'],label='PC-D')
plt.plot(list(pivot_graph.index), pivot_graph['PC-E'],label='PC-E')
plt.legend()





