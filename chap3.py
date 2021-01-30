# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:24:25 2021

@author: leeso
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\3장')

use_log=pd.read_csv('use_log.csv')
customer_master=pd.read_csv('customer_master.csv')
class_master=pd.read_csv('class_master.csv')
campaign_master = pd.read_csv('campaign_master.csv')

customer_join = pd.merge(customer_master, class_master, on='class', how='left')
customer_join = pd.merge(customer_join, campaign_master, on='campaign_id', how='left')
customer_join.isnull().sum()

# 어떤 회원이 많은지

customer_join.pivot_table(index='campaign_name',columns='class_name',aggfunc='size')

customer_join.groupby('class_name').count()['customer_id']
customer_join.groupby('gender').count()['customer_id']
customer_join.groupby('campaign_name').count()['customer_id']
customer_join.groupby('is_deleted').count()['customer_id']

customer_join.dtypes

customer_join['start_date'] = pd.to_datetime(customer_join['start_date'])
customer_start_1804 = customer_join.loc[customer_join['start_date']>pd.to_datetime('20180401')]
customer_start_1804 #1361명


## 최신고객 집계

customer_join['end_date']=pd.to_datetime(customer_join['end_date'])
customer_new1=customer_join.loc[customer_join['is_deleted']==0] #3월에 탈퇴한 사람들은 집계되자 않음

#3월에 있었던 사람들 집계
customer_new2=customer_join.loc[(customer_join['end_date']>=pd.to_datetime('20190331')) | (customer_join['end_date'].isna())]
customer_new2['end_date'].unique()


use_log['usedate']=pd.to_datetime(use_log['usedate'])
use_log['Ym']=use_log['usedate'].dt.strftime('%Y%m')

ym_customer_count=use_log.groupby(['Ym','customer_id'],as_index=False).count().drop('usedate',axis=1)
ym_customer_count.rename(columns={'log_id':'count'},inplace=True)

ym_customer_agg=ym_customer_count.groupby('customer_id').agg(['min','max','mean','median'])['count']
ym_customer_agg = ym_customer_agg.reset_index(drop=False)

ym_customer_agg.max()

use_log['weekday'] = use_log['usedate'].dt.weekday #요일별 집계
use_log_weekday = use_log.groupby(['customer_id','Ym','weekday'],as_index=False).count()
del use_log_weekday['log_id']
use_log_weekday.rename(columns={'usedate':'count'},inplace=True)

use_log_weekday=use_log_weekday.groupby('customer_id',as_index=False).max()[['customer_id','count']]
use_log_weekday['routine']=0
use_log_weekday['routine'] = use_log_weekday['routine'].where(use_log_weekday['count']<4,1)

customer_join2 = pd.merge(customer_join,ym_customer_agg,on='customer_id',how='left')
customer_join3 = pd.merge(customer_join2, use_log_weekday[['customer_id','routine']], on='customer_id', how='left')
customer_join3.isnull().sum()

customer=customer_join3

customer['calc_date']=customer['end_date']
customer['calc_date']= customer['calc_date'].fillna(pd.to_datetime('2019/04/30'))

#날짜 빼기 함수

from dateutil.relativedelta import relativedelta

customer['customer_period']=0

for cus in range(len(customer)) :
    delta_days = relativedelta(customer['calc_date'].iloc[cus], customer['start_date'].iloc[cus])
    customer['customer_period'].iloc[cus] = delta_days.years*12 + delta_days.months #월단위계산
    


stat_table=customer.describe()[['min','max','mean','median']]
customer.groupby('routine').count()['customer_id']
plt.hist(customer['customer_period'])
    

customer_end = customer.loc[customer['is_deleted']==1]
customer_stay = customer.loc[customer['is_deleted']==0]


table_end=customer_end.describe()
table_stay=customer_stay.describe()











