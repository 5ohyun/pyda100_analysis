# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:12:02 2021

@author: leeso
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\2장')

uriage=pd.read_csv('uriage.csv')
kokyaku=pd.read_excel('kokyaku_daicho.xlsx')


len(pd.unique(uriage['item_name'])) # 99

uriage['item_name'] = uriage['item_name'].str.upper()
uriage['item_name'] = uriage['item_name'].str.replace(' ','')

len(pd.unique(uriage['item_name'])) # 26

uriage.isnull().sum()

check_null=uriage['item_price'].isnull()


for check in list(uriage.loc[check_null, "item_name"].unique()) :  #결측치가 있는 item_name
    price = uriage.loc[(~check_null) & (uriage['item_name']==check),'item_price'].max()
    uriage.loc[(check_null) & (uriage['item_name']==check),'item_price'] = price
    

for check in uriage['item_name'].sort_values().unique():
    print(check + '의 max값 : ' + str(uriage.loc[uriage['item_name']==check]['item_price'].max(skipna=False)) +
          ' | min값 : '+ str(uriage.loc[uriage['item_name']==check]['item_price'].min(skipna=False)))
    
for check in uriage['item_name'].sort_values().unique():
    print(uriage.loc[uriage['item_name']==check]['item_price'].max(skipna=False) == uriage.loc[uriage['item_name']==check]['item_price'].min(skipna=False))

kokyaku['고객이름'] = kokyaku['고객이름'].str.replace(' ','')
    
date_num=kokyaku['등록일'].astype('str').str.isdigit() #숫자로 읽히는 데이터 확인
date_num.sum() #22

correct_date = pd.to_timedelta(kokyaku.loc[date_num,'등록일'].astype("float"), unit="D") + pd.to_datetime("1900/01/01")
correct_date2 = pd.to_datetime(kokyaku.loc[~date_num,'등록일']) #슬래시 구분 서식을 -로 변경

kokyaku['등록일']=pd.concat([correct_date,correct_date2])
kokyaku.dtypes

kokyaku['등록연월'] = kokyaku['등록일'].dt.strftime('%Y%m')
kokyaku.groupby('등록연월').count()['고객이름']

uriage['purchase_date']=pd.to_datetime(uriage['purchase_date'])
uriage['purchase_month'] = uriage['purchase_date'].dt.strftime('%Y%m')

merge_data = pd.merge(kokyaku, uriage,left_on='고객이름', right_on='customer_name',how='right')
merge_data = merge_data.drop('customer_name',axis=1)
dump_data = merge_data[['purchase_date','purchase_month','item_name','item_price','고객이름','지역','등록일']] #분석 쉽게 데이터 순서 변경

dump_data.to_csv('renew_data.csv',index=False)


import_data=pd.read_csv('renew_data.csv')
import_data.pivot_table(index='purchase_month',columns='item_name',aggfunc='count',fill_value=0)

import_data.pivot_table(index='purchase_month',columns='item_name',values='item_price',aggfunc='sum',fill_value=0)

import_data.pivot_table(index='purchase_month',columns='고객이름',aggfunc='size',fill_value=0)

import_data.pivot_table(index='purchase_month',columns='지역',aggfunc='size',fill_value=0)

non_customer = pd.merge(kokyaku, uriage,left_on='고객이름', right_on='customer_name',how='left')
non_customer.loc[non_customer['purchase_date'].isnull()]


