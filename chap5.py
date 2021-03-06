# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:57:44 2021

@author: leeso
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta


os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\3장')
customer = pd.read_csv('customer.csv')
use_log_months = pd.read_csv('use_log_months.csv')

#1개월 전의 회원 이력 집계

yearmonth = use_log_months['useYm'].unique()
uselog = pd.DataFrame()

for i in range(1,len(yearmonth)) :
    tmp = use_log_months.loc[use_log_months['useYm']==yearmonth[i]] # 해당월의 사람들 다 산출
    tmp = tmp.rename(columns={'count':'count_0'})
    tmp_before = use_log_months.loc[use_log_months['useYm']==yearmonth[i-1]]
    del tmp_before['useYm']
    tmp_before= tmp_before.rename(columns={'count':'count_1'})
    tmp=pd.merge(tmp,tmp_before,on='customer_id',how='left')
    uselog=pd.concat([uselog,tmp],ignore_index=True)
    
    
    
#탈퇴한사람목록
    
from dateutil.relativedelta import relativedelta
    
delete_customer=customer.loc[customer['is_deleted']==1]
delete_customer = delete_customer.reset_index(drop=True)
delete_customer['exit_date'] = None
delete_customer['end_date'] = pd.to_datetime(delete_customer['end_date'])

for i in range(len(delete_customer)) :
    delete_customer['exit_date'][i] = delete_customer['end_date'].iloc[i] - relativedelta(months=1)

delete_customer['useYm'] = delete_customer['exit_date'].dt.strftime('%Y%m')

uselog.dtypes # useYm : str

delete_customer['useYm']= delete_customer['useYm'].astype(str)
uselog['useYm']= uselog['useYm'].astype(str)

delete_uselog = pd.merge(uselog, delete_customer,on=['customer_id','useYm'],how='left')

delete_uselog = delete_uselog.dropna(subset=['name']) #count가 지워지지않게
len(delete_uselog['customer_id'].unique())




    
conti_customer=customer.loc[customer['is_deleted']==0]
conti_customer = conti_customer.reset_index(drop=True)

conti_uselog = pd.merge(uselog, conti_customer,on='customer_id',how='left')

conti_uselog = conti_uselog.dropna(subset=['name']) #count가 지워지지않게
len(conti_uselog['customer_id'].unique())

conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)
conti_uselog = conti_uselog.drop_duplicates(subset='customer_id')




predict_data=pd.concat([conti_uselog,delete_uselog],ignore_index=True)


# useYm까지 재적기간

predict_data['useYm'] = pd.to_datetime(predict_data['useYm'],format='%Y%m')
predict_data['start_date'] = pd.to_datetime(predict_data['start_date'])

predict_data['period'] = None

for i in range(len(predict_data)) : 
    delta=relativedelta(predict_data['useYm'][i],predict_data['start_date'][i])
    predict_data['period'][i] = int(delta.years*12 +delta.months)



predict_data['period']=predict_data['period'].astype(int)


predict_data=predict_data.dropna(subset=['count_1'])


predict_data.dtypes
target_col = ['campaign_name','class_name','gender','count_1','routine','period','is_deleted']
predict_data = predict_data[target_col]
predict_data

predict_dummy = pd.get_dummies(predict_data)

del predict_dummy['campaign_name_2_일반']
del predict_dummy['class_name_2_야간']
del predict_dummy['gender_M']


from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection

deleted = predict_dummy.loc[predict_dummy['is_deleted']==1]
conti = predict_dummy.loc[predict_dummy['is_deleted']==0].sample(len(deleted)) # 비율 1:1로 맞춤

X = pd.concat([deleted,conti],ignore_index = True)
y = X['is_deleted']
del X['is_deleted']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
y_test_pred = model.predict(X_test)



model.score(X_train, y_train) # 0.98
model.score(X_test,y_test) # 0.89


model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train,y_train)

model.score(X_train, y_train) # 0.93
model.score(X_test,y_test) # 0.91


importance = pd.DataFrame({'feature_name':X.columns,'coefficient':model.feature_importances_})




