# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:49:14 2021

@author: leeso
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\3장')

use_log=pd.read_csv('use_log.csv')
customer=pd.read_csv('customer.csv')

use_log.isnull().sum()
customer.isnull().sum()

customer_clustering = customer[['mean', 'median', 'max', 'min', 'customer_period']]

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#표준화
sc= StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)

kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(customer_clustering_sc)
customer_clustering['cluster'] = clusters.labels_

customer_clustering['cluster'].unique() # 3 1 0 2


customer_clustering.groupby('cluster').count()

customer_clustering.groupby('cluster').mean()

from sklearn.decomposition import PCA

X = customer_clustering_sc
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df['cluster'] = customer_clustering['cluster']

for i in pca_df["cluster"].unique():
    clus=pca_df.loc[pca_df['cluster']==i]
    plt.scatter(clus[0],clus[1])

customer_deleted=pd.concat([customer_clustering,customer['is_deleted']], axis=1)

customer_deleted = customer_deleted.groupby(['cluster','is_deleted'],as_index=False).count()[['cluster','is_deleted','customer_period']]
customer_deleted.rename(columns={'customer_period':'count'},inplace=True)



customer_routine=pd.concat([customer_clustering,customer['routine']], axis=1)

customer_routine = customer_routine.groupby(['cluster','routine'],as_index=False).count()[['cluster','routine','customer_period']]
customer_routine.rename(columns={'customer_period':'count'},inplace=True)


use_log.dtypes

use_log['usedate']=pd.to_datetime(use_log['usedate'])
use_log['useYm']=use_log['usedate'].dt.strftime('%Y%m')
customer_ym_use=use_log.groupby(['useYm','customer_id'],as_index=False).count()
del customer_ym_use['log_id']
customer_ym_use.rename(columns={'usedate':'count'},inplace=True)

ym = list(customer_ym_use['useYm'].unique())
predict_data = pd.DataFrame()
for i in range(6,len(ym)) :
    tmp=customer_ym_use.loc[customer_ym_use['useYm']==ym[i]]
    tmp.rename(columns={'count':'pred_count'},inplace=True)
    for j in range(1,7) :
        tmp_before = customer_ym_use.loc[customer_ym_use['useYm']==ym[i-j]]
        del tmp_before['useYm']
        tmp_before.rename(columns={'count':'count_{}'.format(j-1)},inplace=True)
        tmp = pd.merge(tmp,tmp_before,on='customer_id',how='left')
    predict_data = pd.concat([predict_data,tmp],ignore_index=True)
    

predict_data.dropna(inplace=True)
predict_data= predict_data.reset_index(drop=True)
    



predict_data2= pd.merge(predict_data, customer[['customer_id','start_date']],on='customer_id',how='left')
predict_data2.isnull().sum()

from dateutil.relativedelta import relativedelta
predict_data2.dtypes
predict_data2['start_date'] = pd.to_datetime(predict_data2['start_date'])
predict_data2['useYm'] = pd.to_datetime(predict_data2['useYm'],format='%Y%m')
predict_data2['customer_period']=0

for cus in range(len(predict_data2)) :
    delta_days = relativedelta(predict_data2['useYm'].iloc[cus], predict_data2['start_date'].iloc[cus])
    predict_data2['customer_period'].iloc[cus] = delta_days.years*12 + delta_days.months #월단위계산
    
from sklearn import linear_model
import sklearn.model_selection

model = linear_model.LinearRegression()
predict_data2=predict_data2.loc[predict_data2['start_date']>=pd.to_datetime('20180401')]
X=predict_data2[['count_0','count_1','count_2','count_3','count_4','count_5','customer_period']]
y=predict_data2['pred_count']
X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y)
model.fit(X,y)


model.score(X_train,y_train)
model.score(X_test,y_test)


pd.DataFrame({'Feature_name':X.columns,'coef':model.coef_})


customer_ym_use.to_csv('use_log_months.csv',index=False)