# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:41:14 2021

@author: leeso
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import networkx as nx
#from itertools import product
from pulp import LpVariable, lpSum, value 
from ortoolpy import model_max, addvars, addvals

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\7장')


df_material = pd.read_csv('product_plan_material.csv', index_col="제품")
df_profit = pd.read_csv('product_plan_profit.csv', index_col="제품")
df_stock = pd.read_csv('product_plan_stock.csv', index_col="항목")
df_plan = pd.read_csv('product_plan.csv', index_col="제품")


# 이익 계산 함수
def product_plan(df_profit,df_plan):
    profit = 0
    for i in range(len(df_profit.index)):
        for j in range(len(df_plan.columns)):
            profit += df_profit.iloc[i][j]*df_plan.iloc[i][j]
    return profit

product_plan(df_profit,df_plan)



df = df_material.copy() #
inv = df_stock

m = model_max()

v1 = {(i):LpVariable('v%d'%(i),lowBound=0) for i in range(len(df_profit))}
# 제품 수와 같은 차원으로 정의

m += lpSum(df_profit.iloc[i]*v1[i] for i in range(len(df_profit)))
# v1과 제품별 이익의 곱의 합으로 목적함수 정의

for i in range(len(df_material.columns)):
    m += lpSum(df_material.iloc[j,i]*v1[j] for j in range(len(df_profit)) ) <= df_stock.iloc[:,i]
# 제약조건 정의 - 재고를 넘지않게
    
    
m.solve() # 최적화 문제


df_plan_sol = df_plan.copy()

for k,x in v1.items():
    df_plan_sol.iloc[k] = value(x) # 최적화한 값
    
df_plan_sol  
  
value(m.objective) # 총 이익

# 제약 조건 계산 함수

def condition_stock(df_plan,df_material,df_stock):
    flag = np.zeros(len(df_material.columns))
    for i in range(len(df_material.columns)):  # 원료
        temp_sum = 0
        for j in range(len(df_material.index)):  # 제품
            temp_sum = temp_sum + df_material.iloc[j][i]*float(df_plan.iloc[j]) # 필요한 원료 개수
        if (temp_sum<=float(df_stock.iloc[0][i])):
            flag[i] = 1
            
        print(df_material.columns[i]+"  사용량:"+str(temp_sum)+", 재고:"+str(float(df_stock.iloc[0][i])))
    return flag

print("제약 조건 계산 결과:"+str(condition_stock(df_plan_sol,df_material,df_stock)))




제품 = list('AB')
대리점 = list('PQ')
공장 = list('XY')
레인 = (2,2)

# 운송비 #
tbdi = pd.DataFrame(((j,k) for j in 대리점 for k in 공장), columns=['대리점','공장'])
tbdi['운송비'] = [1,2,3,1]
print(tbdi)

# 수요 #
tbde = pd.DataFrame(((j,i) for j in 대리점 for i in 제품), columns=['대리점','제품'])
tbde['수요'] = [10,10,20,20]
print(tbde)

# 생산 #
tbfa = pd.DataFrame(((k,l,i,0,np.inf) for k,nl in zip (공장,레인) for l in range(nl) for i in 제품), 
                    columns=['공장','레인','제품','하한','상한'])
tbfa['생산비'] = [1,np.nan,np.nan,1,3,np.nan,5,3]
tbfa.dropna(inplace=True)
tbfa.loc[4,'상한']=10
print(tbfa)

from ortoolpy import logistics_network
_, tbdi2, _ = logistics_network(tbde, tbdi, tbfa,dep = "대리점", dem = "수요",fac = "공장",
                                prd = "제품",tcs = "운송비",pcs = "생산비",lwb = "하한",upb = "상한")

print(tbfa)
print(tbdi2)

