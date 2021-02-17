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

