# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:07:08 2021

@author: leeso
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from itertools import product
from pulp import LpVariable, lpSum, value 
from ortoolpy import model_min, addvars, addvals

os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\7장')


df_tc = pd.read_csv('trans_cost.csv', index_col="공장")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')
df_pos = pd.read_csv('trans_route_pos.csv')



np.random.seed(1)
nw = len(df_tc.index) # W1, W2, W3
nf = len(df_tc.columns) # F1, F2, F3, F4
pr = list(product(range(nw), range(nf))) #12개


m1 = model_min() # 최소화 실행할 모델 정의

v1 = {(i,j):LpVariable('v%d_%d'%(i,j),lowBound=0) for i,j in pr} # 주요 변수

m1 += lpSum(df_tc.iloc[i][j]*v1[i,j] for i,j in pr) # 각 요소의 곱의 합으로 함수 정의


# 제약조건

for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i] # 공급량 이내로 생산 가능
    
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j] # 수요량 이상으로 있어야
    
m1.solve()



# 총 운송 비용 계산

df_tr_sol = df_tc.copy()

total_cost = 0

for k,x in v1.items():
    i,j = k[0],k[1] # (2,3)
    df_tr_sol.iloc[i][j] = value(x) # 20
    total_cost += df_tc.iloc[i][j]*value(x)
    
print(df_tr_sol)

print("총 운송 비용:"+str(total_cost))



##

# 데이터 불러오기
df_tr = df_tr_sol.copy() # 최적화 결과

# 객체 생성
G = nx.Graph()

# 노드 설정
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 엣지 설정 & 엣지의 가중치 리스트화
num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i==j):
            # 엣지 추가
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # 엣지 가중치 추가
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)
                

# 좌표 설정
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])
    
# 그리기
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 표시
plt.show()



# 수요측
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 공급측
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("수요 조건 계산 결과:"+str(condition_demand(df_tr_sol,df_demand)))
print("공급 조건 계산 결과:"+str(condition_supply(df_tr_sol,df_supply)))
