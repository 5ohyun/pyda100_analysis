# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 18:38:59 2021

@author: leeso
"""
import os
import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt


os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\6장')

#
df_w = pd.read_csv('network_weight.csv')
df_p = pd.read_csv('network_pos.csv')

# 엣지 가중치 리스트화

size = 10
edge_weights = []
for i in range(len(df_w)):
    for j in range(len(df_w.columns)):
        edge_weights.append(df_w.iloc[i][j]*size)

# 그래프 객체 생성
        
G = nx.Graph()

# 노드 설정

for i in range(len(df_w.columns)):
    G.add_node(df_w.columns[i])

# 엣지 설정
    
for i in range(len(df_w.columns)):
    for j in range(len(df_w.columns)):
        G.add_edge(df_w.columns[i],df_w.columns[j])
    
# 좌표 설정

pos ={}

for i in range(len(df_w.columns)):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])
    

    
nx.draw(G,pos,with_labels=True,font_size=16, node_size=1000, node_color='k', font_color='w',width=edge_weights)
plt.show()


#####

df_tr = pd.read_csv('trans_route.csv',index_col='공장')
df_pos = pd.read_csv('trans_route_pos.csv')
df_tc = pd.read_csv('trans_cost.csv',index_col='공장')
df_tr = pd.read_csv('trans_route.csv', index_col="공장")
df_demand = pd.read_csv('demand.csv')
df_supply = pd.read_csv('supply.csv')
df_tr_new = pd.read_csv('trans_route_new.csv',index_col='공장')


G = nx.Graph()


# 노드 추가 7개

for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])
    
    
# 엣지 설정, 가중치 리스트화
    
    
num_pre = 0
edge_weights =[]
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
                
 

# 운송 비용 함수 - 
def trans_cost(df_tr,df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
        for j in range(len(df_tr.columns)):
            cost += df_tr.iloc[i][j]*df_tc.iloc[i][j]
    return cost

trans_cost(df_tr,df_tc)


# 수요측 제약조건
for i in range(len(df_demand.columns)):
    temp_sum = sum(df_tr[df_demand.columns[i]])
    print(str(df_demand.columns[i])+"실제 운송량:"+str(temp_sum)+" (수요량:"+str(df_demand.iloc[0][i])+")")
    
    if temp_sum>=df_demand.iloc[0][i]:
        print("수요량을 만족시키고있음")
    else:
        print("수요량을 만족시키지 못하고 있음. 운송경로 재계산 필요")

# 공급측 제약조건
for i in range(len(df_supply.columns)):
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])
    print(str(df_supply.columns[i])+"실제 운송량:"+str(temp_sum)+" (공급 범위:"+str(df_supply.iloc[0][i])+")")
    if temp_sum<=df_supply.iloc[0][i]:
        print("공급 범위내")
    else:
        print("공급 범위 초과. 운송경로 재계산 필요")
        
        
##

# 총 운송비용 재계산 
trans_cost(df_tr_new,df_tc)

# 제약조건 계산함수


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

print("수요조건 계산결과:"+str(condition_demand(df_tr_new,df_demand)))
print("공급조건 계산결과:"+str(condition_supply(df_tr_new,df_supply)))
        

