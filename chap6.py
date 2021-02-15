# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:12:19 2021

@author: leeso
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


os.chdir('C:\\Users\\leeso\\Downloads\\pyda100-master\\6장')

factory=pd.read_csv('tbl_factory.csv')

warehouse=pd.read_csv('tbl_warehouse.csv')

cost = pd.read_csv('rel_cost.csv')

trans = pd.read_csv('tbl_transaction.csv')

join_data = pd.merge(trans,cost, left_on=['ToFC','FromWH'],right_on=['FCID','WHID'],how='left')
join_data_2 = pd.merge(join_data, factory,on='FCID',how='left')
join_data_3 = pd.merge(join_data_2,warehouse,on='WHID',how='left')


join_data_3.columns
merge_data= join_data_3[['TRID', 'TransactionDate', 'Quantity',
       'Cost','ToFC', 'FCName', 'FCDemand', 'FCRegion', 'FromWH','WHName',
       'WHSupply', 'WHRegion', 'RCostID']]


north =merge_data.loc[merge_data['FCRegion']=='북부']
south =merge_data.loc[merge_data['FCRegion']=='남부']


(north['Cost'].sum() / north['Quantity'].sum())*10000

(south['Cost'].sum() / south['Quantity'].sum())*10000


north_wh = merge_data.loc[merge_data['WHRegion']=='북부']
south_wh = merge_data.loc[merge_data['WHRegion']=='남부']

north_wh['Cost'].mean()
south_wh['Cost'].mean()


cost_chk = pd.merge(cost,factory,on='FCID',how='left')
cost_chk.loc[cost_chk['FCRegion']=='북부','Cost'].mean()
cost_chk.loc[cost_chk['FCRegion']=='남부','Cost'].mean()

#

G = nx.Graph()

G.add_node('nodeA')
G.add_node('nodeB')
G.add_node('nodeC')

G.add_edge('nodeA','nodeB')
G.add_edge('nodeA','nodeC')
G.add_edge('nodeB','nodeC')

pos={}
pos['nodeA']=(0,0)
pos['nodeB']=(1,1)
pos['nodeC']=(0,1)

nx.draw(G,pos,with_labels=True)

plt.show()

