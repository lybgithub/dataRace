#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import csv

path750 = "I:/JData/top750Test.txt"
path = ["I:/JData/TrainData/action_demo.csv","I:/JData/TrainData/action_demo2.csv","I:/JData/TrainData/action_demo3.csv","I:/JData/TrainData/action_demo4.csv"] 
pathP = "I:/JData/TrainData/Pdemo.csv"
pathResult = "I:/JData/TrainData/Train_Test/402result.csv"

df1 = pd.read_csv(path[0])
df2 = pd.read_csv(path[1])
df3 = pd.read_csv(path[2])
df4 = pd.read_csv(path[3])
df = pd.concat([df1,df2,df3,df4])
df = df[['user_id','sku_id','type','cate']]
df = df[(df.type==4)&(df.cate==8)]
df = df[['user_id','sku_id']] 

dfP = pd.read_csv(pathP)
dfP = dfP['sku_id'].to_frame()              # table A:Product

df_onlyP = pd.merge(df,dfP,on="sku_id")                # have error
df_onlyP = df_onlyP[['user_id','sku_id']]   # table B:user_id，sku_id（sku_id belong to P）
f = open(path750,'r')
user_ids = f.readlines()

csvfile = file(pathResult,'wb')
writer = csv.writer(csvfile)
writer.writerow(['user_id','sku_id'])

for i in user_ids:
    i = int(i)                          # TOP750 users
    df_fixUser = df_onlyP[df_onlyP.user_id==i]  # filter B:select * where user_id=TOP750 
    sku_idSort = df_fixUser['sku_id'].value_counts().index.tolist()   # sort the sku_id
    if(len(sku_idSort)==0):                                           # if the user never buy P,pass
        continue
    sku_id = sku_idSort[0]
    writer.writerow([i,sku_id])
csvfile.close()  
f.close()        
