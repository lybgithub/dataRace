
# coding: utf-8

# In[34]:

f = open("./format/timeformat/4_action.csv")
context = f.readlines()

import numpy as np

train_day1to5 = []
offline_candidate_day6to10 = []
online_candidate_day11to15 = []

for line in context:
    line = line.replace('\n','')
    array = line.split(',')
#     if array[0] == 'user_id':
#         continue
    day = int(array[3])
    uid = (array[0],array[1],day+5,array[7])
    if day <= 5:
        train_day1to5.append(uid)
    elif day <=10:
        offline_candidate_day6to10.append(uid)
    else:
        online_candidate_day11to15.append(uid)


# In[35]:

train_day1to5 = list(set(train_day1to5))
offline_candidate_day6to10 = list(set(offline_candidate_day6to10))
online_candidate_day11to15 = list(set(online_candidate_day11to15))

print('training item number:\t',len(train_day1to5))
print('---------------------\n')
print('offline candidate item number:\t',len(offline_candidate_day6to10))
print('---------------------\n')


# In[40]:

import math
# for feature
ui_dict = [{} for i in range(6)]
for line in context:
    line = line.replace('\n','')
    array = line.split(',')
#     if array[0] == 'user_id':
#         continue
    day = int(array[3])
    uid = (array[0],array[1],day,array[7])
    type0 = int(array[6])-1
    if uid in ui_dict[type0]:
        ui_dict[type0][uid] += 1
    else:
        ui_dict[type0][uid] = 1
        
# for label
ui_buy = {}
for line in context:
    line = line.replace('\n','')
    array = line.split(',')
#     if array[0] == 'user_id':
#         continue

    uid = (array[0],array[1],int(array[3]),array[7])
    if array[6] == '4':
        ui_buy[uid] = 1


# In[49]:

# get train X,y
X = np.zeros((len(offline_candidate_day6to10),6))
y = np.zeros(len(offline_candidate_day6to10),)
id = 0
for uid in offline_candidate_day6to10:
    last_uid = (uid[0],uid[1],uid[2]-5,uid[3])
    for i in range(6):
        X[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)
    y[id] = 1 if uid in ui_buy else 0
    id += 1

print('X= ',X,'\n\n','y = ',y)
print('-------\n\n')
print('train_number = ',len(y),'positive_number=',sum(y), '\n')



# In[41]:

# get predict pX for offline_candidate_day11-15
pX = np.zeros((len(offline_candidate_day6to10),6))
id = 0
for uid in offline_candidate_day6to10:
    last_uid = (uid[0],uid[1],uid[2]-5,uid[3])
    for i in range(6):
        pX[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)
    id += 1


# In[43]:

#get predict tX for online_candidate_day16-20
tX = np.zeros((len(online_candidate_day11to15),6))
id = 0
for uid in online_candidate_day11to15:
    last_uid = (uid[0],uid[1],uid[2]-5,uid[3])
    for i in range(6):
        pX[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)
    id += 1


# In[50]:

#train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,y)


# In[67]:

#product_map
m = open("./JData_Product.csv")
context = m.readlines()

product_map = {}
for line in context:
    line = line.replace('\n','')
    array = line.split(',')
    if array[0] == 'user_id':
        continue
    sku_id = array[0]
    if sku_id not in product_map:
        product_map[sku_id] = 1
print(len(product_map))


# In[66]:

#usr_map
m = open("./JData_UserHadle.csv")
context = m.readlines()

usr_map = {}
for line in context:
    line = line.replace('\n','')
    array = line.split(',')
    if array[0] == 'user_id':
        continue
    user_id = array[0]
    if user_id not in product_map:
        usr_map[user_id] = 1
print(len(usr_map))


# In[56]:

m = open("./format/timeformat/4_action.csv")
context = m.readlines()


product_map_4 = []
for line in context:
    line = line.replace('\n','')
    array = line.split(',')
#     if array[0] == 'user_id':
#         continue
    sku_id = array[1]
    cate = array[7]
    if cate == '8':
        product_map_4.append(sku_id)
product_map_4 = list(set(product_map_4))

m = len(product_map_4)
print(m)


# In[71]:

# evaluate 
py = model.predict_proba(tX)

npy = []
for a in py:
    npy.append(a[1])
py = npy

print('pX=')
print(pX)

## combine 
lx = zip(online_candidate_day11to15,py)
print('-----------------')
lx = sorted(lx,key=lambda x:x[1], reverse=True)
print('-----------------')

wf = open('qdqwdqwdqw','w')
wf.write('user_id,sku_id\n')
id = 0

dicc = {}
for i in range(len(lx)):
    if id == 1200:
        break
    item = lx[i]
    if item[0][3] == '8' and item[0][1] in product_map and item[0][0] not in dicc :
        id += 1
        dicc[item[0][0]]= 1
        wf.write('%s,%s\n'%(item[0][0],item[0][1]))
