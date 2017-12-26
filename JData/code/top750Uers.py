import pandas as pd
path = ["","","",""]   # 四个action.csv路径
df1 = pd.read_csv(path[0])
df2 = pd.read_csv(path[1])
df3 = pd.read_csv(path[2])
df4 = pd.read_csv(path[3])
dfP = pd.read_csv("")  # Product.csv路径
df = pd.concat([df1,df2,df3,df4])

dfP = dfP['sku_id'].to_frame()

df = df[['user_id','sku_id','type','cate']]
df2 = df[(df.type==4)&(df.cate==8)] 
df3 = df2[['user_id','sku_id']]    # 下单品类8行为对应的用户商品对
df_onlyP = pd.merge(df3,dfP)       # 再次过滤，只取p子集中的商品,没有指明column

# 先针对用户下单量对用户进行排序，取top750
user_count = df_onlyP['user_id'].value_counts() # 默认就是降序
userSortList = user_count.index.tolist()  # 使用index获得丢失的key信息
top750User = userSortList[:750]

# 保存至txt
s = '\n'.join([str(x) for x in top750User])
fin = open(r'','w')   # 文件结果的输出路径
fin.write(s)
fin.close()