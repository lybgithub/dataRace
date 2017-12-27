import os
import pandas as pd
flag = True
from genFeatures import get_distance

# 用户在某一起点到达过的终点
def get_user_sloc_eloc(cache_path,train,test):
    result_path = cache_path + 'user_start_end_loc_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid', 'geohashed_start_loc','geohashed_end_loc']].drop_duplicates()
        result = pd.merge(test[['orderid', 'userid','geohashed_start_loc']], user_eloc, on=['userid','geohashed_start_loc'], how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result

# 用户在某一终点到达过的起点,每一条记录都计算一次折返
def get_user_eloc_sloc(cache_path,train,test):
    result_path = cache_path + 'user_end_start_loc_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid', 'geohashed_start_loc','geohashed_end_loc']].drop_duplicates()
        user_eloc = user_eloc.rename(columns={'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'})
        result = pd.merge(test[['orderid', 'userid','geohashed_start_loc']], user_eloc, on=['userid','geohashed_start_loc'], how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(cache_path,train,test):
    result_path = cache_path + 'user_end_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_eloc = train[['userid','geohashed_end_loc']].drop_duplicates()
        result = pd.merge(test[['orderid','userid']],user_eloc,on='userid',how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result
# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(cache_path,train,test):
    result_path = cache_path + 'user_start_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_sloc = train[['userid', 'geohashed_start_loc']].drop_duplicates()
        result = pd.merge(test[['orderid', 'userid']], user_sloc, on='userid', how='left')
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result
# 筛选起始地点去向最多的3个地点(在这个地方加十字卷积，把周边样本也加入，然后用距离筛)
def get_30percent_group(group):
    num = int(group.shape[0] * 0.3)
    if(num<3):
        group = group.tail(3)
    else:
        group = group.tail(num)
    return group

def get_loc_to_loc(cache_path,train,test):
    result_path = cache_path + 'loc_to_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'sloc_eloc_count':'count'})
        sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
        # 过滤距离
        # sloc_eloc_count = get_distance(sloc_eloc_count)
        # sloc_eloc_count = sloc_eloc_count[sloc_eloc_count['distance']<=3000]
        # sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(6)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').apply(get_30percent_group)
        result = pd.merge(test[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='left')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result

# 寻找leak的途径，本次终点为下一次起点
def getNewEndLoc(group):
    loc = list(group['geohashed_start_loc'])
    loc.append(0)
    del loc[0]
    group['new_end_loc'] = loc
    return group

# bikeleak产生的样本加入候选集
def leak_bikeid(cache_path,train,test):
    result_path = cache_path + 'leak_bikeid_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        df = pd.concat([train, test],ignore_index=True)
        df['new_end_loc'] = 0
        leakBike = df.sort_values(['bikeid','orderid']).groupby(['bikeid'], as_index=False).apply(getNewEndLoc)  # 获取leak,没有考虑时间的限制
        result = pd.merge(test[['orderid','bikeid']],leakBike[['orderid','new_end_loc']],on=['orderid'],how='left')
        result.rename(columns={'new_end_loc':'geohashed_end_loc'},inplace=True)
        result = result[result.geohashed_end_loc != 0].reset_index()[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result

# userleak产生的样本加入候选集
def leak_userid(cache_path,train,test):
    result_path = cache_path + 'leak_userid_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train.loc[:, 'starttime'] = pd.to_datetime(train.starttime)  # 转化为时间类型
        test.loc[:, 'starttime'] = pd.to_datetime(test.starttime)  # 转化为时间类型
        df = pd.concat([train, test], ignore_index=True)
        df['day'] = df.starttime.dt.day  # 获取日信息
        df['new_end_loc'] = 0
        df = df.sort_values(['userid', 'day', 'orderid'])
        leakUser = df.groupby(['userid', 'day'], as_index=False).apply(getNewEndLoc)  # 获取leak
        result = pd.merge(test[['orderid','userid']],leakUser[['orderid','new_end_loc']],on=['orderid'],how='left')
        result.rename(columns={'new_end_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[result.geohashed_end_loc != 0].reset_index()[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result


