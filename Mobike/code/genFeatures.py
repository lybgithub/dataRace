import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
import geohash
import math
# from sklearn.cluster import MiniBatchKMeans

# 获得时间的time特征
def get_time(s):
    s['month'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[1]).astype(int)
    s['day'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[-1]).astype(int)
    s['year'] = s['starttime'].apply(lambda x:x.split()[0].split('-')[0]).astype(int)
    s['hour'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[0]).astype(int)
    s['min'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[1]).astype(int)
    s['sec'] = s['starttime'].apply(lambda x:x.split()[1].split(':')[-1].split('.')[0]).astype(int)
    s['time'] = ((((s.month-5)*30+s.day-9)*24+s.hour)*60+s['min'])*60 + s['sec']
    return s.drop('month,day,year,hour,min,sec'.split(','),axis=1)

# 统计用户在历史记录,某个时间段,时间中出现的次数
def user_heat(train,result):
    train['hour'] = pd.to_datetime(train['starttime']).map(lambda x: x.strftime('%H'))
    train['hour_duan'] = train['hour'].map(hour_duan)
    result['hour'] = pd.to_datetime(result['starttime']).map(lambda x: x.strftime('%H'))
    result['hour_duan'] = result['hour'].map(hour_duan)
    user_heat = train.groupby(['userid'], as_index=False)['biketype'].agg({'user_heat': 'count'})
    user_hour_duan_heat = train.groupby(['userid','hour_duan'], as_index=False)['biketype'].agg({'user_hour_duan_heat': 'count'})
    user_hour_heat = train.groupby(['userid', 'hour'], as_index=False)['biketype'].agg({'user_hour_heat': 'count'})
    result = pd.merge(result, user_heat, on=['userid'], how='left')
    result = pd.merge(result, user_hour_duan_heat, on=['userid','hour_duan'], how='left')
    result = pd.merge(result, user_hour_heat, on=['userid','hour'], how='left')

    return result

# 统计用户从某个起点到达不同终点的个数，在某个时间段到达不同终点的个数
def user_sloc_eloc_unique(train,result):
    UniqueLocNum = train.groupby(['userid','geohashed_start_loc'],as_index=False)['geohashed_end_loc'].agg({'unique_eloc':lambda x: x.nunique()})   # 统计一个start_loc有几个不同的end_loc
    UniqueLocNum = UniqueLocNum.reset_index()
    UniqueLocNum2 = train.groupby(['userid', 'hour_duan','geohashed_start_loc'], as_index=False)['geohashed_end_loc'].agg({'unique_eloc_hour_duan': lambda x: x.nunique()})  # 统计一个start_loc有几个不同的end_loc
    UniqueLocNum2 = UniqueLocNum2.reset_index()
    UniqueLocNum3 = train.groupby(['userid', 'hour', 'geohashed_start_loc'], as_index=False)['geohashed_end_loc'].agg({'unique_eloc_hour': lambda x: x.nunique()})  # 统计一个start_loc有几个不同的end_loc
    UniqueLocNum3 = UniqueLocNum3.reset_index()
    result = pd.merge(result,UniqueLocNum,on=['userid','geohashed_start_loc'],how='left')
    result = pd.merge(result, UniqueLocNum2, on=['userid', 'geohashed_start_loc','hour_duan'], how='left')
    result = pd.merge(result, UniqueLocNum3, on=['userid', 'geohashed_start_loc','hour'], how='left')
    del result['hour']
    return result

# 统计在全集(train+test)中某个地点(所有数据)出现的次数，无论起点，终点
def loc_heat(train,result):
    train['hour'] = pd.to_datetime(train['starttime']).map(lambda x: x.strftime('%H'))
    train['hour_duan'] = train['hour'].map(hour_duan)
    result['hour'] = pd.to_datetime(result['starttime']).map(lambda x: x.strftime('%H'))
    result['hour_duan'] = result['hour'].map(hour_duan)
    train = train[['orderid','geohashed_start_loc','geohashed_end_loc','hour','hour_duan']]
    # test = test[['orderid','geohashed_start_loc']]
    train_end_loc = train[['orderid','geohashed_end_loc','hour','hour_duan']]
    train_end_loc = train_end_loc.rename(columns={'geohashed_end_loc':'geohashed_start_loc'})
    del train['geohashed_end_loc']
    loc = pd.concat([train,train_end_loc],ignore_index=True)
    start_loc_heat = loc.groupby('geohashed_start_loc',as_index=False)['orderid'].agg({'start_loc_heat':'count'})
    start_loc_heat2 = loc.groupby(['geohashed_start_loc','hour_duan'], as_index=False)['orderid'].agg({'start_loc_heat_hour_duan': 'count'})
    start_loc_heat3 = loc.groupby(['geohashed_start_loc','hour'], as_index=False)['orderid'].agg({'start_loc_heat_hour': 'count'})

    start_loc_heat.drop_duplicates(inplace=True)   # 不考虑时间因素
    start_loc_heat = start_loc_heat[['geohashed_start_loc', 'start_loc_heat']]

    start_loc_heat2.drop_duplicates(inplace=True)  # 时间段
    start_loc_heat2 = start_loc_heat2[['geohashed_start_loc','hour_duan','start_loc_heat_hour_duan']]

    start_loc_heat3.drop_duplicates(inplace=True)  # 时间点
    start_loc_heat3 = start_loc_heat3[['geohashed_start_loc', 'hour', 'start_loc_heat_hour']]
    # 不考虑时间
    end_loc_heat = start_loc_heat.copy()
    end_loc_heat = end_loc_heat.rename(columns={'geohashed_start_loc':'geohashed_end_loc','start_loc_heat': 'end_loc_heat'})
    # 时间段
    end_loc_heat2 = start_loc_heat2.copy()
    end_loc_heat2 = end_loc_heat2.rename(columns={'geohashed_start_loc': 'geohashed_end_loc', 'start_loc_heat_hour_duan': 'end_loc_heat_hour_duan'})
    # 时间点
    end_loc_heat3 = start_loc_heat3.copy()
    end_loc_heat3 = end_loc_heat3.rename(columns={'geohashed_start_loc': 'geohashed_end_loc', 'start_loc_heat_hour': 'end_loc_heat_hour'})

    result = pd.merge(result,start_loc_heat,on=['geohashed_start_loc'],how='left')
    result = pd.merge(result,end_loc_heat, on=['geohashed_end_loc'], how='left')
    result = pd.merge(result, start_loc_heat2, on=['geohashed_start_loc','hour_duan'], how='left')
    result = pd.merge(result, end_loc_heat2, on=['geohashed_end_loc','hour_duan'], how='left')
    result = pd.merge(result, start_loc_heat3, on=['geohashed_start_loc','hour'], how='left')
    result = pd.merge(result, end_loc_heat3, on=['geohashed_end_loc','hour'], how='left')

    result['start_loc_heat'] = result['start_loc_heat']/result['distance']
    result['start_loc_heat_hour_duan'] = result['start_loc_heat_hour_duan'] / result['distance']
    result['start_loc_heat_hour'] = result['start_loc_heat_hour'] / result['distance']
    result['end_loc_heat'] = result['end_loc_heat'] / result['distance']
    result['end_loc_heat_hour_duan'] = result['end_loc_heat_hour_duan'] / result['distance']
    result['end_loc_heat_hour'] = result['end_loc_heat_hour'] / result['distance']
    return result

# 是否为周末
def is_weekend(x):
    if x == 6 or x == 7:
        return 1
    return 0
# transfer to weekday
def get_week_day(datetime_element):
    date_to_weekday=datetime_element.starttime.dt.weekday
    return date_to_weekday + 1
# 加入时间->day
def add_weekday_week(result):
    # determine whether it is weekend
    result['week_day'] = result.starttime.dt.weekday
    result['is_weekend'] = result['week_day'].apply(lambda x: is_weekend(x))
    return result
#加入hour count
# def add_hour_count(result):
#     result['hour'] = result.starttime.dt.hour
#     hour_count=result.groupby(['userid','hour'],as_index=False)['hour'].agg({'hour_count': 'count'})
#     result=pd.merge(result,hour_count,on=['userid','hour'],how='left')
#     # del result['hour']
#     return result
#加入day count
def add_day_count(result):
    result['hour'] = result.starttime.dt.hour
    result['day'] = result.starttime.dt.day
    day_count = result.groupby(['userid','day'],as_index=False)['day'].agg({'day_count':'count'})
    hour_count = result.groupby(['userid', 'hour'], as_index=False)['hour'].agg({'hour_count': 'count'})
    result = pd.merge(result, hour_count, on=['userid', 'hour'], how='left')
    result=pd.merge(result,day_count,on=['userid','day'],how='left')
    return result
# 计算两点之间的欧氏距离
def get_distance_degree(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(geohash.decode(loc))
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    degree = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
        degree.append(cal_degree(lon1,lat1,lon2,lat2))
    result.loc[:,'distance'] = distance
    result.loc[:,'degree'] = degree
    return result

def get_distance(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(geohash.decode(loc))
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
    result.loc[:,'distance'] = distance
    return result

def get_distance_log(result):
    Result=result.copy()
    ex_list = list(Result.distance)
    ex_list = [ex_list[i] for i in filter(lambda i: ex_list[i] != 0, range(len(ex_list)))]
    Result = Result[Result.distance.isin(ex_list)]
    Result['distance_log'] = Result['distance'].apply(lambda x: math.log(x))
    result=pd.merge(result,Result[['orderid','distance_log','geohashed_start_loc','geohashed_end_loc']],on=['orderid','geohashed_start_loc','geohashed_end_loc'],how='left')
    return result
# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

# 对于分组，计算本条记录起终点与下一条记录起终点的距离差，时间差，速度
def help_bike_leak(group):
    start_loc = list(group['geohashed_start_loc'])
    end_loc = list(group['geohashed_end_loc'])
    time = list(group['starttime'])

    next_time_delta = []
    next_s_s_dis = []
    # next_e_e_dis = []
    next_s_e_dis = []
    next_e_s_dis = []
    # bike_s_s_v = []
    # bike_e_e_v = []
    for i in range(len(start_loc)-1):
        lat_start_i, lon_start_i = geohash.decode(start_loc[i])
        lat_start_i2, lon_start_i2 = geohash.decode(start_loc[i+1])
        lat_end_i, lon_end_i = geohash.decode(end_loc[i])
        lat_end_i2, lon_end_i2 = geohash.decode(end_loc[i])
        ## 本条起点与下条起点的距离差
        next_s_s_dis.append(cal_distance(lat_start_i, lon_start_i,lat_start_i2, lon_start_i2))
        ## 本条终点与下条终点的距离差
        # next_e_e_dis.append(cal_distance(lat_end_i, lon_end_i,lat_end_i2, lon_end_i2))
        ## 本条起点与下条终点的距离差
        next_s_e_dis.append(cal_distance(lat_start_i, lon_start_i,lat_end_i2, lon_end_i2))
        ## 本条终点与下条起点的距离差
        next_e_s_dis.append(cal_distance(lat_end_i, lon_end_i,lat_start_i2, lon_start_i2))
        ## 本条记录与下条记录的时间差
        # next_time_delta.append((datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S")-datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")).total_seconds())
        next_time_delta.append((time[i+1]-time[i]).total_seconds())

    # bike_s_s_v = [i / j for i, j in zip(next_s_s_dis, next_time_delta)]
    # bike_e_e_v = [i / j for i, j in zip(next_e_e_dis, next_time_delta)]
    next_s_s_dis.append(0)
    # next_e_e_dis.append(0)
    next_s_e_dis.append(0)
    next_e_s_dis.append(0)
    next_time_delta.append(1000000)
    # bike_s_s_v.append(0)
    # bike_e_e_v.append(0)

    group['next_s_s_dis'] = next_s_s_dis
    # group['next_e_e_dis'] = next_e_e_dis
    group['next_s_e_dis'] = next_s_e_dis
    group['next_e_s_dis'] = next_e_s_dis
    group['next_time_delta'] = next_time_delta
    # group['bike_s_s_v'] = bike_s_s_v
    # group['bike_e_e_v'] = bike_e_e_v
    return group

# def help_user_leak(group):
#     start_loc = list(group['geohashed_start_loc'])
#     end_loc = list(group['geohashed_end_loc'])
#     time = list(group['starttime'])
#
#     next_time_delta = []
#     next_s_s_dis = []
#     next_e_e_dis = []
#     next_s_e_dis = []
#     next_e_s_dis = []
#     user_s_s_v = []
#     user_e_e_v = []
#     for i in range(len(start_loc)-1):
#         # lat_start_i, lon_start_i = geohash.decode(start_loc[i])
#         lat_start_i2, lon_start_i2 = geohash.decode(start_loc[i+1])
#         lat_end_i, lon_end_i = geohash.decode(end_loc[i])
#         # lat_end_i2, lon_end_i2 = geohash.decode(end_loc[i])
#         ## 本条起点与下条起点的距离差
#         # next_s_s_dis.append(cal_distance(lat_start_i, lon_start_i,lat_start_i2, lon_start_i2))
#         ## 本条终点与下条终点的距离差
#         # next_e_e_dis.append(cal_distance(lat_end_i, lon_end_i,lat_end_i2, lon_end_i2))
#         ## 本条起点与下条终点的距离差
#         # next_s_e_dis.append(cal_distance(lat_start_i, lon_start_i,lat_end_i2, lon_end_i2))
#         ## 本条终点与下条起点的距离差
#         next_e_s_dis.append(cal_distance(lat_end_i, lon_end_i,lat_start_i2, lon_start_i2))
#         ## 本条记录与下条记录的时间差
#         # next_time_delta.append((datetime.strptime(time[i+1], "%Y-%m-%d %H:%M:%S")-datetime.strptime(time[i], "%Y-%m-%d %H:%M:%S")).total_seconds())
#         next_time_delta.append((time[i+1]-time[i]).total_seconds())
#
#     # user_s_s_v = [i / j for i, j in zip(next_s_s_dis, next_time_delta)]
#     # user_e_e_v = [i / j for i, j in zip(next_e_e_dis, next_time_delta)]
#     # next_s_s_dis.append(0)
#     # next_e_e_dis.append(0)
#     # next_s_e_dis.append(0)
#     next_e_s_dis.append(0)
#     next_time_delta.append(1000000)
#     # user_s_s_v.append(0)
#     # user_e_e_v.append(0)
#
#     # group['next_user_s_s_dis'] = next_s_s_dis
#     # group['next_user_e_e_dis'] = next_e_e_dis
#     # group['next_user_s_e_dis'] = next_s_e_dis
#     group['next_user_e_s_dis'] = next_e_s_dis
#     group['next_user_time_delta'] = next_time_delta
#     # group['user_s_s_v'] = user_s_s_v
#     # group['user_e_e_v'] = user_e_e_v
#     return group

# 对于同一bikeid本条记录的起终点与下条记录起终点的距离，时间差，速度
def bike_leak_features(result):
    result = result.sort_values(['bikeid','orderid'])
    result = result.groupby('bikeid').apply(help_bike_leak)
    return result

# 对于同一userid本条记录的起点与下条记录起终点的距离，时间差，速度
# def user_leak_features(result):
#     result = result.sort_values(['userid','orderid'])
#     result = result.groupby('userid').apply(help_user_leak)
#     return result

#加入count特征（不加每天时间段的情况）
def get_user_start_end_count(train,result):
    # train为历史记录，result为构建好的样本
    # 历史记录中用户的记录数
    user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
    # 历史记录中某一个地点作为终点的记录数
    end_count = train.groupby(['geohashed_end_loc'], as_index=False)['geohashed_end_loc'].agg({'end_count': 'count'})
    # 历史记录中某一个地点作为起点的记录数
    start_count = train.groupby(['geohashed_start_loc'], as_index=False)['geohashed_start_loc'].agg({'start_count': 'count'})
    # 历史记录中某一个用户到达某一个终点的记录数
    user_end_count = train.groupby(['userid','geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'user_end_count':'count'})
    # 历史记录中某一个用户到达某一个起点的记录数
    user_start_count = train.groupby(['userid','geohashed_start_loc'],as_index=False)['geohashed_start_loc'].agg({'user_start_count':'count'})
    # 历史记录中某一个起点到达某一个终点的记录数
    start_end_count = train.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['geohashed_end_loc'].agg({'start_end_count':'count'})
    # 历史记录中某一个用户从某一个起点到达某一个终点的记录数
    user_start_end_count= train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_start_end_count':'count'})
    # 历史记录中某一个用户从此起点到终点折返的记录数
    user_end_start_count = train.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg({'user_end_start_count': 'count'})
    user_end_start_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc', 'geohashed_end_loc': 'geohashed_start_loc'}, inplace=True)

    result = pd.merge(result,user_count,on=['userid'],how='left')
    result = pd.merge(result,end_count,on=['geohashed_end_loc'],how='left')
    result = pd.merge(result,start_count,on=['geohashed_start_loc'],how='left')
    result = pd.merge(result,user_end_count,on=['userid', 'geohashed_end_loc'],how='left')
    result = pd.merge(result,user_start_count,on=['userid', 'geohashed_start_loc'],how='left')
    result = pd.merge(result,start_end_count,on=['geohashed_start_loc','geohashed_end_loc'],how='left')
    result = pd.merge(result,user_start_end_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
    result = pd.merge(result, user_end_start_count, on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'],how='left')

    return result
#加入prob特征（不加每天时间段的情况，计算后验概率）
def get_user_start_end_prob(result):
    # result['user_prob'] = result.fenzi / result.user_count
    # result['start_prob'] = result.fenzi / result.start_count
    # result['end_prob'] = result.fenzi / result.end_count
    result['user_end_prob'] = result.user_end_count / result.user_count
    result['end_user_prob'] = result.user_end_count / result.end_count
    result['user_start_prob'] = result.user_start_count / result.user_count
    result['start_user_prob'] = result.user_start_count / result.start_count
    result['start_end_prob'] = result.start_end_count / result.start_count
    result['end_start_prob'] = result.start_end_count / result.end_count
    result['user_start_end_prob'] = result.user_start_end_count / result.user_count
    result['start_user_end_prob'] = result.user_start_end_count / result.start_count
    result['end_user_start_prob'] = result.user_start_end_count / result.end_count
    return result
#加入entropy特征（不加每天时间段的情况）
def get_user_start_end_entropy(result):
    # 样本中每一个用户，开始地点,结束地点的熵
    # result['user_entropy_each'] = result['user_prob'].apply(lambda x: -x * math.log(x))
    # user_entropy = result.groupby(['userid'], as_index=False)['user_entropy_each'].agg({'user_entropy': 'sum'})
    # result = pd.merge(result, user_entropy, on=['userid'], how='left')
    # result['start_entropy_each'] = result['start_prob'].apply(lambda x: -x * math.log(x))
    # start_entropy = result.groupby(['geohashed_start_loc'], as_index=False)['start_entropy_each'].agg({'start_entropy': 'sum'})
    # result = pd.merge(result, start_entropy, on=['geohashed_start_loc'], how='left')
    # result['end_entropy_each'] = result['end_prob'].apply(lambda x: -x * math.log(x))
    # end_entropy = result.groupby(['geohashed_end_loc'], as_index=False)['end_entropy_each'].agg({'end_entropy': 'sum'})
    # result = pd.merge(result, end_entropy, on=['geohashed_end_loc'], how='left')
    result['user_end_entropy_each'] = result['user_end_prob'].apply(lambda x: -x * math.log(x))
    user_end_entropy = result.groupby(['userid', 'geohashed_end_loc'], as_index=False)['user_end_entropy_each'].agg({'user_end_entropy': 'sum'})
    result = pd.merge(result, user_end_entropy, on=['userid', 'geohashed_end_loc'], how='left')
    result['end_user_entropy_each'] = result['end_user_prob'].apply(lambda x: -x * math.log(x))
    end_user_entropy = result.groupby(['userid', 'geohashed_end_loc'], as_index=False)['end_user_entropy_each'].agg({'end_user_entropy': 'sum'})
    result = pd.merge(result, end_user_entropy, on=['userid', 'geohashed_end_loc'], how='left')
    result['user_start_entropy_each'] = result['user_start_prob'].apply(lambda x: -x * math.log(x))
    user_start_entropy = result.groupby(['userid', 'geohashed_start_loc'], as_index=False)['user_start_entropy_each'].agg({'user_start_entropy': 'sum'})
    result = pd.merge(result, user_start_entropy, on=['userid', 'geohashed_start_loc'], how='left')
    result['start_user_entropy_each'] = result['start_user_prob'].apply(lambda x: -x * math.log(x))
    start_user_entropy = result.groupby(['userid', 'geohashed_start_loc'], as_index=False)['start_user_entropy_each'].agg({'start_user_entropy': 'sum'})
    result = pd.merge(result, start_user_entropy, on=['userid', 'geohashed_start_loc'], how='left')
    # del result['user_entropy_each'], result['start_entropy_each']
    del result['end_user_entropy_each'], result['user_start_entropy_each'], result['start_user_entropy_each']
    return result

#加入时间段，时间点的count，此函数中另外新添两个特征，hour,hour_duan
def get_user_start_end_hour_count(train, result):
    train['hour'] = pd.to_datetime(train['starttime']).map(lambda x: x.strftime('%H'))
    train['hour_duan'] = train['hour'].map(hour_duan)
    result['hour'] = pd.to_datetime(result['starttime']).map(lambda x: x.strftime('%H'))
    result['hour_duan'] = result['hour'].map(hour_duan)
    # 用户在某一时间段到达某终点的记录数
    user_end_hour_count = train.groupby(['userid', 'geohashed_end_loc', 'hour_duan'], as_index=False)['geohashed_end_loc'].agg({'user_end_hour_duan_count': 'count'})
    result = pd.merge(result, user_end_hour_count, on=['userid', 'geohashed_end_loc', 'hour_duan'], how='left')
    # 用户在某一时间段此路径的记录数
    user_start_end_hour_count = train.groupby(['userid', 'geohashed_start_loc','geohashed_end_loc', 'hour_duan'], as_index=False)['geohashed_start_loc'].agg({'user_start_end_hour_duan_count': 'count'})
    result = pd.merge(result, user_start_end_hour_count, on=['userid', 'geohashed_start_loc','geohashed_end_loc','hour_duan'], how='left')
    # 用户在某一时间点到达某终点的记录数
    user_end_hour_count2 = train.groupby(['userid', 'geohashed_end_loc', 'hour'], as_index=False)['geohashed_end_loc'].agg({'user_end_hour_count': 'count'})
    result = pd.merge(result, user_end_hour_count2, on=['userid', 'geohashed_end_loc', 'hour'], how='left')
    # 用户在某一时间点此路径的记录数
    user_start_end_hour_count2 = train.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc', 'hour'], as_index=False)['geohashed_start_loc'].agg({'user_start_end_hour_count': 'count'})
    result = pd.merge(result, user_start_end_hour_count2,on=['userid', 'geohashed_start_loc', 'geohashed_end_loc', 'hour'], how='left')
    del result['hour']
    return result

#加入时间段的prob
def get_user_start_end_hour_prob(result):
    result['user_end_hour_prob'] = result.user_end_hour_count / result.user_end_count
    # result['user_start_hour_prob'] = result.user_start_hour_count / result.user_start_count
    result['user_start_end_hour_prob'] = result.user_start_end_hour_count/result.user_hour_heat   # 用户在某个时间走某个路径的可能性
    result['user_start_end_hour_duan_prob'] = result.user_start_end_hour_duan_count / result.user_hour_duan_heat   # 用户在某个时间段走某个路径的可能性
    return result

# 加入时间，时间段的熵
def get_user_start_end_hour_entropy(result):
    result['user_end_hour_entropy_each'] = result['user_end_hour_prob'].apply(lambda x: -x * math.log(x))
    user_end_hour_entropy = result.groupby(['userid', 'geohashed_end_loc', 'hour_duan'], as_index=False)['user_end_hour_entropy_each'].agg({'user_end_hour_entropy': 'sum'})
    result = pd.merge(result, user_end_hour_entropy, on=['userid', 'geohashed_end_loc', 'hour_duan'], how='left')
    # result['user_start_hour_entropy_each'] = result['user_start_hour_prob'].apply(lambda x: -x * math.log(x))
    # user_start_hour_entropy = result.groupby(['userid', 'geohashed_start_loc', 'hour_duan'], as_index=False)['user_start_hour_entropy_each'].agg({'user_start_hour_entropy': 'sum'})
    # result = pd.merge(result, user_start_hour_entropy, on=['userid', 'geohashed_start_loc', 'hour_duan'], how='left')
    # del result['user_end_hour_entropy_each']
    return result

def cal_degree(lon1,lat1,lon2,lat2):
    dx = (lon2-lon1)*math.cos(lat1/57.2958)
    dy = lat2-lat1
    if((dx==0) and (dy==0)):
        degree = -1
    else:
        if(dy==0):
            dy = 1.0e-24
        degree = 180-90*(math.copysign(1,dy))-math.atan(dx/dy)*57.2958
        degree = round(degree/45)*45
        if degree>315:
           degree = 0
    return degree

def equi_dis(result):
    result['equi_dis'] = result['distance']/(result['start_end_count']**1.1)
    result['user_equi_dis'] = result['distance'] / (result['user_start_end_count'] ** 1.1)
    return result

# def hour_duan(hour):
#     hour = int(hour)
#     if ((hour >= 0) & (hour <=7))|((hour >= 23) & (hour <= 24)): #凌晨
#         return 0
#     elif ((hour >= 8) & (hour <= 10))|((hour >= 17) & (hour <=20)): #上下班高峰期
#         return 1
#     elif ((hour >= 11) & (hour <= 12))|((hour >= 15) & (hour <=16)): #上午和下午
#         return 2
#     elif (hour >=13) & (hour<=14): #中午
#         return 3
#     elif (hour >=21) & (hour<=22): #晚上
#         return 4
def hour_duan(hour):
    hour = int(hour)
    hotHourList1 = [7,8,9,12,13,14,18,19,20,21]
    hotHourList2 = [12, 13, 14]
    hotHourList3 = [18, 19, 20, 21]
    if hour in hotHourList1:
        return 1
    elif hour in hotHourList2:
        return 2
    elif hour in hotHourList3:
        return 3
    else:
        return 0

def eloc_decode(result):
    result['e_lon'] = result['geohashed_end_loc'].apply(lambda x: geohash.decode(x)[1])
    result['e_lat'] = result['geohashed_end_loc'].apply(lambda x: geohash.decode(x)[0])
    return result

def cluster_feature(train,result):
    result['e_lon'] = result['geohashed_end_loc'].apply(lambda x: geohash.decode(x)[1])
    result['e_lat'] = result['geohashed_end_loc'].apply(lambda x: geohash.decode(x)[0])
    coords = np.vstack((train[['s_lon', 's_lat']].values,
                        train[['e_lon', 'e_lat']].values))
    # sample_ind = np.random.permutation(len(coords))[:800000]
    # kmeans = MiniBatchKMeans(n_clusters=50, batch_size=10000).fit(coords[sample_ind])
    kmeans = MiniBatchKMeans(n_clusters=20, batch_size=10000).fit(coords[:800000])
    # kmeans = KMeans(n_clusters=40).fit(coords[:800000])
    result.loc[:, 's_cluster'] = kmeans.predict(result[['s_lon', 's_lat']])
    result.loc[:, 'e_cluster'] = kmeans.predict(result[['e_lon', 'e_lat']])
    # s_cluster = pd.get_dummies(result['s_cluster'], prefix='s', prefix_sep='_')
    # e_cluster = pd.get_dummies(result['e_cluster'], prefix='e', prefix_sep='_')
    # result = pd.concat([result,s_cluster],axis=1)
    # result = pd.concat([result,e_cluster],axis=1)
    return result

# GBDT将产生的新特征与之前的新特征融合在一起
def mergeToOne(X, X2):
    X3 = []
    for i in range(X.shape[0]):
        tmp = np.array([list(X[i]), list(X2[i])])
        X3.append(list(np.hstack(tmp)))
    X3 = np.array(X3)
    return X3

# 使用gbdt产生新特征
def gbdt_new_features(result):
    clf = XGBClassifier(
        learning_rate=0.2,         # 默认0.3
        n_estimators=30,            # 树的个数
        max_depth=7,
        min_child_weight=10,
        gamma=0.5,
        subsample=0.75,
        colsample_bytree=0.75,
        objective='binary:logistic',  # 逻辑回归损失函数
        nthread=8,                     # cpu线程数
        scale_pos_weight=1,
        reg_alpha=1e-05,
        reg_lambda=10,
        seed=1024)                      # 随机种子
    predictors = [i for i in result.columns if i not in
                  ['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'userid', 'bikeid', 'starttime', 'label',
                   'biketype', 'start_lon', 'start_lat', 'end_lon', 'end_lat']]
    old_features = result[predictors].values
    result1,result2= train_test_split(result, test_size=0.6, random_state=0)
    del result1
    clf.fit(result2[predictors].values,result2['label'].values)
    new_features = clf.apply(old_features)
    features = mergeToOne(old_features,new_features)
    return features    # array类型