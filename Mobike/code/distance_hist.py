
#move column3 in leak_result_830_1823 to result_830_0028
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import time
import geohash
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def loaddata(train_path):
    df_train = pd.read_csv(train_path)
    return df_train

# 计算两点之间的欧氏距离
def get_distance(result):
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(geohash.decode(loc))
        # deloc.append(geohash.decode_exactly(loc)[:2])
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
    # result.loc[:,'distance'] = distance
    result['distance']=distance
    return result[['orderid','distance']]
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

def distance_classfication():
    #直方图分类
    pass

def Thermal_chart(df,Thermal_chartPath):
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.orderid)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.distance)
    # plt.show()
    plt.savefig(Thermal_chartPath)
if __name__ == '__main__':
    train_path = 'data/train.csv'  # 训练数据的路径
    xgb_image_path = 'image/server_hist100_trick_{}'.format(time.time()) + '.jpg'  # 输出图片的路径
    # distance_path = 'image/server_ditance_train.csv'    # 输出距离统计csv
    # Thermal_chartPath = 'image/thermal_{}'.format(time.time()) + '.jpg'  # 输出热力图的路径
    train = loaddata(train_path)
    result = get_distance(train)
    # Thermal_chart(result, Thermal_chartPath)
    # result.sort_values('distance').to_csv(distance_path,index=False)
    series_result=pd.Series(result.distance,index=result.orderid).sort_values(ascending=False)
    # plt.xlim((0,10000))         #x轴范围
    new_ticks = np.linspace(0, 1000, 40000)
    plt.xticks(new_ticks)
    series_result.plot.hist(stacked=True, bins=100)
    plt.savefig(xgb_image_path)
