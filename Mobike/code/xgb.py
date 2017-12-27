import matplotlib
matplotlib.use('Agg')                          # Force matplotlib to not use any Xwindows backend.
import matplotlib.pylab as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})   # 自适应调节图片的大小，解决xlabel显示不完全的bug
import gc
import time
import xgboost as xgb
from genSample import *
from genFeatures import *
from genLibs import *


cache_path = 'mobike_cache_cluster/'
train_path = 'data/train.csv'
test_path = 'data/test.csv'
resultPath = 'result_925_1849_3day.csv'
result_probPath = 'result_prob_925_1849_3day.csv'
xgb_image_path = 'image/xgb925_1849.jpg'
flag = True

# 构造样本
def get_sample(train,test):
    result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(cache_path,train, test)            # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_end_loc']
        user_start_loc = get_user_start_loc(cache_path,train, test)        # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_start_loc']
        loc_to_loc = get_loc_to_loc(cache_path,train, test)                # 筛选起始地点去向最多的3个地点
        leak_bike = leak_bikeid(cache_path, train, test)
        leak_user = leak_userid(cache_path,train,test)
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            leak_bike[['orderid','geohashed_end_loc']],
                            leak_user[['orderid','geohashed_end_loc']]
                            ]).drop_duplicates()
        # 根据end_loc添加标签(0,1)
        test_temp = test.copy()
        test_temp.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
        result = pd.merge(result, test_temp, on='orderid', how='left')
        result['label'] = (result['label'] == result['geohashed_end_loc']).astype(int)
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=6)
    return result

# 制作训练集
def make_train_set(train_all,train,test):
    print('开始构造样本...')
    result = get_sample(train,test)                                         # 构造备选样本

    print('开始构造特征...')
    result = get_time(result)                                               # 转化为s的特征
    result['starttime'] = pd.to_datetime(result['starttime'])
    # result = cluster_feature(train_all, result)  # 产生聚类的特征
    # result.to_csv('trainFea.csv', index=False)
    result = get_distance_degree(result)                                    # 获取起始点和最终地点的欧式距离
    result = get_distance_log(result)                                       # log距离
    result = add_weekday_week(result)                                       # 增加weekday 和判断weekend

    # 不考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_count(train, result)
    result = get_user_start_end_prob(result)
    result = get_user_start_end_entropy(result)
    # 考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_hour_count(train, result)


    result = bike_leak_features(result)                                     # 获取bikeid对应的leak特征
    result = loc_heat(train_all,result)                                     # 某个地点的热度
    result = user_heat(train,result)                                        # 某个用户的热度，是否是常用用户
    result = user_sloc_eloc_unique(train, result)                           # 用户从某个地点可选择的终点有几个

    result = get_user_start_end_hour_prob(result)  # 加上时间特征的概率
    result = get_user_start_end_hour_entropy(result)  # 加上时间特征的熵

    result = add_day_count(result)                                          # 增加day_count,hour_count特征
    # result = add_hour_count(result)                                       # 计算hour和hour_count
    # result = user_leak_features(result)                                   # 获取userid对应的leak特征
    # result = cal_dis_ang(result)                                          # 加入方向
    result = equi_dis(result)                                               # 计算等效距离
    result.fillna(0,inplace=True)
    print('result.columns:\n{}'.format(result.columns))
    print('添加真实label')
    result = get_label(cache_path,train_path,test_path,result)
    return result

def make_test_set(train,test):
    print('开始构造样本...')
    result = get_sample(train,test)                                         # 构造备选样本

    # result = eloc_decode(result)                                          # 给终点加上经纬度
    result = get_time(result)                                               # 转化为s的特征
    result['starttime'] = pd.to_datetime(result['starttime'])
    print('开始构造特征...')

    # result = cluster_feature(train, result)                                 # 产生聚类的特征
    # result.to_csv('testFea.csv',index=False)
    result = get_distance_degree(result)                                    # 获取起始点和最终地点的欧式距离
    result = get_distance_log(result)                                       # log距离
    result = add_weekday_week(result)                                       # 增加weekday 和判断weekend

    # 不考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_count(train, result)
    result = get_user_start_end_prob(result)
    result = get_user_start_end_entropy(result)
    # 考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_hour_count(train, result)


    result = bike_leak_features(result)                                     # 获取bikeid对应的leak特征
    result = loc_heat(train, result)  # 某个地点的热度
    result = user_heat(train, result)  # 某个用户的热度，是否是常用用户
    result = user_sloc_eloc_unique(train, result)  # 用户从某个地点可选择的终点有几个
    result = add_day_count(result)                                          # 增加day_count,hour_count特征

    result = get_user_start_end_hour_prob(result)  # 加上时间特征的概率
    result = get_user_start_end_hour_entropy(result)  # 加上时间特征的熵

    # result = add_hour_count(result)                                       # 计算hour和hour_count
    # result = user_leak_features(result)                                   # 获取userid对应的leak特征
    # result = cal_dis_ang(result)                                          # 加入方向
    result = equi_dis(result)                                               # 计算等效距离
    result.fillna(0,inplace=True)
    print('result.columns:\n{}'.format(result.columns))
    print('添加真实label')
    result = get_label(cache_path,train_path,test_path,result)
    return result

def xgbSubmission():
    t0 = time.time()
    train = pd.read_csv(train_path)  # 5/10~5/24  15天
    test = pd.read_csv(test_path)  # 2/25~6/1   8天
    # train['starttime'] = pd.to_datetime(train['starttime'])
    # test['starttime'] = pd.to_datetime(test['starttime'])   # 转化为datatime格式
    train1 = train[(train['starttime'] < '2017-05-22 00:00:00')]
    train2 = train[(train['starttime'] >= '2017-05-22 00:00:00')]    # 3天
    # train1, train2 = train_test_split(train, test_size=0.35, random_state=0)  # train:test -> 15:8
    train2.loc[:, 'geohashed_end_loc'] = np.nan
    test.loc[:, 'geohashed_end_loc'] = np.nan

    print('构造训练集')
    train_feat = make_train_set(train,train1, train2)
    # train_feat.to_csv('train_feat.csv',index=False)
    print('构造线上测试集')
    test_feat = make_test_set(train, test)
    # test_feat.to_csv('test_feat.csv', index=False)
    del train, test, train1, train2

    predictors = [i for i in train_feat.columns if i not in
                  ['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'bikeied', 'userid', 'starttime', 'label',
                    'biketype', 'week_day','day', 's_lon','s_lat', 'e_lon', 'e_lat','index','index_x','is_weekend','hour_duan','user_heat',
                   'index_y','user_start_end_hour_duan_count','start_loc_heat_hour_duan','user_start_end_entropy_each','start_loc_heat',
                   'user_hour_duan_heat','user_start_end_entropy']]
    params = {
        'objective': 'binary:logistic',
        'eta': 0.01,
        'colsample_bytree': 0.886,
        'min_child_weight': 30,    #8
        'max_depth': 10,           #12
        'subsample': 0.886,
        'alpha': 10,
        'gamma': 30,
        'lambda': 50,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 1,
        'seed': 201703,
        'missing': -1
    }

    xgbtrain = xgb.DMatrix(train_feat[predictors], train_feat['label'])
    xgbtest = xgb.DMatrix(test_feat[predictors])
    model = xgb.train(params, xgbtrain, num_boost_round=1500)
    feat_imp = pd.Series(model.get_fscore(), index=predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Real Feature Importance Score')
    out_png = xgb_image_path
    plt.savefig(out_png)
    del train_feat, xgbtrain
    gc.collect()

    test_feat.loc[:, 'pred'] = model.predict(xgbtest)
    result = reshape(test_feat)
    result_prob = reshape_prob(test_feat)      # 输出概率结果
    test = pd.read_csv(test_path)
    result = pd.merge(test[['orderid']], result, on='orderid', how='left')
    result_prob = pd.merge(test[['orderid']], result_prob, on='orderid', how='left')
    result.fillna('0', inplace=True)
    result_prob.fillna('0', inplace=True)
    result.to_csv(resultPath, index=False, header=False)
    result_prob.to_csv(result_probPath, index=False, header=False)
    print('一共用时{}秒'.format(time.time() - t0))


if __name__ == "__main__":
    xgbSubmission()
    # xgbCV()
