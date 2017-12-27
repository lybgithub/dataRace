import matplotlib
matplotlib.use('Agg')                          # Force matplotlib to not use any Xwindows backend.
import matplotlib.pylab as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})   # 自适应调节图片的大小，解决xlabel显示不完全的bug
import gc
import time
import lightgbm as lgb
from genSample import *
from genFeatures import *
from genLibs import *

cache_path = 'mobike_cache_tail6/'
train_path = 'data/train.csv'
test_path = 'data/test.csv'
result_final_path = 'lgb_result_911_2153.csv'
lgb_image_path = 'image/lgb911_2153.jpg'
flag = True

# cache_path = 'mobike_cache_tail6/'
# train_path = 'data/train_small.csv'
# test_path = 'data/test_small.csv'
# result_final_path = 'lgb_result_911_2203.csv'
# lgb_image_path = 'image/lgb911_2203.jpg'
# flag = True

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
def make_train_set(train,test):
    print('开始构造样本...')
    result = get_sample(train,test)                                         # 构造备选样本

    print('开始构造特征...')
    result['starttime'] = pd.to_datetime(result['starttime'])
    result = get_distance_degree(result)                                    # 获取起始点和最终地点的欧式距离
    result = get_distance_log(result)                                       # log距离
    result = add_weekday_week(result)                                       # 增加weekday 和判断weekend
    result = add_day_count(result)                                          # 增加day_count特征
    # 不考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_count(train, result)
    result = get_user_start_end_prob(result)
    result = get_user_start_end_entropy(result)
    # 考虑时间段时间点的count,prob,entropy特征
    result = get_user_start_end_hour_count(train, result)
    result = bike_leak_features(result)                                     # 获取bikeid对应的leak特征
    result = add_hour_count(result)                                         # 计算hour和hour_count
    # result = user_leak_features(result)                                   # 获取userid对应的leak特征
    # result = cal_dis_ang(result)                                          # 加入方向
    result = equi_dis(result)                                               # 计算等效距离
    result.fillna(0,inplace=True)
    print('result.columns:\n{}'.format(result.columns))
    print('添加真实label')
    result = get_label(cache_path,train_path,test_path,result)
    return result

def lgbSubmission():
    t0 = time.time()
    train = pd.read_csv(train_path)  # 5/10~5/24  15天
    test = pd.read_csv(test_path)  # 2/25~6/1   8天
    # train['starttime'] = pd.to_datetime(train['starttime'])
    # test['starttime'] = pd.to_datetime(test['starttime'])   # 转化为datatime格式
    train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
    train2 = train[(train['starttime'] >= '2017-05-23 00:00:00')]    # 2天
    train2.loc[:, 'geohashed_end_loc'] = np.nan
    test.loc[:, 'geohashed_end_loc'] = np.nan

    print('构造训练集')
    train_feat = make_train_set(train1, train2)
    # train_feat.to_csv('train_feat.csv',index=False)
    print('构造线上测试集')
    test_feat = make_train_set(train, test)
    # test_feat.to_csv('test_feat.csv', index=False)
    del train, test, train1, train2

    predictors = [i for i in train_feat.columns if i not in
                  ['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'bikeied','userid','starttime', 'label',
                   'start_lon','biketype','week_day','is_weekend','day','start_lat', 'end_lon', 'end_lat']]
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        # 'num_leaves': 31,
        'max_depth':12,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.886,
        'bagging_freq': 5,
        'lambda_l1':10,
        'lambda_l2':10,
        'num_threads':8,
        'verbose': 0
    }

    lgb_train = lgb.Dataset(train_feat[predictors].values, train_feat['label'].values)
    # lgb_test = lgb.Dataset(test_feat[predictors].values)  #cannot use dataset instance for prediction
    lgb_test = test_feat[predictors].values
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=120,
                    # valid_sets=lgb_eval,
                    # categorical_feature=[1, 2, 3],
                    # early_stopping_rounds=10
                    )
    feat_imp = pd.Series(gbm.feature_importance(importance_type='split'), index=predictors).sort_values(ascending=False)
    print('Start ploting...')
    feat_imp.plot(kind='bar', title='Real Feature Importances')
    plt.ylabel(' Feature Importance Score')
    # plt.show()
    plt.ylabel('Real Feature Importance Score')
    out_png = lgb_image_path
    plt.savefig(out_png)

    del train_feat, lgb_train
    gc.collect()

    test_feat.loc[:, 'pred'] = gbm.predict(lgb_test)
    result = reshape(test_feat)
    test = pd.read_csv(test_path)
    result = pd.merge(test[['orderid']], result, on='orderid', how='left')
    result.fillna('0', inplace=True)
    result.to_csv(result_final_path, index=False, header=False)
    print('一共用时{}秒'.format(time.time() - t0))

if __name__ == "__main__":
    lgbSubmission()
    # xgbCV()
