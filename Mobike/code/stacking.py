from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import gc
import time
import lightgbm as lgb
from genSample import *
from genFeatures import *
from genLibs import *

cache_path = 'mobike_cache_tail6/'
train_path = 'data/train.csv'
test_path = 'data/test.csv'
result_final_path = 'stacking_result_912_1811.csv'
# lgb_image_path = 'image/stacking911_2153.jpg'
flag = True

# cache_path = 'mobike_cache_tail6/'
# train_path = 'data/train_small.csv'
# test_path = 'data/test_small.csv'
# result_final_path = 'stacking_result_912_1040.csv'
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

def loadData():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
    train2 = train[(train['starttime'] >= '2017-05-23 00:00:00')]  # 2天
    train2.loc[:, 'geohashed_end_loc'] = np.nan
    test.loc[:, 'geohashed_end_loc'] = np.nan
    print('构造训练集')
    train_feat = make_train_set(train1, train2)
    print('构造线上测试集')
    test_feat = make_train_set(train, test)
    del train, test, train1, train2
    predictors = [i for i in train_feat.columns if i not in
                  ['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'bikeied', 'userid', 'starttime', 'label',
                   'start_lon', 'biketype', 'week_day', 'is_weekend', 'day', 'start_lat', 'end_lon', 'end_lat']]
    X = train_feat[predictors]
    y = train_feat['label']
    X_submission = test_feat[predictors]
    return  X.values, y.values, X_submission.values,test_feat

def stackingSubmission():
    n_folds = 3
    # verbose = True
    shuffle = False
    t0 = time.time()
    X, y, X_submission, test_feat = loadData()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [
        # GradientBoostingClassifier(learning_rate=0.05, subsample=0.8, max_depth=12, n_estimators=200),
        'xgboost',
        'lightGBM'
    ]
    params_xgb = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 12,
        'subsample': 0.886,
        'alpha': 10,
        'gamma': 30,
        'lambda': 50,
        'verbose_eval': True,
        'nthread': 8,
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'seed': 201703,
        'missing': -1
    }
    params_lgb = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 128,
        # 'max_depth': 12,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.886,
        'bagging_freq': 5,
        'lambda_l1': 10,
        'lambda_l2': 10,
        'num_threads': 8,
        'verbose': 0
    }

    print("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        if (clf == 'xgboost'):
            for i, (train, test) in enumerate(skf):
                print('fold', i)
                X_train = X[train]
                y_train = y[train]
                X_test = X[test]
                xgbtrain = xgb.DMatrix(X_train, y_train)
                xgbtest = xgb.DMatrix(X_test)
                xgbsubmission = xgb.DMatrix(X_submission)
                model = xgb.train(params_xgb, xgbtrain, num_boost_round=120)
                y_submission = model.predict(xgbtest)
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = model.predict(xgbsubmission)
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        elif (clf == 'lightGBM'):
            for i, (train, test) in enumerate(skf):
                print('fold', i)
                X_train = X[train]
                y_train = y[train]
                X_test = X[test]
                lgb_train = lgb.Dataset(X_train, y_train)
                print('Start lgb training...')
                # train
                gbm = lgb.train(params_lgb,
                                lgb_train,
                                num_boost_round=120,
                                # valid_sets=lgb_eval,
                                # categorical_feature=[21],
                                # early_stopping_rounds=5
                                )
                # predict
                y_submission = gbm.predict(X_test)
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = gbm.predict(X_submission)
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        else:
            for i, (train, test) in enumerate(skf):
                print('fold', i)
                # print(test)
                X_train = X[train]
                y_train = y[train]
                X_test = X[test]
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                dataset_blend_train[test, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print('stacking')
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    gc.collect()

    test_feat.loc[:, 'pred'] = clf.predict_proba(dataset_blend_test)[:, 1]
    result = reshape(test_feat)
    test = pd.read_csv(test_path)
    result = pd.merge(test[['orderid']], result, on='orderid', how='left')
    result.fillna('0', inplace=True)
    result.to_csv(result_final_path, index=False, header=False)
    print('一共用时{}秒'.format(time.time() - t0))

if __name__ == "__main__":
    stackingSubmission()
    # xgbCV()
