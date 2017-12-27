import geohash
import numpy as np
import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV #网格搜索
from xgboost.sklearn import XGBClassifier
from sklearn import  metrics
import matplotlib.pylab as plt

def tiaocan(clf,train_x, train_y,predictors):

    #根据xgb自带的cv得到n_estimator(即树的数目）
    xgb_param = clf.get_xgb_params()
    xgtrain = xgb.DMatrix(train_x, label=train_y)
    # cv结果
    cv_folds = 5
    early_stopping_rounds = 50
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds)  # 是否显示目前几颗树额
    clf.set_params(n_estimators=cvresult.shape[0])
    tree_number = cvresult.shape[0]
    print('xgb自带的cv得到的树的数目',tree_number)
    clf.fit(train_x, train_y, eval_metric='auc')
    #查看model特征
    feat_imp =pd.Series(clf.feature_importances_, index=predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    # 预测
    train_predictions = clf.predict(train_x)
    train_predprob = clf.predict_proba(train_x)[:, 1]  # 1的概率
    # 打印
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, train_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_y, train_predprob))

    # 第二步：max_depth 和 min_child_weight 参数调优。n_estimator是上一步输出的
    param_test1 = {'max_depth': range(3, 15, 2), 'min_child_weight': range(1, 6, 2)}
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=tree_number, max_depth=5, min_child_weight=1,
                                gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                nthread=4, scale_pos_weight=1, seed=27), param_grid=param_test1, scoring='roc_auc',iid=False, cv=5)
    gsearch1.fit(train_x, train_y)
    print('第一步调参分数', gsearch1.grid_scores_)
    print('第一步调参最佳参数', gsearch1.best_params_)
    print('第一步调参最佳分数', gsearch1.best_score_)

    # 第三步：gamma参数调优
    param_test2 = {'gamma': [i  for i in range(10, 40)]}
    gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=tree_number,
                                                    max_depth=gsearch1.best_params_['max_depth'],
                                                    min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0,
                                                    subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                                    nthread=4,
                                                    scale_pos_weight=1, seed=27), param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch2.fit(train_x, train_y)
    print('第2步调参分数', gsearch2.grid_scores_)
    print('第2步调参最佳参数', gsearch2.best_params_)
    print('第2步调参最佳分数', gsearch2.best_score_)

    #第四步：调整subsample 和 colsample_bytree 参数
    param_test3 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=tree_number, max_depth=gsearch1.best_params_['max_depth'],
                                                    min_child_weight=gsearch1.best_params_['min_child_weight'],
                                                    gamma=gsearch2.best_params_['gamma'], subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch3.fit(train_x, train_y)
    print('第3步调参分数', gsearch3.grid_scores_)
    print('第3步调参最佳参数', gsearch3.best_params_)
    print('第3步调参最佳分数', gsearch3.best_score_)

    #第五步：正则化参数调优
    param_test4 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100] }
    gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=tree_number, max_depth=gsearch1.best_params_['max_depth'],
                                                    min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=gsearch2.best_params_['gamma'], subsample=gsearch3.best_params_['subsample'],
                                                    colsample_bytree=gsearch3.best_params_['colsample_bytree'],objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(train_x, train_y)

    return tree_number,gsearch1.best_params_,gsearch2.best_params_,gsearch3.best_params_,gsearch4.best_params_

# 相差的分钟数
def diff_of_minutes(time1, time2):
    d = {'5': 0, '6': 31, }
    try:
        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
        try:
            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

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

# 分组排序 feat1->orderid, feat2->pred
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 对结果进行整理
def reshape(pred):
    result = pred.copy()
    result = rank(result,'orderid','pred',ascending=False)
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result

def reshape_prob(pred):
    result = pred.copy()
    result = rank(result, 'orderid', 'pred', ascending=False)
    result = result[result['rank'] < 3][['orderid', 'geohashed_end_loc', 'rank','pred']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid',0,1,2,00,11,22]
    return result

# 测评函数
def map1(cache_path,result,testCV):
    result_path = cache_path + 'CV_true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        # train = pd.read_csv(train_path)
        true = dict(zip(testCV['orderid'].values,testCV['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    return score

# 获取真实标签
def get_label(cache_path,train_path,test_path,data):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test['geohashed_end_loc'] = np.nan
        DATA = pd.concat([train,test])
        true = dict(zip(DATA['orderid'].values, DATA['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data['label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    return data