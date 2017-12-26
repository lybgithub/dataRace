from gen_feat import make_train_set
from gen_feat import make_test_set
from gen_feat import get_labels_8
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier as gbdt
from sklearn.linear_model import LogisticRegression as lg
from gen_feat import report
from numpy import float32
import numpy as np 
from sklearn.preprocessing import Imputer

def getKey(item):
        return item[1]

def logistic_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    
    y_train = list(map(int, y_train))
    # print(np.any(np.isnan(X_train)))
    # print(np.all(np.isfinite(X_train)))
    clf = lg()  # 使用类，参数全是默认的  
    clf.fit(X_train,y_train)
    

    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)

    y_hat = clf.predict(sub_trainning_data.values)
    sub_user_index['label'] = y_hat
    pred = sub_user_index[sub_user_index['label'] == 1]    
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('../sub/submissionLOG508.csv', index=False, index_label=False)




def gbdt_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    training_data = training_data.fillna(0)
    print(training_data.info())
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    # X_train = X_train.astype(int)
    y_train = list(map(int, y_train))
    
    param = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 1.0,
          'learning_rate': 0.01, 'min_samples_leaf': 1,'random_state': 3,'max_features':0.8}
    clf = gbdt(param)

    clf.fit(X_train,y_train)
    

    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)

    sub_trainning_data = sub_trainning_data.fillna(0)


    y_hat = clf.predict(sub_trainning_data.values)
    sub_user_index['label'] = y_hat
    pred = sub_user_index[sub_user_index['label'] == 1]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('../sub/submissionGBDT508.csv', index=False, index_label=False)

def gbdt_cv():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-06'
    test_end_date = '2016-04-11'

    sub_start_date = '2016-03-10'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    param = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 1.0,
          'learning_rate': 0.01, 'min_samples_leaf': 1,'random_state': 3,'max_features':0.8}
    clf = gbdt(param)
    clf.fit(X_train,y_train)
    
    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)   # use this data to see the offline score
    
    test = sub_trainning_date.values    
    y = clf.predict(test)

    pred = sub_user_index.copy()
    y_true = get_labels_8(sub_test_start_date, sub_test_end_date)   # during the test date, real label for cate 8
    # y_true = sub_user_index.copy()
    pred['label'] = y    # add the new column which is the predict label for the test date

    ans = []
    for i in range(0,30):
        pred = sub_user_index.copy()
        pred['label'] = y
        pred = pred[pred.label >= i / 100]
        # print(pred)
        rep = report(pred, y_true)
        print('%s : score:%s' %(i/100,rep))
        ans.append([i / 100, rep])

    print('ans:%s' %ans)

    threshold = sorted(ans, key=getKey, reverse=True)[0][0]
    bestscore = sorted(ans, key=getKey, reverse=True)[0][1]
    print('best threshold:%s' % threshold)
    print('best score:%s' % bestscore)
    
    
def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0) # select some features
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)  # don't use these
    param = {'learning_rate' : 0.05, 'n_estimators': 1000, 'max_depth': 5, 
        'min_child_weight': 1, 'gamma': 0, 'subsample': 1, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 20
    param['nthread'] = 5
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst = list(plst)
    plst += [('eval_metric', 'auc')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    # make test data
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)   # predict this subdata,the DMatrix Object is array
    
    y_hat = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y_hat
    pred = sub_user_index[sub_user_index['label'] >= 0.05]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('../sub/submission424.csv', index=False, index_label=False)



def xgboost_cv2():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-06'
    test_end_date = '2016-04-11'

    sub_start_date = '2016-03-10'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)

    dtrain = xgb.DMatrix(X_train.values, label=y_train)
    dtest = xgb.DMatrix(X_test.values, label=y_test)
    
    param = {'learning_rate' : 0.05, 'n_estimators': 1000, 'max_depth':10,
        'min_child_weight': 1, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic','eval_metric':'auc'}
    num_round = 300

    param['nthread'] = 5
    #     param['eval_metric'] = "auc"
    #     plst = param.items()
    #     plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(param, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)   # use this data to see the offline score
    test = xgb.DMatrix(sub_trainning_date.values)    
    y = bst.predict(test)

    pred = sub_user_index.copy()
    y_true = get_labels_8(sub_test_start_date, sub_test_end_date)   # during the test date, real label for cate 8
    # y_true = sub_user_index.copy()
    pred['label'] = y    # add the new column which is the predict label for the test date
    # print(pred[(pred.label >= 0.12)].shape)
    # print("y_true:") 
    # print(y_true) 
    # pred = pred[(pred.label >= 0.35)]
    # print(len(pred))
    # print(pred)

    ans = []
    for i in range(0,30):
        pred = sub_user_index.copy()
        pred['label'] = y
        pred = pred[pred.label >= i / 100]
        # print(pred)
        rep = report(pred, y_true)
        print('%s : score:%s' %(i/100,rep))
        ans.append([i / 100, rep])

    print('ans:%s' %ans)

    threshold = sorted(ans, key=getKey, reverse=True)[0][0]
    bestscore = sorted(ans, key=getKey, reverse=True)[0][1]
    print('best threshold:%s' % threshold)
    print('best score:%s' % bestscore)


def xgboost_cv():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
   # param['eval_metric'] = "auc"
    plst = param.items()
    plst = list(plst)
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    test = xgb.DMatrix(sub_trainning_date)
    y = bst.predict(test)

    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y   
    y_true['label'] = sub_label
    report(pred, y_true)
    # report(y, y_true)


if __name__ == '__main__':
    # xgboost_cv2()
    logistic_make_submission()
    # gbdt_make_submission()
    # xgboost_make_submission()
    # train_start_date = '2016-03-10'
    # train_end_date = '2016-04-11'
    # test_start_date = '2016-04-11'
    # test_end_date = '2016-04-16'

    # sub_start_date = '2016-03-15'
    # sub_end_date = '2016-04-16'

    # sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)

