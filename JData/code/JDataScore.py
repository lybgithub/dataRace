# 读取的csv文件都不包含表头，格式为:user_id,sku_id
import pandas as pd
# file1为预测数据文件，file2为真实数据文件,F11只关注是否下单
def f1_score(file1,file2):
    data_hat = pd.read_csv(file1, header=None)
    data_real = pd.read_csv(file2,header=None)
    count = 0.0
    preSum = float(len(data_hat.index))
    realSum = float(len(data_real.index))
    for i,d_hat in enumerate(data_hat.values):
        user_id = d_hat[0]
        sku_id = d_hat[1]
        for j,d_real in enumerate(data_real.values):
            user_id2 = d_real[0]         # 具体索引，视情况而定
            sku_id2 = d_real[1]   
            if(user_id==user_id2):
                count = count+1
                break
    Precise = count/preSum
    Recall = count/realSum 
    F11 = 6*Recall*Precise/(5*Recall+Precise)
    return F11
 # F2关注的是user_id和sku_id同时正确
def f2_score(file1,file2):
    data_hat = pd.read_csv(file1, header=None)
    data_real = pd.read_csv(file2,header=None)
    count = 0.0
    preSum = float(len(data_hat.index))
    realSum = float(len(data_real.index))
    for i,d_hat in enumerate(data_hat.values):
        user_id = d_hat[0]
        sku_id = d_hat[1]
        for j,d_real in enumerate(data_real.values):
            user_id2 = d_real[0]           # 具体索引，视情况而定
            sku_id2 = d_real[1]   
            if(user_id==user_id2 and sku_id==sku_id2):
                count = count+1
                break
    Precise = count/preSum
    Recall = count/realSum 
    F12 = 5*Recall*Precise/(2*Recall+3*Precise)
    return F12

    # 调用这个函数就好，总得分
def F(file1,file2):
    F11 = f1_score(file1,file2)
    F12 = f2_score(file1,file2)
    Score=0.4*F11 + 0.6*F12
    return Score


# 亲测可用~
if __name__ == "__main__":
    file1 = "I:/JData/pre.csv"
    file2 = "I:/JData/real.csv"
    score = F(file1,file2)
    print score