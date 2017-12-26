def sku_buyed_num(start_date,end_date):
        dump_path = '../cache/sku_buyed_num_stat_%s_%s.plk' % (start_date, end_date)
        if os.path.exists(dump_path):
            sku_buyedNum = pickle.load(open(dump_path,'rb'))
        else:
            actions = get_actions(start_date,end_date)
            actions = actions.reset_index()
            
            timeSeries = actions['time']
            timeList = list(timeSeries)    
            dateList = [i.split(' ')[0] for i in timeList]       
            dateSeries = pd.Series(dateList)    
            df_date = dateSeries.to_frame(name='date') 
            
            df_new = pd.concat([actions,df_date],axis=1) 
              
            df_new = df_new[(df_new.type==4)] 
            df_new = df_new[(df_new.cate==8)] 
            df_new = df_new.drop_duplicates(['sku_id','date'])
            df_new = df_new[['user_id','sku_id','type','date']]
            sku_buyedNum = df_new.groupby(['sku_id'],as_index=False).sum()
            sku_buyedNum['sku_buyed_num'] = sku_buyedNum['type']/4
            sku_buyedNum['sku_buyed_num'] = sku_buyedNum['sku_buyed_num'].astype(int)
            
            del(sku_buyedNum['type'])
            del(sku_buyedNum['user_id'])
            pickle.dump(sku_buyedNum, open(dump_path, 'wb'))
        return sku_buyedNum

