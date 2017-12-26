start_days = "2016-02-01"      # useless
user = get_basic_user_feat()   # encoded user_table
product = get_basic_product_feat() # encoded product_table
user_acc = get_accumulate_user_feat(start_days, train_end_date) # for each user,the transformed rate
product_acc = get_accumulate_product_feat(start_days, train_end_date) # for each sku,the transformed rate
comment_acc = get_comments_product_feat(train_start_date, train_end_date) # encoded the comment num
labels = get_labels(test_start_date, test_end_date) # the certain period's buy total

# new features:
buynum = get_buynumber(train_start_date,train_end_date)
sellnum = get_SellNumber(train_start_date,train_end_date)
user_buyNum = user_buy_num(train_start_date,train_end_date)
print(user_buyNum.head(2))
sku_buyedNum = sku_buyed_num(train_start_date,train_end_date)
print(sku_buyedNum.head(2))
# actions = get_accumulate_action_feat(train_start_date, train_end_date)
actions = None
for i in (60,10, 1, 3, 5, 7,2, 15, 21,30):
    start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
    start_days = start_days.strftime('%Y-%m-%d')
    if actions is None:
        actions = get_action_feat(start_days,train_end_date)
    else:
        actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                           on=['user_id', 'sku_id'])