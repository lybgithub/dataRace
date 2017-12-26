#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import csv
from datetime import datetime

if __name__ == "__main__":
    action_1_path = "../JData_Action_201602.csv"
    action_2_path = "../JData_Action_201603.csv"
    action_3_path = "../JData_Action_201604.csv"

    action_1 = pd.read_csv(action_1_path)
    action_2 = pd.read_csv(action_2_path)
    action_3 = pd.read_csv(action_3_path)
    actions = pd.concat([action_1, action_2, action_3],ignore_index=True)

    find_potential_user(actions)

def more_than_one_day(group):
    
    if(len(group[group["type"] == 4])==0):
        group["potential_flag"] = 2
    else:
        last_buy_day = max(group[group["type"] == 1]["date"])
        earliest_behave_day = min(group["date"])  
        if(last_buy_day - earliest_behave_day).days > 0: 
            group["potential_flag"] = 1
        else:
            group["potential_flag"] = 0
    return group
def find_potential_user(df):
    df['date'] = pd.to_datetime(df['time']).dt.date
    df = df.groupby(['user_id','sku_id']).apply(more_than_one_day)
    df = df[['user_id','sku_id','potential_flag']]
    df.to_csv("../potential_ui.csv",index=False)
