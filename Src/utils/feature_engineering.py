import pandas as pd

import os
import numpy as np
import gc
import tqdm
import itertools

def get_sample_sales_unique(df_sales,df_submission_sample):
    df_sales2_n = pd.concat([df_sales,df_submission_sample])

    unique_values = []

    uniques_s100 = df_sales2_n['S100'].unique()
    for value_s100 in tqdm.tqdm(uniques_s100):
        sub_group_s100 = df_sales[(df_sales['S100']==value_s100)]
        uniques_i100   = sub_group_s100['I100'].unique()
        for value_i100 in uniques_i100:
            sub_group_i100 = sub_group_s100[(sub_group_s100['I100']==value_i100)]
            uniques_c100   = sub_group_i100['C100'].unique()
            for value_c100 in uniques_c100:
                sub_group_c100 = sub_group_i100[(sub_group_i100['C100']==value_c100)]
                uniques_c101   = sub_group_c100['C101'].unique()
                for value_c101 in uniques_c101:
                    unique_values.append([value_s100,value_i100,value_c100,value_c101])
    del df_sales2_n
    gc.collect()


    unique_values = pd.DataFrame(unique_values,columns = ['S100','I100','C100','C101'],index=None)
    unique_values['index']=unique_values.index
    dates = []
    cnt = 0

    actual_date = df_sales['DATE'].min()

    while(True):
        dates.append(actual_date)
        if actual_date>=df_submission_sample['DATE'].max():
            break
        actual_date = actual_date+ pd.DateOffset(days=7)

    unique_dates = pd.DataFrame(dates,columns = ['DATE'],index=None)
    df2 = pd.DataFrame([e for e in itertools.product(unique_values.index, unique_dates.DATE)], columns=['index','DATE'])
    sample_sales_unique = unique_values.merge(df2,on=['index']).drop(columns=['index'])
    del df2
    gc.collect()
    
    return sample_sales_unique


def fe_dates(df):
    
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    df['year']  = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day']   = df['DATE'].dt.day
    
    
    df['day_of_week'] = df['DATE'].dt.day_of_week
    df['day_of_year'] = df['DATE'].dt.day_of_year
    
    df['is_year_start']    = df['DATE'].dt.is_year_start
    df['is_quarter_start'] = df['DATE'].dt.is_quarter_start
    df['is_month_start']   = df['DATE'].dt.is_month_start
    df['is_month_end']    = df['DATE'].dt.is_month_end
    
    return df


################

# Claculate groupby statics for lag date 
def calc_stats(df, end,window,groupby=None,aggregates='mean',value='QTT'):
    
    # dates
    last_date = pd.to_datetime(end) - pd.Timedelta(days=7)
    first_date = pd.to_datetime(end) - pd.Timedelta(days= window)
    # Aggregate
    df1 = df[(df.DATE >=first_date) & (df.DATE<= last_date) ]
    df_agg = df1.groupby(groupby)[value].agg(aggregates)
    # Change name of columns
    df_agg.name =  str(end).split(' ')[0]+'_' + '_'.join(groupby)+'_'+aggregates+'_'+ str(window)
    return df_agg.reset_index()

#sales_by_store_item
def sales_by_store_item(df, end, aggregates='mean', value='QTT'):
    
    print('Adding sales by store item')
    data = calc_stats(df,end, window=1,aggregates=aggregates, 
                      groupby=['S100','I100'], value=value)
    print('window 1 added')
    
    for window in  [14,28,90,180,365]:
        agg = calc_stats(df,end, window=window, aggregates=aggregates,
                         groupby=['S100','I100'], value=value )
        data = pd.merge(data,agg)
        print('window %d added'% window)
    return data

# sales by store item dayofweek
def sales_by_store_item_dayofweek(df, end, aggregates='mean', value='sales'):
    
    print('Adding sales by store item dayofweek')
    data = calc_stats(df,end, window=7, aggregates=aggregates,
                      groupby = ['S100','I100','month'], value=value)
    print('window 7 added')
    
    for window in  [14,28,28*2,28*3,28*6,28*12]:
        agg = calc_stats(df,end, window=window, aggregates=aggregates,
                         groupby=['S100','I100','month'], value=value )
        data = pd.merge(data,agg)
        print('window %d added'% window)
    return data

# sales_by_store_item_day
def sales_by_store_item_day(df, end, aggregates='mean', value='QTT'):
    
    print('Adding sales by store item day')
    data = calc_stats(df,end, window=365, aggregates=aggregates,
                      groupby = ['S100','I100','day'], value=value)
    print('window 365 added')
    
    return data

# Sales by item
def sales_by_item(df, end, aggregates='mean', value='QTT'):
    
    print('Adding sales by item ')
    data = calc_stats(df,end, window=7, aggregates=aggregates,
                      groupby = ['I100'], value=value)
    print('window 7 added')
    
    for window in  [14,28,28*2]:
        agg = calc_stats(df,end, window=window, aggregates=aggregates,
                         groupby=['I100'], value=value )
        data = pd.merge(data,agg)
        print('window %d added'% window)
    return data
def calc_roll_stat(df,end,groupby=None,window=1,aggregate='mean'):
    # Rolling statistics method
    last_date = pd.to_datetime(end) - pd.Timedelta(days=7)
    first_date = pd.to_datetime(end) - pd.Timedelta(days=window)
    df1 = df[(df.DATE >= first_date) & (df.DATE <= last_date)]
    
    dfPivot = df1.set_index(['DATE']+groupby)['QTT'].unstack().unstack()
    dfPivot = dfPivot.rolling(window=window).mean().fillna(method='bfill')
    return dfPivot.stack().stack().rename(aggregate+str(window))

def calc_expand_stat(df,end,groupby=None,window=1,aggregate='mean'):
    # Expanding statistics method
    last_date = pd.to_datetime(end) - pd.Timedelta(days=7)
    first_date = pd.to_datetime(end) - pd.Timedelta(days=window)
    df1 = df[(df.DATE >= first_date) & (df.DATE <= last_date)]
    
    dfPivot = df1.set_index(['DATE']+groupby)['QTT'].unstack().unstack()
    dfPivot = dfPivot.expanding(min_periods=window).mean().fillna(method='bfill')
    dfPivot = dfPivot.stack().stack().rename(aggregate+'_'+str(window)).reset_index()
    return dfPivot

def create_data1(sales,test,date):
    
    # Date input
    for i in range(2):
        end = pd.to_datetime(date) - pd.Timedelta(days=7*(i+1))
        print(end)
    
        # Rolling feature
        for aggregates in ['mean','min','max','sum','std']:

            # store/item
            print('-'*20+'Aggregate by '+aggregates+'-'*20)
            data = sales_by_store_item(sales,end, aggregates=aggregates,value='sales')
            sales = pd.merge(sales,data,on=['S100','I100'],how='left')
            test = pd.merge(test,data,on=['S100','I100'], how='left')

            # store/item/dayofweek
            df = sales_by_store_item_dayofweek(sales,end, aggregates=aggregates,value='sales')
            #data = pd.merge(data,df,)
            sales = pd.merge(sales,df,on=['S100','I100','month'],how='left')
            test = pd.merge(test,df,on=['S100','I100','month'], how='left')

            # store/item/day
            df = sales_by_store_item_day(sales,end, aggregates=aggregates,value='sales')
            #data = pd.merge(data,df)
            sales = pd.merge(sales,df,on=['S100','I100','day'],how='left')
            test = pd.merge(test,df,on=['S100','I100','day'], how='left')

            # sales/item
            df = sales_by_item(sales,end, aggregates=aggregates, value='sales')
            data = pd.merge(data,df)
            #data = pd.merge(sales,data)
            sales = pd.merge(sales,df, on=['I100'],how='left')
            test = pd.merge(test,df, on=['I100'], how='left')

    return sales,test


def agg_cnt_col(df, merging_cols, new_col,aggregation):
    temp = df.groupby(merging_cols).agg(aggregation).reset_index()
    temp.columns = merging_cols + [new_col]
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

def new_item_sales(df, merging_cols, new_col):
    temp = (
        df
        .query('item_age==0')
        .groupby(merging_cols)['QTT']
        .mean()
        .reset_index()
        .rename(columns={'QTT': new_col})
    )
    df = pd.merge(df, temp, on=merging_cols, how='left')
    return df

def lag_feature(df, lag, col, merge_cols):        
    temp = df[merge_cols + [col]]
    temp = temp.groupby(merge_cols).agg({f'{col}':'first'}).reset_index()
    temp.columns = merge_cols + [f'{col}_lag{lag}']
    temp['date_block_num'] += lag
    df = pd.merge(df, temp, on=merge_cols, how='left')
    df[f'{col}_lag{lag}'] = df[f'{col}_lag{lag}'].fillna(0).astype('float32')
    return df

def past_information(df, merging_cols, new_col, aggregation):
    temp = []
    for i in range(24205,24262+1):
        block = df.query(f'date_block_num < {i}').groupby(merging_cols).agg(aggregation).reset_index()
        block.columns = merging_cols + [new_col]
        block['date_block_num'] = i
        block = block[block[new_col]>0]
        temp.append(block)
    temp = pd.concat(temp)
    df = pd.merge(df, temp, on=['date_block_num']+merging_cols, how='left')
    return dfgro