import json
import itertools
import multiprocessing
import time






import numpy as np
import pandas as pd

import pyomo.environ as pyo

from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, value





def get_train_validate(df, params, transaction_date):
    
    df_train = df.loc[
        (df.sell_date >= transaction_date  - pd.Timedelta(weeks = 52 * params['history_years']))\
        & (df.sell_date <= transaction_date)\
        & (df.hold_weeks == params['train_hold_weeks'])
    ]
    
    df_validate = df.loc[
        (df.sell_date >= transaction_date + pd.Timedelta(weeks = params['val_hold_weeks']))\
        & (df.sell_date <= transaction_date + pd.Timedelta(weeks =  params['val_hold_weeks']) + pd.Timedelta(days = 7))\
         & (df.hold_weeks == params['val_hold_weeks'])
    ]

    return df_train, df_validate


def top_min_return_strategy(df_train, df_validate, params):

    
    
    #Get the minimum return by stock
    df_mix = df_train[['symbol', 'annual_return']].groupby('symbol').agg(lambda x: np.percentile(x, 5)).sort_values(by = 'annual_return', ascending = False).rename(columns = {'return': 'min_return'})
    
    
    #Restrict training data to the top min-return stocks
    symbols = df_mix.index[:params['num_stocks']]
    symbols = list(symbols.intersection(df_validate.symbol))

    
    #Get the validation matrix as the return on the first sell date past the specified sell date (hold_years hears after the buy date)
    df_validate = df_validate.set_index('symbol').loc[symbols]
    buy_price = df_validate.buy_price.mean()
    sell_price = df_validate.sell_price.mean()
    period_return = sell_price/buy_price - 1
    
    

    
    
   
    df_result = pd.DataFrame({
        'symbol': [list(symbols)], 
        'period_return': [period_return], 
        'buy_price': [buy_price], 
        'sell_price': [sell_price], 
        'buy_prices': [list(df_validate.buy_price.values)],  
        'sell_prices': [list(df_validate.sell_price.values)]
    })
    df_result.loc[:, 'hold_weeks'] = params['val_hold_weeks']
    return df_result





def apply_strategy_to_transaction_date(tup):
    try:
        strategy, df_return, params, transaction_date = tup
    
        df = df_return.loc[
            (df_return.min_sell_date <= transaction_date - pd.Timedelta(weeks = 52 * (params['history_years'])))
            &  (df_return.max_sell_date >= transaction_date + pd.Timedelta(weeks =  params['val_hold_weeks']))
        ]
        
            
        #restrict to min and max current (transaction day) price
        df_transaction_day_price = df.sort_values(by = ['symbol', 'sell_date'])[['symbol', 'sell_price']]\
            .groupby('symbol')\
            .agg('last')\
            .rename(columns = {'sell_price': 'transaction_day_price'})
        df = df.set_index('symbol').join(df_transaction_day_price).reset_index()
        df = df.loc[
            (df.transaction_day_price >= params['min_price'])\
            & (df.transaction_day_price <= params['max_price'])
        ]
        df_train, df_validate = get_train_validate(df, params, transaction_date)
        if df_train.shape[0] < 100:
            print('not enough data for {}'.format(transaction_date))
            return None
        df_result = strategy(df_train, df_validate, params)
        df_result.loc[:, 'transaction_date'] = transaction_date
        print(df_result.transaction_date.iloc[0], flush = True)
        return df_result
    except:
        return None


def apply_strategy(df_return, strategy, params):
    
    earliest_transaction_date = pd.to_datetime('2008-04-01') + pd.Timedelta(weeks = 52 * params['history_years'])




    transaction_dates = pd.date_range(start = earliest_transaction_date, end = params['latest_transaction_date'], freq = 'W')

    workload = [(strategy, df_return, params, transaction_date) for transaction_date in transaction_dates]
 
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    # results = []
    # for tup in workload:
    #     t1 = time.time()
    #     result = apply_strategy_to_transaction_date(tup)
    #     print(time.time() - t1)
    #     if result is not None:
    #         results += [result]
        
    results = pool.map(apply_strategy_to_transaction_date, workload)

    DF_result = pd.DataFrame()
    for df_result in results:
        if df_result is not None:
            DF_result = pd.concat([DF_result, df_result])
    for key, val in params.items():
        DF_result.loc[:, key] = val

    return DF_result



if __name__ == "__main__":  
    
    #Read in return data
    df_return = pd.read_parquet('s3://jdinvestment/returns_all.parquet')
   
    strategy = top_min_return_strategy
    DF_result = pd.DataFrame()
    for history_years in [5, 7, 10]:
        for val_hold_weeks in [4, 52, 156]:
            print("running history_years = {}".format(history_years), flush = True)
            params = {
                'num_stocks': 20,
                'train_hold_weeks': 156,
                'val_hold_weeks': val_hold_weeks,
                'history_years': history_years,
                'min_price': 100,
                'max_price': 10000,
                'latest_transaction_date' : 'Dec 31, 2021'
            }
            
    
            
        
            
            df_result = apply_strategy(df_return, strategy, params)
            DF_result = pd.concat([DF_result, df_result])
    
            df_result.to_parquet('s3://jdinvestment/top_min_results_{}_{}.parquet'.format(history_years, val_hold_weeks))
    DF_result.to_parquet('s3://jdinvestment/top_min_results_1.parquet)
