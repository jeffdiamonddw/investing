import json
import itertools
import multiprocessing






import numpy as np
import pandas as pd

import pyomo.environ as pyo

from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, value




def myadfuller(x):
    try:
        result = adfuller(x)[1]
    except:
        result = None
    return result


def interp_nans(_x):
    x = np.array(_x)
    if np.isnan(x).all():
        return x
    nans = np.isnan(x)
    x[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), x[~nans])
    return x

def interp_inf_nan(x):
    x = np.array(x)
    x[np.isinf(x)] = np.nan
    return np.apply_along_axis(interp_nans, arr = x, axis = 0) 


def get_train_validate(df, params, transaction_date):
    
    df_train = df.loc[
        (df.sell_date >= transaction_date  - pd.Timedelta(weeks = 52 * params['history_years']))\
        & (df.sell_date <= transaction_date)
    ]
    
    df_validate = df.loc[
        (df.sell_date >= transaction_date + pd.Timedelta(weeks = 52 * params['hold_years']))\
        & (df.sell_date <= transaction_date + pd.Timedelta(weeks = 52 * params['hold_years']) + pd.Timedelta(days = 7))\
    ]

    return df_train, df_validate


def top_min_return_strategy(df_train, df_validate, params):

    
    
    #Get the minimum return by stock
    df_mix = df_train[['symbol', 'return']].groupby('symbol').agg(lambda x: np.percentile(x, 5)).sort_values(by = 'return', ascending = False).rename(columns = {'return': 'min_return'})
    
    
    #Restrict training data to the top min-return stocks
    symbols = df_mix.index[:params['num_stocks']]
    symbols = list(symbols.intersection(df_validate.symbol))

    
    #Get the validation matrix as the return on the first sell date past the specified sell date (hold_years hears after the buy date)
    df_validate_keep = df_validate.set_index('symbol').loc[symbols]

    df_train = df_train.set_index('symbol').loc[symbols].reset_index()
    df_ret = pd.pivot_table(df_train, index = 'symbol', columns = 'sell_date', values = 'return')
    df_ret.to_parquet('test_return.parquet')

    df_ret_val = pd.pivot_table(df_validate_keep, index = 'symbol', columns = 'sell_date', values = 'return')
    df_ret_val.to_parquet('test_return_val.parquet')
    
    
    my_return = df_validate_keep['return'].mean()
    df_result = pd.DataFrame({'symbol': [list(symbols)], 'return': [my_return] })
    df_result.loc[:, 'hold_years'] = params['hold_years']
    return df_result


def array_to_dict(X):
    return {tup: float(X[tup]) for tup in itertools.product(*[range(d) for d in X.shape])}


def max_min_strategy(df_train, df_validate, params):

    


    #Get the minimum return by stock
    df_mix = df_train[['symbol', 'return']].groupby('symbol').agg(lambda x: np.percentile(x, 5)).sort_values(by = 'return', ascending = False).rename(columns = {'return': 'min_return'})
    
    
    #Restrict training data to the top min-return stocks
    symbols = df_mix.index[:params['num_stocks']]
    symbols = list(symbols.intersection(df_validate.symbol))



    
    #Get the validation matrix as the return on the first sell date past the specified sell date (hold_years hears after the buy date)
    df_train = df_train.set_index('symbol').loc[symbols]
    df_validate = df_validate.set_index('symbol').loc[symbols]
    
    R = pd.pivot_table(df_train, index = 'symbol', columns = 'sell_date', values = 'return').values
    R_val = pd.pivot_table(df_validate, index = 'symbol', columns = 'sell_date', values = 'return').values
    
  

    m = params['max_frac']
    S = params['max_num_trades']
    
    model = pyo.ConcreteModel()

    model.num_stocks = pyo.Param(within=pyo.NonNegativeIntegers, initialize = R.shape[0])
    model.num_periods = pyo.Param(within=pyo.NonNegativeIntegers, initialize = R.shape[1])


    model.s= pyo.RangeSet(0, model.num_stocks - 1)
    model.p = pyo.RangeSet(0, model.num_periods - 1)



    model.R = pyo.Param(model.s, model.p, initialize = array_to_dict(R))
    model.S = pyo.Param(initialize = S)
    model.m = pyo.Param(initialize = m)

    model.x = pyo.Var(model.s, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.s, domain = pyo.Binary)
    model.r = pyo.Var(model.p, domain = pyo.NonNegativeReals)
    model.mu = pyo.Var(domain = pyo.Reals)

    def return_constraint(model, p):
        return sum(model.R[s,p] * model.x[s] for  s in model.s) == model.r[p]
    model.return_constraint = pyo.Constraint(model.p, rule = return_constraint)


    def min_constraint(model, p):
        return model.r[p] >= model.mu
    model.min_constraint = pyo.Constraint(model.p, rule = min_constraint)


    def turn_on_constraint(model, s):
        return model.x[s] <= model.y[s]
    model.turn_on_constraint = pyo.Constraint(model.s, rule = turn_on_constraint)

    def max_num_on_constraint(model):
        return sum(model.y[s] for s in model.s) <= model.S
    model.max_num_on_constraint = pyo.Constraint(rule = max_num_on_constraint)

    def normalize_constraint(model):
        return sum(model.x[s] for s in model.s) == 1
    model.normalize_constraint = pyo.Constraint(rule = normalize_constraint)

    def max_frac_constraint(model, s):
        return model.x[s] <= model.m
    model.max_frac_constraint = pyo.Constraint(model.s, rule = max_frac_constraint)

    def obj_expression(model):
        return model.mu
    model.OBJ = pyo.Objective(rule=obj_expression, sense = pyo.maximize)

    opt = pyo.SolverFactory("cbc")
   

    results = opt.solve(model, tee = False)
    x = np.array(list(model.x.get_values().values()))
    obj = value(model.OBJ)
    val = float(np.dot(x,R_val)[0])
    x_bench = np.zeros(100)
    x_bench[:20] = .05
    val_bench = float(np.dot(x_bench, R_val)[0])

    keep = x>0
    _solution = sorted(
        list(
            zip(
                list(x[keep]), 
                list(np.array(symbols)[keep])
            )
        ),
        reverse = True
    )
    solution = [(str(s), float(x)) for x,s in _solution]
    
    df_out = pd.DataFrame(
        {
            'solution': [solution],
            'train_return': [obj],
            'val_return': [val],
            'bench_val_return': [val_bench]
        }
    )
    return df_out




def max_min_return_strategy(df_train, df_validate, params):

    
    
    #Get the minimum return by stock
    df_mix = df_train[['symbol', 'return']].groupby('symbol').agg(lambda x: np.percentile(x, 5)).sort_values(by = 'return', ascending = False).rename(columns = {'return': 'min_return'})
    
    
    #Restrict training data to the top min-return stocks
    symbols = df_mix.index[:params['num_stocks']]
    symbols = list(symbols.intersection(df_validate.symbol))



    
    #Get the validation matrix as the return on the first sell date past the specified sell date (hold_years hears after the buy date)
    df_validate_keep = df_validate.set_index('symbol').loc[symbols]
    
    my_return = df_validate_keep['return'].mean()
    df_result = pd.DataFrame({'symbol': [list(symbols)], 'return': [my_return] })
    df_result.loc[:, 'transaction_date'] = transaction_date
    df_result.loc[:, 'hold_years'] = params['hold_years']
    return df_result



def apply_strategy_to_transaction_date(tup):
    strategy, df_return, params, transaction_date = tup
    df = df_return.loc[
            (df_return.min_sell_date <= transaction_date - pd.Timedelta(weeks = 52 * (params['history_years'])))\
            & (df_return.max_sell_date >= transaction_date + pd.Timedelta(weeks = 52 * params['hold_years']))
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
    print(df_result)
    return df_result


def apply_strategy(df_return, strategy, params):
    
    earliest_transaction_date = pd.to_datetime('2008-04-01') + pd.Timedelta(weeks = 52 * params['history_years'])
   
    _df_ret = df_return.loc[df_return.hold_years == params['hold_years']]

    df_max_sell_date = _df_ret[['symbol', 'sell_date']].groupby('symbol').agg('max').rename(columns={'sell_date': 'max_sell_date'})
    df_min_sell_date = _df_ret[['symbol', 'sell_date']].groupby('symbol').agg('min').rename(columns={'sell_date': 'min_sell_date'})
    df_ret = _df_ret.set_index('symbol').join(df_max_sell_date).join(df_min_sell_date).reset_index()


    transaction_dates = pd.date_range(start = earliest_transaction_date, end = params['latest_transaction_date'], freq = 'W')

    workload = [(strategy, df_ret, params, transaction_date) for transaction_date in transaction_dates]
    #workload = workload[:10]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    # results = []
    # for tup in workload:
    #     result = apply_strategy_to_transaction_date(tup)
    #     print(result)
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
    
   
    strategy = max_min_strategy
    params = {
        'num_stocks': 100,
        'max_num_trades': 50,
        'max_frac': .05, 
        'max_addfuller': .05,
        'hold_years': 3,
        'history_years': 5,
        'min_price': 100,
        'max_price': 10000,
        'latest_transaction_date' : 'Jan 1, 2021'
    }
    

    #Read in return data
    df_return = pd.read_parquet('data/returns_2.parquet')
    df_result = apply_strategy(df_return, strategy, params)

    df_result.to_parquet('results_1.parquet')
 
        
   
    




