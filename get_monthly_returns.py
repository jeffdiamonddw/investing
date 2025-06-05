import numpy as np
import pandas as pd
import scipy.interpolate
import itertools
import multiprocessing




def get_return(week_close, hold_weeks):
    if hold_weeks > len(week_close):
        return []
    
    dates = np.array(list(week_close))[:, 0]
    min_date = dates.min()
    days = np.array([x.days for x in dates - np.array(min_date)])
    close = np.array(list(week_close))[:, 1]
    interp = scipy.interpolate.interp1d(days, close, kind = 'linear', fill_value = 'extrapolate')
    
    first_day = days.min() + (   (4 - dates.min().dayofweek) % 7  ) 
    all_days = np.arange(first_day, max(days), 7)
    all_close = interp(all_days)

    
    all_dates = pd.to_datetime([min_date + pd.Timedelta(days = d) for d in all_days])

    
    df_price = pd.DataFrame({'date': all_dates, 'close': all_close}).set_index('date')
    
    sell_dates = pd.to_datetime(pd.date_range(start = all_dates.min() + pd.Timedelta(weeks = hold_weeks), end = max(all_dates), freq = '7d'))
    buy_dates = pd.to_datetime(pd.date_range(start = all_dates.min() , end = max(all_dates) - pd.Timedelta(weeks = hold_weeks), freq = '7d'))
    ratios = (df_price.loc[sell_dates].values/df_price.loc[buy_dates]).values.flatten()
    sell_prices = [float(x) for x in df_price.loc[sell_dates].values.flatten()]
    buy_prices = [float(x) for x in df_price.loc[buy_dates].values.flatten()]
    annual_return = [float(x) for x in ratios**(1/(hold_weeks/52)) - 1]

    return list(zip(list(buy_dates.values), list(sell_dates.values), list(buy_prices), list(sell_prices), list(annual_return)))


def get_symbol_returns(tup):
    try:
        df_week, symbol, hold_weeks = tup
        df = df_week.loc[df_week.symbol == symbol]
        week_close = df.price_tup.values
        result = get_return(week_close, hold_weeks)
        print("done (symbol, hold_weeks) = ({},{})".format(symbol, hold_weeks), flush = True)
        return [(symbol, hold_weeks) + rslt for rslt in result] 
    except:
        return None

def get_returns(df_weekly, hold_weeks_list = [4]):
   
    
    df_week = df_weekly.copy()
    df_week.loc[:, 'price_tup'] = list(zip(df_week.date, df_week.close))

    workload = [(df_week, symbol, hold_weeks) for symbol, hold_weeks in itertools.product(list(df_week.symbol.drop_duplicates().values), hold_weeks_list)]
    
    # results= []
    # for tup in workload[:10]:
    #     result = get_symbol_returns(tup)
    #     results += result

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    _results = pool.map(get_symbol_returns, workload)

    results = []
    for rslt in _results:
        if rslt is not None:
            results += rslt


    df_return = pd.DataFrame(results, columns = ['symbol', 'hold_weeks', 'buy_date', 'sell_date', 'buy_price', 'sell_price', 'annual_return' ])

    return df_return


if __name__ == '__main__':

    df_weekly = pd.read_parquet('s3://jdinvestment/weekly_data_copy.parquet')

    
    df_return = get_returns(df_weekly)
    df_return.to_parquet('s3://jdinvestment/returns_4_52.parquet', index = False)
