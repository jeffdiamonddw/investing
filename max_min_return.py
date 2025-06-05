import pandas as pd
import numpy as np
import pyomo.environ as pyo
import itertools
import time

from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, value












def array_to_dict(X):
    return {tup: float(X[tup]) for tup in itertools.product(*[range(d) for d in X.shape])}


def max_min_strategy(df_train, df_validate, params):

    R = df_train.values
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
    val = np.dot(x,df_val.values).min()
    x_bench = np.zeros(100)
    x_bench[:20] = .05
    val_bench = np.dot(x_bench, df_val.values).min()
    

    df_out = pd.DataFrame(
        {
            'solution': [list(x)],
            'symbols': [list(df_train.index)],
            'train_return': [obj],
            'val_return': [val],
            'bench_val_return': [val_bench]
        }
    )
    return df_out



df_train = pd.read_parquet('test_return.parquet')
df_val = pd.read_parquet('test_return_val.parquet')
params = {
    'max_frac': .05,
    'max_num_trades': 50
}
df = max_min_strategy(df_train, df_val, params)
print(df)



