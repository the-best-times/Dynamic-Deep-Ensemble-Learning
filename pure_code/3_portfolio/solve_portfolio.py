import numpy as np
import gurobipy as gbp
from gurobipy import GRB




def create_decision_variables(model, asset_num, x0, lending_ratio, cons_type):

    lb_ = -lending_ratio * x0
    uf = model.addVar(ub=np.inf, lb=lb_, vtype=gbp.GRB.CONTINUOUS, name='uf')


    if cons_type == 'biased_long':
        u = model.addMVar(asset_num, ub=np.inf, lb=-np.inf, vtype=gbp.GRB.CONTINUOUS, name='u')
    elif cons_type in ['non_short', 'negative_screen']:
        u = model.addMVar(asset_num, ub=np.inf, lb=0, vtype=gbp.GRB.CONTINUOUS, name='u')
    else:
        raise ValueError('Invalid cons_type provided.')
    return u, uf



def add_common_constraints(model, u, uf, asset_num, x0, expected_score, target_score, cons_type, ESG_cons, M=None):

    model.addConstr(gbp.quicksum(u[i] for i in range(asset_num)) + uf == x0, name='self_financing_constraint')

    if cons_type == 'biased_long':
        model.addConstr(gbp.quicksum(u[i] for i in range(asset_num)) >= 1e-5, name='biased_long_constraint')
    elif cons_type == 'negative_screen' and M is not None:
        for i in range(asset_num):
            model.addConstr(M[i, i] * u[i] >= 0, name=f'negative_screen_constraint_{i}')


    if ESG_cons:
        bar_score = (1 / x0) * gbp.quicksum(expected_score[i] * u[i] for i in range(asset_num))
        model.addConstr(bar_score >= target_score, name='ESG_score_constraint')

    return model



def get_common_investment_params(investment_preference_param_dict):

    x0 = investment_preference_param_dict['x0']
    cons_type = investment_preference_param_dict['cons_type']
    M = investment_preference_param_dict['M']
    ESG_cons = investment_preference_param_dict['ESG_cons']
    lending_ratio = investment_preference_param_dict['lending_ratio']


    return x0, cons_type, ESG_cons, M, lending_ratio




def solve_native(investment_preference_param_dict, asset_num):

    x0, _, _, _, lending_ratio = get_common_investment_params(investment_preference_param_dict)
    total_asset_num = 1 + asset_num
    wealth = (1+ lending_ratio) * x0
    u = (wealth / total_asset_num) * np.ones(asset_num)
    uf = x0 - np.sum(u)
    return u, uf



def solve_risk_parity(investment_preference_param_dict, asset_num, cov, expected_score):

    x0, cons_type, ESG_cons, M, lending_ratio = get_common_investment_params(investment_preference_param_dict)
    target_score = investment_preference_param_dict['target_score']

    m = gbp.Model()
    m.setParam('NonConvex', 2)


    u, uf = create_decision_variables(m, asset_num, x0, lending_ratio, cons_type)


    variance = gbp.quicksum(u[i] * cov[i, j] * u[j] for i in range(asset_num) for j in range(asset_num))


    std = m.addVar(ub=np.inf, lb=0, vtype=gbp.GRB.CONTINUOUS, name='std')
    inv_std = m.addVar(ub=np.inf, lb=0, vtype=gbp.GRB.CONTINUOUS, name='1/std')
    m.addConstr(std * std == variance, name='std_constraint')
    m.addConstr(std * inv_std == 1, name='inv_std_constraint')


    MRC = m.addMVar(asset_num, ub=np.inf, lb=-np.inf, vtype=gbp.GRB.CONTINUOUS, name='MRC')
    for i in range(asset_num):
        m.addConstr(MRC[i] == inv_std * gbp.quicksum(cov[i, j] * u[j] for j in range(asset_num)), name=f'MRC_constraint_{i}')


    TRC = m.addMVar(asset_num, ub=np.inf, lb=-np.inf, vtype=gbp.GRB.CONTINUOUS, name='TRC')
    for i in range(asset_num):
        m.addConstr(TRC[i] == u[i] * MRC[i], name=f'TRC_constraint_{i}')


    obj = gbp.quicksum((TRC[i] - TRC[j]) ** 2 for i in range(asset_num) for j in range(asset_num))
    m.setObjective(obj, gbp.GRB.MINIMIZE)

    m = add_common_constraints(m, u, uf, asset_num, x0, expected_score, target_score, cons_type, ESG_cons, M)
    m.optimize()

    return u.x, uf.x, m.objVal




def solve_mv(investment_preference_param_dict, asset_num, expected_return, cov, expected_score, riskfree_return):

    x0, cons_type, ESG_cons, M, lending_ratio = get_common_investment_params(investment_preference_param_dict)
    target_score = investment_preference_param_dict['target_score']
    target_return = investment_preference_param_dict['target_return']

    m = gbp.Model()


    u, uf = create_decision_variables(m, asset_num, x0, lending_ratio, cons_type)


    obj = gbp.quicksum(u[i] * cov[i, j] * u[j] for i in range(asset_num) for j in range(asset_num))
    m.setObjective(obj, gbp.GRB.MINIMIZE)


    m.addConstr(gbp.quicksum(expected_return[i] * u[i] for i in range(asset_num)) + riskfree_return * uf >= target_return, name='expected_return_constraint')


    m = add_common_constraints(m, u, uf, asset_num, x0, expected_score, target_score, cons_type, ESG_cons, M)


    m.optimize()

    return u.x, uf.x, m.objVal












