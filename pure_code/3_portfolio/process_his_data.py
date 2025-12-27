

import numpy as np
import pandas as pd


def cal_statistics(data):

    statistics = {}
    if data.ndim == 1:
        statistics['mean'] = np.mean(data)
        statistics['cov'] = np.array([[0]])
    else:
        statistics['mean'] = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)


        eigvals, eigvecs = np.linalg.eigh(cov)
        if np.all(eigvals > 0):
            statistics['cov'] = cov
        else:
            eigvals[eigvals < 0] = 1e-10

            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            statistics['cov'] = cov
    return statistics



def weight_statistics(stat1, stat2, weight1, weight2):

    total_weight = weight1 + weight2
    weighted_mean = (weight1 * stat1['mean'] + weight2 * stat2['mean']) / total_weight
    weighted_cov = (weight1 * stat1['cov'] + weight2 * stat2['cov']) / total_weight
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    if np.any(eigvals <= 0):
        eigvals[eigvals <= 0] = 1e-10
        weighted_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return {'mean': weighted_mean, 'cov': weighted_cov}




def daily_stat_translate_to_annual(return_statistics, annual_riskfree_rate, trading_days_per_year):

    annual_return_statistics = {}

    daily_riskfree_rate = annual_riskfree_rate / trading_days_per_year
    daily_riskfree_return = np.exp(daily_riskfree_rate)
    annual_riskfree_return = np.exp(annual_riskfree_rate)

    daily_excess_return = return_statistics['mean'] - daily_riskfree_return


    annual_excess_return = trading_days_per_year * daily_excess_return
    annual_return_statistics['mean'] = annual_excess_return + annual_riskfree_return
    annual_return_statistics['cov'] = trading_days_per_year * return_statistics['cov']
    return annual_return_statistics



def daily_return_translate_to_annual(daily_return_scenarios, annual_riskfree_rate, trading_days_per_year):

    daily_riskfree_rate = annual_riskfree_rate / trading_days_per_year
    daily_riskfree_return = np.exp(daily_riskfree_rate)
    annual_riskfree_return = np.exp(annual_riskfree_rate)


    daily_excess_return_scenarios = daily_return_scenarios - daily_riskfree_return


    annual_excess_return_scenarios = trading_days_per_year * daily_excess_return_scenarios
    annual_return_scenarios = annual_excess_return_scenarios + annual_riskfree_return

    return annual_return_scenarios, annual_riskfree_return


