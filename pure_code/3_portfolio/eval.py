import numpy as np
import pandas as pd

def evaluation_wealth(wealths, score, rf, x0, T):

    mean = np.mean(wealths)
    var = np.var(wealths)
    std = np.std(wealths)

    semi_var = np.var(wealths[wealths < mean])
    semi_std = np.std(wealths[wealths < mean])

    mean_std = mean / std
    mean_semi_std = mean / semi_std

    cvar_95 = np.mean(np.sort(wealths)[0:int(0.05 * len(wealths))])
    cvar_90 = np.mean(np.sort(wealths)[0:int(0.1 * len(wealths))])
    cvar_5 = -np.mean(np.sort(wealths)[0:int(0.05 * len(wealths))])
    cvar_10 = -np.mean(np.sort(wealths)[0:int(0.1 * len(wealths))])

    mean_cvar_5 = (mean - x0 * rf ** T) / (x0 * rf ** T - cvar_95)
    mean_cvar_10 = (mean - x0 * rf ** T) / (x0 * rf ** T - cvar_90)

    var_95 = np.percentile(wealths, 5)
    var_90 = np.percentile(wealths, 10)
    var_5 = -np.percentile(wealths, 5)
    var_10 = -np.percentile(wealths, 10)

    sharpe_ratio = (mean - x0 * rf ** T) / std

    return_riskfree =  x0 * rf ** T
    semi_var_rf = np.var(wealths[wealths < return_riskfree])
    sortino_ratio = (mean - x0 * rf ** T) / np.sqrt(semi_var_rf)


    rachev_ratio_5 = np.mean(np.sort(x0 *rf - wealths)[0:int(0.05 * len(wealths))]) / np.mean(np.sort(wealths -x0 * rf)[0:int(0.05 * len(wealths))])
    rachev_ratio_10 = np.mean(np.sort(x0 *rf - wealths)[0:int(0.1 * len(wealths))]) / np.mean(np.sort(wealths - x0 *rf)[0:int(0.1 * len(wealths))])

    skewness = np.mean((wealths - mean) ** 3) / std ** 3
    kurtosis = np.mean((wealths - mean) ** 4) / std ** 4 - 3


    wealth_series = pd.Series(wealths)
    cumulative_max = wealth_series.cummax()
    drawdown = (cumulative_max - wealth_series) / cumulative_max
    maximum_drawdown = drawdown.max()


    mean_score = np.mean(score)
    Delta_ratio_score = mean_score / std

    scorino_ratio_score = mean_score / np.sqrt(semi_var)



    eval_statistics = {
        'mean': round(mean, 4),
        'var': round(var, 4),
        'std': round(std, 4),
        'semi_var': round(semi_var, 4),
        'semi_std': round(semi_std, 4),
        'mean/std': round(mean_std, 4),
        'mean/semi_std': round(mean_semi_std, 4),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'Sortino_ratio': round(sortino_ratio, 4),
        'CVaR_95%': round(cvar_95, 4),
        'CVaR_90%': round(cvar_90, 4),
        'CVaR_5%': round(cvar_5, 4),
        'CVaR_10%': round(cvar_10, 4),
        'VaR_95%': round(var_95, 4),
        'VaR_90%': round(var_90, 4),
        'VaR_5%': round(var_5, 4),
        'VaR_10%': round(var_10, 4),
        'mean/cvar_5(STARR_5)': round(mean_cvar_5, 4),
        'mean/cvar_10(STARR_5)': round(mean_cvar_10, 4),
        'Rachev_Ratio_5': round(rachev_ratio_5, 4),
        'Rachev_Ratio_10': round(rachev_ratio_10, 4),
        'skewness': round(skewness, 4),
        'kurtosis': round(kurtosis, 4),
        'maximum_drawdown': round(maximum_drawdown, 4),
        'mean_score': round(mean_score, 4),
        'Delta_ratio_score': round(Delta_ratio_score, 4),
        'scorino_ratio_score': round(scorino_ratio_score, 4)
    }
    statistics_df = pd.DataFrame(eval_statistics, index=[0])

    return eval_statistics




















