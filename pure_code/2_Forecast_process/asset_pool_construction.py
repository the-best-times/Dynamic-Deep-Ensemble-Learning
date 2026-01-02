
import sys
import numpy as np
import pandas as pd
import random
import os
import re
import gc
import shutil
import time
import matplotlib.pyplot as plt
import gurobipy as gbp
import tensorflow as tf
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from datetime import datetime
from openpyxl import Workbook, load_workbook
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from scipy.stats import gaussian_kde
from keras.losses import Huber




from process_sample_data import create_predict_param_dict, load_historical_data
from solve_ML import get_predict_data
from solve_big_many_models import random_select_models, remove_model, add_model_update_probability

sys.path.append(os.path.abspath('../3_portfolio'))
from eval import evaluation_wealth
from solve_portfolio import solve_risk_parity, solve_mv, solve_native
from process_his_data import cal_statistics, weight_statistics, daily_return_translate_to_annual, daily_stat_translate_to_annual






def select_asset(basis_param_dict, prediction_model_list, code_list, cal_forward_info, code_split_node_list):


    def get_next_iteration_folder(saved_dir):

        existing_folders = [f for f in os.listdir(saved_dir) if os.path.isdir(os.path.join(saved_dir, f))]
        pattern = r'cal_num_(\d+)'
        existing_iterations = []
        for folder in existing_folders:
            match = re.search(pattern, folder)
            if match:
                existing_iterations.append(int(match.group(1)))
        if not existing_iterations:
            return os.path.join(saved_dir, 'cal_num_1')


        next_iteration = max(existing_iterations) + 1
        return os.path.join(saved_dir, f'cal_num_{next_iteration}')

    path_basis_dict = basis_param_dict['path_basis_dict']
    results_saved_folder_path = path_basis_dict['results_saved_folder_path_pool']

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    weight_historical = structural_basis_param_dict['weight_historical']



    total_model_num = structural_basis_param_dict['total_model_num']
    remove_model_prob = structural_basis_param_dict['remove_model_prob']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    total_periods_num = len(date_split_node_list) - 1
    weight_historical = structural_basis_param_dict['weight_historical']
    target_score = structural_basis_param_dict['investment_preference_param']['target_score']
    


    all_forward_data_dict_dict = {}
    all_real_observed_data_dict_dict = {}

    cal_num = len(code_split_node_list)
    for cal_count in range(cal_num):
        if cal_count < len(code_split_node_list) - 1:
            code_list_ = code_list[code_split_node_list[cal_count]:code_split_node_list[cal_count + 1]]
        else:
            code_list_ = code_list[code_split_node_list[cal_count]:]

        forward_data_dict_dict, real_observed_data_dict_dict = get_forward_looking_info_result(basis_param_dict, prediction_model_list, code_list_, cal_forward_info, cal_count)
        all_forward_data_dict_dict.update(forward_data_dict_dict)
        all_real_observed_data_dict_dict.update(real_observed_data_dict_dict)

    performance_list = get_performance_ratios_results(basis_param_dict, code_list, all_forward_data_dict_dict)
    print('performance_list', performance_list)

    if len(performance_list) != len(code_list):
        raise ValueError('Mismatch between length of performance_list and code_list.')


    model_group = determine_model_group(prediction_model_list, basis_param_dict)

    saved_dir = os.path.join(results_saved_folder_path, 'select_assets_result', f'{model_group}',
                             f'nodes_{total_periods_num}_models_{total_model_num}_removeProb_{remove_model_prob}', f'{select_method}_{indicator}_num_{num_select}',
                             f'his_{weight_historical}_score_{target_score}')
    os.makedirs(saved_dir, exist_ok=True)
    if select_method == 'random':
        saved_dir = get_next_iteration_folder(saved_dir)
        os.makedirs(saved_dir, exist_ok=True)


    select_asset_list = process_asset_select(performance_list, code_list, asset_select_param_dict)

    all_data_dict = combine_selected_assets_info(basis_param_dict, select_asset_list, all_forward_data_dict_dict, all_real_observed_data_dict_dict)
    u = eval_selected_assets(basis_param_dict, all_data_dict, saved_dir)


    file_path = os.path.join(saved_dir, 'select_assets_lists.txt')
    with open(file_path, 'w') as file:
        for item in select_asset_list:
            file.write(f"{item}\n")
    print('The select asset code list file creation completed.')
    return select_asset_list


def get_eval_result(basis_param_dict, prediction_model_list, code_list, cal_forward_info, code_split_node_list):

    path_basis_dict = basis_param_dict['path_basis_dict']
    results_saved_folder_path = path_basis_dict['results_saved_folder_path_pool']

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    asset_select_param_dict = structural_basis_param_dict['asset_select_param_dict']
   


    all_forward_data_dict_dict = {}
    all_real_observed_data_dict_dict = {}

    cal_num = len(code_split_node_list)
    for cal_count in range(cal_num):
        if cal_count < len(code_split_node_list) - 1:
            code_list_ = code_list[code_split_node_list[cal_count]:code_split_node_list[cal_count + 1]]
        else:
            code_list_ = code_list[code_split_node_list[cal_count]:]

        forward_data_dict_dict, real_observed_data_dict_dict = get_forward_looking_info_result(basis_param_dict, prediction_model_list, code_list_, cal_forward_info, cal_count)
        all_forward_data_dict_dict.update(forward_data_dict_dict)
        all_real_observed_data_dict_dict.update(real_observed_data_dict_dict)


    avg_metrics_dict, _ = evaluate_forward_looking_info(basis_param_dict, prediction_model_list, all_forward_data_dict_dict, all_real_observed_data_dict_dict, code_list)


    return avg_metrics_dict


def eval_selected_assets(basis_param_dict, all_data_dict, saved_dir):


    def write_statistics_to_excel(writer, sheet_name, mean, cov, gap=3):

        mean_df = pd.DataFrame([mean.flatten()])
        mean_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=0)
        cov_df = pd.DataFrame(cov)
        cov_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(mean_df) + gap)


    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    weight_historical = structural_basis_param_dict['weight_historical']
    data_property_list = structural_basis_param_dict['data_property_list']
    investment_preference_param = structural_basis_param_dict['investment_preference_param']
    trading_days_per_year = structural_basis_param_dict['trading_days_per_year']
    annual_riskfree_rate = structural_basis_param_dict['annual_riskfree_rate']
    x0 = investment_preference_param['x0']


    stat_data_dict = {key: {data_property: {} for data_property in data_property_list} for key in ['forward_data', 'historical_data', 'weighted_data']}



    for key, data_dict in all_data_dict.items():
        if key == 'real_observed_data':
            daily_return = data_dict['return']
            annual_return, annual_riskfree_return = daily_return_translate_to_annual(daily_return, annual_riskfree_rate, trading_days_per_year)
            all_data_dict[key]['return'] = annual_return
        elif key in ['forward_data', 'historical_data']:
            for data_property in data_property_list:

                stat_data_dict[key][data_property] = cal_statistics(data_dict[data_property])
                if data_property == 'return':
                    return_statistics = stat_data_dict[key][data_property]
                    stat_data_dict[key][data_property] = daily_stat_translate_to_annual(return_statistics, annual_riskfree_rate, trading_days_per_year)


    weight_forward = 1 - weight_historical
    for data_property in data_property_list:
        stat_data_dict['weighted_data'][data_property] = weight_statistics(stat_data_dict['historical_data'][data_property], stat_data_dict['forward_data'][data_property],
                                                                           weight_historical, weight_forward)

    mean_return = stat_data_dict['weighted_data']['return']['mean']
    cov_return = stat_data_dict['weighted_data']['return']['cov']
    mean_score = stat_data_dict['weighted_data']['ESG']['mean']



    return_scenarios_eval = all_data_dict['real_observed_data']['return']
    ESG_scenarios_eval = all_data_dict['real_observed_data']['ESG']
    num_scenarios_eval, asset_num = return_scenarios_eval.shape



    solve_model = 'mv'
    if solve_model == 'risk_parity':
        u, uf, obj = solve_risk_parity(investment_preference_param, asset_num, cov_return, mean_score)
    elif solve_model == 'mv':
        u, uf, obj = solve_mv(investment_preference_param, asset_num, mean_return, cov_return, mean_score, annual_riskfree_return)
    elif solve_model == 'native':
        u, uf = solve_native(investment_preference_param, asset_num)
    else:
        raise ValueError(f'Unknown model type: {solve_model}.')
    print("uf", uf)
    print("u is", u)

    x_T = np.zeros(num_scenarios_eval)
    y_T = np.zeros(num_scenarios_eval)
    for i in range(num_scenarios_eval):
        x_T[i] = np.dot(return_scenarios_eval[i], u) + uf * annual_riskfree_return
        y_T[i] = np.dot(ESG_scenarios_eval[i], u) / x0

    eval_stat = evaluation_wealth(x_T, y_T, annual_riskfree_return, x0, T=1)
    print(eval_stat)


    data_file_path = os.path.join(saved_dir, 'statistics_data.xlsx')
    eval_stat_file_path = os.path.join(saved_dir, 'eval_stat.xlsx')


    with pd.ExcelWriter(data_file_path, engine='openpyxl') as writer:

        pd.DataFrame(return_scenarios_eval).to_excel(writer, sheet_name='annual_return_scenarios_eval', index=False)
        pd.DataFrame(ESG_scenarios_eval).to_excel(writer, sheet_name='annual_ESG_scenarios_eval', index=False)


        for key in stat_data_dict.keys():
            for data_property in data_property_list:
                mean = stat_data_dict[key][data_property]['mean']
                cov = stat_data_dict[key][data_property]['cov']
                sheet_name = f'{key}_{data_property}'
                write_statistics_to_excel(writer, sheet_name, mean, cov, gap=5)



    with pd.ExcelWriter(eval_stat_file_path, engine='openpyxl') as writer:
        pd.DataFrame([u.flatten()]).to_excel(writer, sheet_name='policies', index=False, header=False, startrow=0)
        pd.DataFrame([uf]).to_excel(writer, sheet_name='policies', index=False, header=False, startrow=1)


        eval_stat_df = pd.DataFrame(eval_stat.values(), index=eval_stat.keys()).T
        eval_stat_df.to_excel(writer, sheet_name='eval_stat', index=False, header=True)
    return u




def process_asset_select(performance_list, code_list, asset_select_param_dict):


    def preprocess_performance_data(performance_list, code_list, key_list):

        valid_performance_list = []
        valid_code_list = []

        for item, code in zip(performance_list, code_list):
            is_valid = True

            for key in key_list:
                ratio_value = item.get(key, np.nan)
                if np.isnan(ratio_value):
                    is_valid = False
                    break
            if is_valid:
                valid_performance_list.append(item)
                valid_code_list.append(code)
        return valid_performance_list, valid_code_list

    key_list = ['traditional_Sharpe_ratio', 'traditional_Sortino_ratio', 'ESG_adjusted_Sharpe_ratio', 'ESG_adjusted_Sortino_ratio', 'mean_return', 'mean_ESG']
    valid_performance_list, valid_code_list = preprocess_performance_data(performance_list, code_list, key_list)
    code_list = valid_code_list
    performance_list = valid_performance_list

    select_method =  asset_select_param_dict['select_method']
    num_select = asset_select_param_dict['num_select']
    indicator = asset_select_param_dict['indicator']
    if indicator == 'traditional':
        key_ =  ['traditional_Sharpe_ratio', 'traditional_Sortino_ratio']




    if select_method == 'dual_sort':
        num_select_ = 5 * num_select


        combined_list = [{**perf_dict, 'code_item': code_item} for perf_dict, code_item in zip(performance_list, code_list)]

        sorted_results = {key: sorted(combined_list, key=lambda x: x[key], reverse=True)[:num_select_] for key in key_}

        selected_code_lists = {key: [item['code_item'] for item in sorted_results[key]] for key in key_}

        select_asset_list = [code for code in selected_code_lists[key_[0]] if all(code in selected_code_lists[key] for key in key_[1:])]


        if len(select_asset_list) < num_select:
            raise ValueError(f'The size is smaller than {num_select}. Consider increasing the number of stocks selected initially.')
        select_asset_list = select_asset_list[:num_select]

        code_list_dict = {code: idx for idx, code in enumerate(code_list)}
        select_asset_list.sort(key=lambda code: code_list_dict[code])


    elif select_method == 'random':
        select_asset_list = random.sample(code_list, min(num_select, len(code_list)))
    else:
        raise ValueError("Invalid select_method. Use 'dual_sort', 'random'.")

    return select_asset_list



def get_performance_ratios_results(basis_param_dict, code_list, forward_data_dict_dict):

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    path_basis_dict = basis_param_dict['path_basis_dict']
    data_folder_path = path_basis_dict['data_folder_path']

    performance_list = []


    for code_item in tqdm(code_list, desc="Calculating Performance Ratios", unit="asset"):

        fordward_data_dict = forward_data_dict_dict[code_item]
        historical_data_dict = load_historical_data(data_folder_path, date_split_node_list, code_item)

        total_eval_dict = generation_eval_result_with_combine_info(fordward_data_dict, historical_data_dict, structural_basis_param_dict)

        performance_list.append(total_eval_dict)

    return performance_list


def get_forward_looking_info_result(basis_param_dict, prediction_model_list, code_list, cal_forward_info, cal_count):


    def load_from_excel(file_path, code_list):

        data_dict = {}
        try:
            for code_item in code_list:
                df = pd.read_excel(file_path, sheet_name=code_item)
                data_dict[code_item] = df.to_dict(orient='list')
        except Exception as e:
            print(f"An error occurred while loading the Excel file: {e}")
        return data_dict

    def save_to_excel(file_path, code_item, data_dict, mode='a'):

        try:

            if not os.path.exists(file_path):
                with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                    df = pd.DataFrame(data_dict)
                    df.to_excel(writer, sheet_name=code_item, index=False)
            else:

                with pd.ExcelWriter(file_path, mode=mode, engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df = pd.DataFrame(data_dict)
                    df.to_excel(writer, sheet_name=code_item, index=False)

        except PermissionError as e:
            print(f"Permission denied: {e}. Make sure the file is not open or locked.")
        except Exception as e:
            print(f"An error occurred while saving the Excel file: {e}")


    def save_model_info_to_txt(model_info_dict, file_path, keys_to_save):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, 'w') as file:
                for period_item, info_dict in model_info_dict.items():
                    file.write(f"{period_item}:\n")

                    if isinstance(info_dict, dict):
                        for key, item in info_dict.items():
                            if key in keys_to_save:
                                file.write(f"  {key}: {item}\n")
                    else:

                        file.write(f"  Value: {info_dict}\n")
        except Exception as e:
            print(f"An error occurred while saving model info: {e}")


    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    path_basis_dict = basis_param_dict['path_basis_dict']

    data_property_list = structural_basis_param_dict['data_property_list']

    results_saved_folder_path = path_basis_dict['results_saved_folder_path_pool']
    total_model_num = structural_basis_param_dict['total_model_num']
    remove_model_prob = structural_basis_param_dict['remove_model_prob']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    total_periods_num = len(date_split_node_list) - 1

    model_group = determine_model_group(prediction_model_list, basis_param_dict)

    saved_dir = os.path.join(results_saved_folder_path, f'forward_info_{model_group}', f'nodes_{total_periods_num}_models_{total_model_num}_removeProb_{remove_model_prob}', f'result_num_{cal_count}')
    os.makedirs(saved_dir, exist_ok=True)


    forward_info_file_path = os.path.join(saved_dir, 'forward_data_result.xlsx')
    real_observed_info_file_path = os.path.join(saved_dir, 'real_observed_data.xlsx')
    running_info_saved_dir = os.path.join(saved_dir, 'Running_info')


    if cal_forward_info:
        total_items_num = len(code_list)
        with tqdm(total=total_items_num, desc='Calculating forward-looking info', unit='asset') as pbar:
            for code_item in code_list:
                print(f'Processing {code_item}...')

                forward_data_dict = {}
                real_observed_data_dict = {}
                for data_property in data_property_list:
                    print(f'Calculating the {data_property} for {code_item}...')

                    ML_model_saved_dir = os.path.join(f'result_num_{cal_count}', code_item, data_property)
                    if os.path.exists(ML_model_saved_dir):
                        shutil.rmtree(ML_model_saved_dir)


                    last_period_model_info_dict, model_info_dict = \
                        generate_Big_many_models_predict_results(basis_param_dict, prediction_model_list, ML_model_saved_dir, running_info_saved_dir, code_item, data_property)
                    real_observed_data_dict[data_property] = last_period_model_info_dict['observed_data']
                    forward_data_dict[data_property] = generation_forward_looking_info(last_period_model_info_dict)


                    Model_info_saved = os.path.join(saved_dir, 'Model_info', code_item, f'Model_info_{data_property}.txt')
                    keys_to_save = ['model_list', 'model_age_list', 'weights_list']

                save_to_excel(forward_info_file_path, code_item, forward_data_dict, mode='a')
                save_to_excel(real_observed_info_file_path, code_item, real_observed_data_dict, mode='a')
        return None
    else:

        forward_data_dict_dict = load_from_excel(forward_info_file_path, code_list)
        real_observed_data_dict_dict = load_from_excel(real_observed_info_file_path, code_list)

    return forward_data_dict_dict, real_observed_data_dict_dict



def evaluate_forward_looking_info(basis_param_dict, prediction_model_list, forward_data_dict_dict, real_observed_data_dict_dict, code_list):


    def mean_directional_accuracy(y_true, y_pred):

        return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))

    def theils_u_statistic(y_true, y_pred):

        y_random_walk = np.roll(y_true, shift=1)
        y_random_walk[0] = y_true[0]
        numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.mean((y_true - y_random_walk) ** 2))

        if denominator == 0:
            return 0
        TU = numerator / denominator
        return TU


    def tracking_signal(y_true, y_pred):

        cfe = np.sum(y_true - y_pred)
        mad = np.mean(np.abs(y_true - y_pred))

        if mad == 0:
            return 0
        TS = cfe / mad
        return TS

    data_property_list = basis_param_dict['structural_basis_param_dict']['data_property_list']

    eval_dict_dict_dict = {}
    metric_lists = ['MSE', 'RMSE', 'RMSPE', 'MAE', 'MAPE', 'SMAPE', 'Huber_Loss', 'MDA',
                    'Theil_U', 'TS', 'R2_score', 'Max_Error', 'RMSLE', 'Explained_Variance']
    metrics_list_dict = {data_property: {metric: [] for metric in metric_lists} for data_property in data_property_list}


    for code_item in code_list:
        forward_data_dict = forward_data_dict_dict[code_item]
        real_observed_data_dict = real_observed_data_dict_dict[code_item]

        eval_dict_dict = {}

        for data_property in data_property_list:
            eval_dict = {}
            forward_data = np.array(forward_data_dict[data_property])
            real_observed_data = np.array(real_observed_data_dict[data_property])



            eval_dict['MSE'] = mean_squared_error(real_observed_data, forward_data)
            eval_dict['RMSE'] = np.sqrt(eval_dict['MSE'])
            eval_dict['R2_score'] = r2_score(real_observed_data, forward_data)
            eval_dict['MAE'] = mean_absolute_error(real_observed_data, forward_data)


            eval_dict['Max_Error'] = np.max(np.abs(real_observed_data - forward_data))
            eval_dict['RMSPE'] = np.sqrt(np.mean((real_observed_data - forward_data) ** 2) / np.mean(real_observed_data ** 2))
            eval_dict['RMSLE'] = np.sqrt(np.mean((np.log(real_observed_data + 1) - np.log(forward_data + 1)) ** 2))
            eval_dict['Explained_Variance'] = explained_variance_score(real_observed_data, forward_data)


            eval_dict['MAPE'] = np.mean(np.abs((real_observed_data - forward_data) / real_observed_data))

            eval_dict['SMAPE'] = np.mean(2.0 * np.abs(real_observed_data - forward_data) / (np.abs(real_observed_data) + np.abs(forward_data)))

            eval_dict['Theil_U'] = theils_u_statistic(real_observed_data, forward_data)
            eval_dict['TS'] = tracking_signal(real_observed_data, forward_data)
            eval_dict['MDA'] = mean_directional_accuracy(real_observed_data, forward_data)

            huber_loss_fn = Huber(delta=1.0)
            huber_loss_value = huber_loss_fn(real_observed_data, forward_data)
            eval_dict['Huber_Loss'] = huber_loss_value.numpy()

            for metric, value in eval_dict.items():
                metrics_list_dict[data_property][metric].append(value)

            eval_dict_dict[data_property] = eval_dict

        eval_dict_dict_dict[code_item] = eval_dict_dict


    avg_metrics_dict = {}
    std_metrics_dict = {}
    q25_metrics_dict = {}
    q75_metrics_dict = {}
    for data_property, metrics in metrics_list_dict.items():
        avg_metrics_dict[data_property] = {metric: np.mean(values) for metric, values in metrics.items()}
        std_metrics_dict[data_property] = {metric: np.std(values) for metric, values in metrics.items()}
        q25_metrics_dict[data_property] = {metric: np.percentile(values, 25) for metric, values in metrics.items()}
        q75_metrics_dict[data_property] = {metric: np.percentile(values, 75) for metric, values in metrics.items()}

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    path_basis_dict = basis_param_dict['path_basis_dict']

    results_saved_folder_path = path_basis_dict['results_saved_folder_path_pool']
    total_model_num = structural_basis_param_dict['total_model_num']
    remove_model_prob = structural_basis_param_dict['remove_model_prob']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    total_periods_num = len(date_split_node_list) - 1

    model_group = determine_model_group(prediction_model_list, basis_param_dict)

    saved_dir = os.path.join(results_saved_folder_path, f'forward_info_{model_group}', f'nodes_{total_periods_num}_models_{total_model_num}_removeProb_{remove_model_prob}')
    os.makedirs(saved_dir, exist_ok=True)

    eval_forward_info_file_path = os.path.join(saved_dir, f'eval_forward_info.xlsx')

    avg_metrics_df = pd.DataFrame(avg_metrics_dict).T
    std_metrics_df = pd.DataFrame(std_metrics_dict).T
    q25_metrics_df = pd.DataFrame(q25_metrics_dict).T
    q75_metrics_df = pd.DataFrame(q75_metrics_dict).T


    eval_metrics_flattened = [(code, data_property, metric, eval_dict_dict_dict[code][data_property][metric]) for code in eval_dict_dict_dict for data_property in eval_dict_dict_dict[code]
                              for metric in eval_dict_dict_dict[code][data_property]]
    eval_metrics_df = pd.DataFrame(eval_metrics_flattened, columns=['Code', 'Data_Property', 'Metric', 'Value'])


    return_metrics_df = eval_metrics_df[eval_metrics_df['Data_Property'] == data_property_list[0]].pivot(index='Code', columns='Metric', values='Value')
    return_metrics_df = return_metrics_df.reindex(columns=metric_lists)

    ESG_metrics_df = eval_metrics_df[eval_metrics_df['Data_Property'] == data_property_list[1]].pivot(index='Code', columns='Metric', values='Value')
    ESG_metrics_df = ESG_metrics_df.reindex(columns=metric_lists)


    with pd.ExcelWriter(eval_forward_info_file_path, engine='xlsxwriter') as writer:

        avg_metrics_df.to_excel(writer, sheet_name='Average Metrics', index=True)
        std_metrics_df.to_excel(writer, sheet_name='Std Deviation', index=True)
        q25_metrics_df.to_excel(writer, sheet_name='q25_metrics', index=True)
        q75_metrics_df.to_excel(writer, sheet_name='q75_metrics', index=True)
        return_metrics_df.to_excel(writer, sheet_name='Return Metrics', index=True)
        ESG_metrics_df.to_excel(writer, sheet_name='ESG Metrics', index=True)
        eval_metrics_df.to_excel(writer, sheet_name='Evaluation Metrics with code', index=False)
    return avg_metrics_dict, eval_dict_dict_dict



def generate_Big_many_models_predict_results(basis_param_dict, prediction_model_list, ML_model_saved_dir, running_info_saved_dir, code_item, data_property):

    path_basis_dict = basis_param_dict['path_basis_dict']
    results_saved_folder_path = path_basis_dict['results_saved_folder_path_pool']

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    likelihood_method = structural_basis_param_dict['likelihood_method']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    total_periods_num = len(date_split_node_list) - 1

    total_model_num = structural_basis_param_dict['total_model_num']
    remove_model_prob = structural_basis_param_dict['remove_model_prob']
    nu = structural_basis_param_dict['nu']


    initial_model_list = random_select_models(prediction_model_list, total_model_num)
    initial_model_age_list = np.zeros(total_model_num)
    initial_model_weights_list = (1 / total_model_num) * np.ones(total_model_num)

    model_info_dict = {f'periods_{t}': {} for t in range(total_periods_num)}

    model_info_dict[f'periods_{0}'] = {
        'model_list': initial_model_list.copy(),
        'model_age_list': initial_model_age_list.copy(),
        'weights_list': initial_model_weights_list.copy(),
        'prediction_results': {},
        'observed_data': {} }


    predict_param_dict_dict = create_predict_param_dict(basis_param_dict, code_item, data_property)


    for current_period in range(total_periods_num):
        print(f'This is period: {current_period}')

        current_period_info_dict = {
            'model_list': model_info_dict[f'periods_{current_period}']['model_list'].copy(),
            'model_age_list': model_info_dict[f'periods_{current_period}']['model_age_list'].copy(),
            'weights_list': model_info_dict[f'periods_{current_period}']['weights_list'].copy()
        }
        current_period_info_dict['weights_list'] = np.array(current_period_info_dict['weights_list'])
        if current_period < total_periods_num - 1:
            if total_model_num > 1:

                posterior_prob = calculate_posterior_prob(current_period_info_dict['model_list'], current_period_info_dict['weights_list'],
                                                          result_data_dict['predict_eval_scores'], likelihood_method)


                models_dict_to_remove = {'model_list': current_period_info_dict['model_list'], 'model_age_list': current_period_info_dict['model_age_list'],
                                         'post_weights_list': posterior_prob}
                survival_model_dict, death_model_dict = remove_model(models_dict_to_remove, remove_model_prob)

                death_model_dict['model_list'] = np.array(death_model_dict['model_list'])
                remove_model_num = len(death_model_dict['model_list'])
                model_info_dict[f'periods_{current_period + 1}'] = {
                    'model_list': updated_models_dict['model_list'],
                    'model_age_list': updated_models_dict['model_age_list'],
                    'weights_list': updated_models_dict['weights_list']}
            else:

                model_info_dict[f'periods_{current_period + 1}'] = {
                    'model_list': current_period_info_dict['model_list'],
                    'model_age_list': 1 + current_period_info_dict['model_age_list'],
                    'weights_list': current_period_info_dict['weights_list']}

    last_period_model_info_dict = model_info_dict[f'periods_{total_periods_num - 1}']
    return last_period_model_info_dict, model_info_dict




def generate_single_period_predict_results(ML_model_saved_dir, running_info_saved_dir, ML_predict_param_dict, code_item, data_property,
                                           total_periods_num, total_model_num, model_list, model_age_list, period_item):

    predict_result_dict = {'predict_data': {}, 'predict_eval_scores': {}, 'observed_data': {}}


    observed_data = ML_predict_param_dict['data_parameters']['observed_data']
    if observed_data is not None:
        predict_result_dict['observed_data'] = observed_data



    log_file_path = os.path.join(running_info_saved_dir, f'running_log_{code_item}.txt')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    total_model_num = len(model_list)


    with open(log_file_path, 'a') as log_file:
        with tqdm(total=total_model_num, desc='Getting prediction data', unit='model') as pbar:
            for model_type, model_age in zip(model_list, model_age_list):
                if model_age == 0:
                    model_born_period = period_item
                    Train = True
                else:
                    model_born_period = int(period_item - model_age)
                    Train = False


                model_dir = os.path.join(ML_model_saved_dir, f'periods_{model_born_period}', f'model_{model_type}')
                os.makedirs(model_dir, exist_ok=True)
                predict_dict, real_dict, eval_dict = get_predict_data(data_property, model_type, model_dir,  ML_predict_param_dict, Train)
               
                predict_result_dict['predict_eval_scores'][model_type] = eval_dict['test']

    return predict_result_dict



def generation_forward_looking_info(last_period_model_info_dict):

    weights_list = last_period_model_info_dict['weights_list']
    model_list = last_period_model_info_dict['model_list']
    predict_data_dict = last_period_model_info_dict['prediction_results']

    if len(model_list) != len(weights_list):
        raise ValueError('The length of model_list and weights_list must be the same.')
    forward_data = None


    for model_type, model_weight in zip(model_list, weights_list):
        if model_type not in predict_data_dict:
            raise ValueError(f"Model type '{model_type}' not found in predict_data_dict.")

        data = np.array(predict_data_dict[model_type])

        if forward_data is None:
            forward_data = np.zeros_like(data)

        forward_data += model_weight * data

    return forward_data



def generation_eval_result_with_combine_info(forward_data_dict, historical_data_dict, structural_basis_param_dict):

    weight_historical = structural_basis_param_dict['weight_historical']


    historical_eval_dict = calculate_performance_eval_ratios(historical_data_dict, structural_basis_param_dict)
    forward_eval_dict = calculate_performance_eval_ratios(forward_data_dict, structural_basis_param_dict)

    total_eval_dict = {}
    for key in historical_eval_dict.keys():
        total_eval_dict[key] = (weight_historical * historical_eval_dict[key] + (1 - weight_historical) * forward_eval_dict[key])
    return total_eval_dict



def calculate_posterior_prob(model_list, prior_prob_list, predict_dict, likelihood_method):

    model_num = len(prior_prob_list)
    if len(model_list) != model_num:
        raise ValueError('Length of model_list and prior_prob_list must be the same!')

    posterior_prob_list = np.zeros(model_num)


    for i in range(model_num):
        model_type = model_list[i]


        if model_type not in predict_dict:
            raise ValueError(f"Model type '{model_type}' not found in predict_dict.")

        eval_scores_dict = predict_dict[model_type]
        likelihood = calculate_likelihood(eval_scores_dict, likelihood_method)
        posterior_prob_list[i] = likelihood * prior_prob_list[i]


    posterior_prob_list = np.clip(posterior_prob_list, 0, None)

    total_posterior = np.sum(posterior_prob_list)
    if total_posterior == 0:
        raise ValueError('The sum of posterior probabilities is zero, which may indicate incorrect likelihoods.')

    posterior_prob_list /= total_posterior

    return posterior_prob_list




def calculate_likelihood(eval_scores_dict, likelihood_method):

    epsilon = 1e-6
    residuals = np.array(eval_scores_dict['residuals']).flatten()
    mse = eval_scores_dict['MSE']
    Huber_loss = eval_scores_dict['Huber_loss']

    if likelihood_method == 'residuals_std':
        std_residuals = np.std(residuals)
        likelihood = 1 / (std_residuals + epsilon)

    elif likelihood_method == 'MSE':
        likelihood = 1 / (mse + epsilon)

    elif likelihood_method == 'Huber_loss':
        likelihood = 1 / (Huber_loss + epsilon)

    elif likelihood_method == 'experience':
        likelihood = np.mean(np.exp(-np.abs(residuals)))

    elif likelihood_method == 'nonparametric':
        kde = gaussian_kde(residuals)
        likelihood = kde.evaluate(np.mean(residuals))[0]

    else:
        raise ValueError(f'Unknown method: {likelihood_method}')

    return likelihood



def calculate_performance_eval_ratios(data_dict, structural_basis_param_dict):

    if 'return' not in data_dict or 'ESG' not in data_dict:
        raise KeyError("data_dict must contain both 'return' and 'ESG' keys.")


    annual_riskfree_rate = structural_basis_param_dict['annual_riskfree_rate']
    trading_days_per_year = structural_basis_param_dict['trading_days_per_year']
    ESG_perfect_score = structural_basis_param_dict['ESG_perfect_score']

    daily_riskfree_rate = annual_riskfree_rate / trading_days_per_year
    riskfree_return = np.exp(daily_riskfree_rate)


    statistics_dict = calculate_partial_statistics(data_dict, riskfree_return)


    mean_return = statistics_dict['return']['mean']
    std_return = statistics_dict['return']['std']
    downside_std_return = statistics_dict['return']['downside_std']
    mean_ESG = statistics_dict['ESG']['mean']

    epsilon = 1e-5

    excess_return = mean_return - riskfree_return

    if std_return < epsilon:
        Sharpe_ratio = np.nan
    else:
        Sharpe_ratio = excess_return / std_return

    if downside_std_return < epsilon:
        Sortino_ratio = np.nan
    else:
        Sortino_ratio = excess_return / downside_std_return


    ESG_adjusted_rate = mean_ESG / ESG_perfect_score


    ESG_adjusted_Sharpe_ratio = (1 + ESG_adjusted_rate)  * Sharpe_ratio if not np.isnan(Sharpe_ratio) else np.nan
    ESG_adjusted_Sortino_ratio = (1 + ESG_adjusted_rate) * Sortino_ratio if not np.isnan(Sortino_ratio) else np.nan


    performance_eval_ratios_dict = {'traditional_Sharpe_ratio': Sharpe_ratio, 'traditional_Sortino_ratio': Sortino_ratio,
                                    'mean_return': mean_return, 'mean_ESG': mean_ESG, 'ESG_adjusted_Sharpe_ratio': ESG_adjusted_Sharpe_ratio,
                                    'ESG_adjusted_Sortino_ratio': ESG_adjusted_Sortino_ratio}
    return performance_eval_ratios_dict




def calculate_partial_statistics(data_dict, riskfree_return):

    statistics_dict = {}

    for key, data in data_dict.items():
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        mean = np.mean(data)
        std = np.std(data)
        statistics_dict[key] = {'mean': mean, 'std': std}

        if key == 'return':

            semi_var_below_rf = np.var(data[data < riskfree_return])
            semi_std_below_rf = np.sqrt(semi_var_below_rf)
            statistics_dict[key]['downside_std'] = semi_std_below_rf

    return statistics_dict


