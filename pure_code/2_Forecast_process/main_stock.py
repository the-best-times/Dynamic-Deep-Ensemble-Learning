
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings, os, shutil
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=RuntimeWarning, module='tensorflow')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


from process_sample_data import  load_list_from_text, report_column_name, generate_statistics, generate_factor_model_params_dict
from solve_ML import get_predict_data, evaluation_predict
from asset_pool_construction import get_forward_looking_info_result, get_eval_result, select_asset



current_working_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_working_dir))




cal_forward_info = False


data_parent_dir = os.path.join(parent_dir, 'Data_and_Results', 'Data')
original_file_path = os.path.join(data_parent_dir, 'only_time_example.xlsx')
check_output_folder_path = os.path.join(data_parent_dir, 'Processed_Data', 'sample_data')
insample_data_folder_path = os.path.join(check_output_folder_path, 'insample_data')

results_saved_folder_statistics =  os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_describe_statistical')
results_saved_folder_adjust =  os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_adjust_hyperparam')
if cal_forward_info:
    results_saved_folder_path_pool = os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_asset_construction_')
    results_saved_folder_path_param = os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_param_estimation_')
else:
    results_saved_folder_path_pool = os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_asset_construction')
    results_saved_folder_path_param =  os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_param_estimation')


code_list = load_list_from_text(check_output_folder_path, 'code_list.txt')
code_item_list = [f'stock_{code}' for code in code_list]

report_col_name = False
# report_col_name = True
if report_col_name:
    report_column_name(insample_data_folder_path, code_list)




data_property_list = ['return', 'ESG']

trading_days_per_year = 252
look_back = round(trading_days_per_year/12 * 12)
delay = 0
look_forward = round(trading_days_per_year/12 * 3)
test_size = look_forward
valid_prob = 0.1
plot_train = False




annual_riskfree_rate = 0.015

ESG_perfect_score = 10
date_split_node_list = ['2020-01-01', '2021-07-01', '2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01']
# date_split_node_list = ['2020-01-01', '2022-01-01', '2023-01-01', '2023-07-01']
# date_split_node_list = ['2020-01-01', '2023-07-01']






ML_model_list = ['Lasso', 'Ridge', 'Elasticnet', 'Linear', 'SGD', 'BayesianRidge', 'PassiveAggressive', 'Huber', 'OMP','ARD', 'LassoLars', 'RandomForest',
                 'AdaBoost', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting', 'Bagging', 'Knearest', 'SVR_linear', 'SVR_poly', 'SVR_rbf', 'SVR_sigmoid',
                 'PLS', 'CCA', 'DecisionTree', 'KernelRidge', 'XGBoost','CatBoost', 'GBDT', 'MLP']

DML_model_list = ['MLP_deep', 'CNN', 'CNN_channel', 'CNN_spatial', 'CNN_cbam', 'RCNN', 'RCNN_channel', 'RCNN_spatial', 'RCNN_cbam', 'DenseNet',
                   'DenseNet_channel', 'DenseNet_spatial', 'DenseNet_cbam', 'TCN', 'TCN_channel', 'TCN_spatial', 'TCN_cbam','RNN', 'LSTM', 'GRU',
                   'BiRNN_sum', 'BiLSTM_sum', 'BiGRU_sum','BiRNN_mul', 'BiLSTM_mul','BiGRU_mul',
                   'BiRNN_ave', 'BiLSTM_ave', 'BiGRU_ave', 'BiRNN_concat', 'BiLSTM_concat', 'BiGRU_concat',
                   'RNN_attention', 'LSTM_attention', 'GRU_attention', 'BiRNN_sum_attention', 'BiLSTM_sum_attention', 'BiGRU_sum_attention',
                   'BiRNN_mul_attention', 'BiLSTM_mul_attention', 'BiGRU_mul_attention',
                   'BiRNN_ave_attention', 'BiLSTM_ave_attention', 'BiGRU_ave_attention',
                   'BiRNN_concat_attention', 'BiLSTM_concat_attention', 'BiGRU_concat_attention',
                   'CNN_RNN', 'CNN_LSTM', 'CNN_GRU', 'CNN_BiRNN_sum', 'CNN_BiLSTM_sum', 'CNN_BiGRU_sum',
                   'CNN_BiRNN_mul', 'CNN_BiLSTM_mul', 'CNN_BiGRU_mul', 'CNN_BiRNN_ave', 'CNN_BiLSTM_ave', 'CNN_BiGRU_ave',
                   'CNN_BiRNN_concat', 'CNN_BiLSTM_concat', 'CNN_BiGRU_concat',
                   'CNN_RNN_attention', 'CNN_LSTM_attention', 'CNN_GRU_attention',
                   'CNN_BiRNN_sum_attention', 'CNN_BiLSTM_sum_attention', 'CNN_BiGRU_sum_attention',
                   'CNN_BiRNN_mul_attention', 'CNN_BiLSTM_mul_attention', 'CNN_BiGRU_mul_attention',
                   'CNN_BiRNN_ave_attention', 'CNN_BiLSTM_ave_attention', 'CNN_BiGRU_ave_attention',
                   'CNN_BiRNN_concat_attention', 'CNN_BiLSTM_concat_attention', 'CNN_BiGRU_concat_attention']




hyperparameter_dict = { 'return': {'epoch': 130, 'batch_sizes': 496, 'num_components': 19},  'ESG': {'epoch': 140, 'batch_sizes': 416, 'num_components': 13} }



dim_reduction = True
reduced_method = 'Autoencoders'
likelihood_method = 'Huber_loss'



total_model_num = 10
remove_model_prob = 0.20
nu = 0.10


weight_historical = 0

asset_select_param_dict = {'num_select': 10, 'select_method': 'dual_sort', 'indicator': 'mean'}


x0 = 1

target_return = 1.10
target_score = 6
lending_ratio = 0.30
cons_type = 'non_short'
M = None
ESG_cons = True
investment_preference_param = {'x0': x0,  'lending_ratio': lending_ratio, 'target_return' : target_return, 'target_score' : target_score,
                               'cons_type': cons_type, 'M': M, 'ESG_cons': ESG_cons }


predict_basis_param_dict = {'ML_model_list': ML_model_list, 'DML_model_list': DML_model_list,  'plot_train': plot_train, 'hyperparameter_dict': hyperparameter_dict,
                            'dim_reduction': dim_reduction, 'reduced_method': reduced_method}

structural_basis_param_dict = {'date_split_node_list': date_split_node_list, 'look_back': look_back, 'delay': delay, 'look_forward': look_forward, 'test_size': test_size, 'valid_prob': valid_prob,
                               'data_property_list': data_property_list, 'weight_historical': weight_historical, 'ESG_perfect_score': ESG_perfect_score,
                               'trading_days_per_year': trading_days_per_year, 'annual_riskfree_rate': annual_riskfree_rate,
                               'total_model_num': total_model_num, 'remove_model_prob': remove_model_prob, 'nu':nu,
                               'likelihood_method': likelihood_method, 'asset_select_param_dict': asset_select_param_dict,
                               'investment_preference_param': investment_preference_param}

path_basis_dict = {'original_file_path': original_file_path, 'data_folder_path': insample_data_folder_path, 'results_saved_folder_statistics': results_saved_folder_statistics,
                   'results_saved_folder_path_pool': results_saved_folder_path_pool, 'results_saved_folder_path_param': results_saved_folder_path_param}



basis_param_dict = {'structural_basis_param_dict': structural_basis_param_dict, 'predict_basis_param_dict': predict_basis_param_dict, 'path_basis_dict': path_basis_dict}


prediction_model_list_ML = ['Lasso', 'Ridge', 'Elasticnet', 'Linear', 'SGD', 'BayesianRidge', 'PassiveAggressive', 'Huber', 'OMP','ARD',
                            'LassoLars', 'RandomForest', 'AdaBoost', 'ExtraTrees', 'GradientBoosting', 'HistGradientBoosting', 'Bagging', 'Knearest', 'SVR_linear', 'PLS',
                            'CCA', 'DecisionTree', 'KernelRidge', 'XGBoost']

prediction_model_list_DML = ['MLP_deep', 'CNN', 'CNN_channel', 'CNN_spatial', 'CNN_cbam', 'RCNN', 'RCNN_channel', 'RCNN_spatial', 'RCNN_cbam', 'DenseNet',
                   'DenseNet_channel', 'DenseNet_spatial', 'DenseNet_cbam', 'TCN', 'TCN_channel', 'TCN_spatial', 'TCN_cbam',
                   'RNN', 'LSTM', 'GRU', 'BiRNN_concat', 'BiLSTM_concat', 'BiGRU_concat',
                   'RNN_attention', 'LSTM_attention', 'GRU_attention',  'BiRNN_concat_attention', 'BiLSTM_concat_attention', 'BiGRU_concat_attention',
                   'CNN_RNN', 'CNN_LSTM', 'CNN_GRU', 'CNN_BiRNN_concat', 'CNN_BiLSTM_concat', 'CNN_BiGRU_concat',
                   'CNN_RNN_attention', 'CNN_LSTM_attention', 'CNN_GRU_attention', 'CNN_BiRNN_concat_attention', 'CNN_BiLSTM_concat_attention', 'CNN_BiGRU_concat_attention']


# prediction_model_list =  prediction_model_list_ML
prediction_model_list = prediction_model_list_DML


# construct_asset_pool = False
construct_asset_pool = True

eval_forward_info = True
# eval_forward_info = False






code_split_node_list = [0, 20, 40, 60, 80, 100]
cal_count =  0


if construct_asset_pool:

    if cal_forward_info:
        if cal_count < len(code_split_node_list) - 1:
            code_list_ = code_item_list[code_split_node_list[cal_count]:code_split_node_list[cal_count + 1]]
        else:
            code_list_ = code_item_list[code_split_node_list[cal_count]:]


        get_forward_looking_info_result(basis_param_dict, prediction_model_list, code_list_, cal_forward_info, cal_count)
        print('The forward looking information calculation is complete!')

    else:
        if eval_forward_info:
            avg_metrics_dict = get_eval_result(basis_param_dict, prediction_model_list, code_item_list, cal_forward_info, code_split_node_list)
            print(avg_metrics_dict)
        else:
            select_asset_list = select_asset(basis_param_dict, prediction_model_list, code_item_list, cal_forward_info, code_split_node_list)
            print(select_asset_list)




