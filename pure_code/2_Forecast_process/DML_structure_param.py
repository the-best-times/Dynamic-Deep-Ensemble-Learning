
import numpy as np
import pandas as pd
import os
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.losses import Huber



from ML_algorithm import build_deep_ML_model, DML_model_type_to_category
from process_time_series_data import transform_data_shape, split_and_combine_data, normalize_data, inverse_normalize_data, \
    dimensionality_reduction,  evaluation_predict, build_autoencoders_model, calculate_scores
from process_sample_data import generate_ML_predict_params, extract_ML_predict_params_by_period


def get_parameters(data_property, model_type, data_x, data_y):

    common_params_dict = {
        'output_units': data_y.shape[1],
        'timesteps': data_x.shape[1],
        'num_features': data_x.shape[2]
    }


    model_parameters_dict = common_params_dict.copy()
    model_category = DML_model_type_to_category(model_type)

    return_params = {
        'MLP_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'random_normal',
            'activation_fun': 'relu',
            'optimizer': 'Adam',
            'learning_rate': 0.1,
            'units_list': [192, 208, 176]
        },
        'CNN_category': {
            'kernel_init': 'he_normal',
            'bias_init': 'ones',
            'activation_fun': 'sigmoid',
            'optimizer': 'Adamax',
            'learning_rate': 0.0075,
            'num_filter_list': [16, 32, 208],
            'kernel_size_list': [4, 4, 4],
            'pool_size_list': [4, 3, 2],
            'dropout_rate': 0.45,
            'hidden_units_list': [112, 16]
        },
        'CNN_variant_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'zeros',
            'activation_fun': 'relu',
            'optimizer': 'Adamax',
            'learning_rate': 0.075,
            'num_filter_list': [160],
            'kernel_size_list': [4],
            'pool_size_list': [4],
            'dropout_rate': 0.10,
            'hidden_units_list': [144, 240]
        },
        'TCN_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'zeros',
            'activation_fun': 'relu',
            'optimizer': 'Adamax',
            'learning_rate': 0.15,
            'num_filter_list': [256, 224, 80],
            'kernel_size_list': [3, 2, 3],
            'dropout_rate': 0.40,
            'hidden_units_list': [64, 176]
        },
        'RNN_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'zeros',
            'activation_fun': 'relu',
            'optimizer': 'Adamax',
            'learning_rate': 0.2,
            'recurrent_init': 'orthogonal',
            'units_list': [160, 192],
            'dropout_rate': 0.50,
            'hidden_units_list': [64, 112]
        },
        'RNN_attention_category': {
            'kernel_init': 'he_normal',
            'bias_init': 'ones',
            'activation_fun': 'relu',
            'optimizer': 'RMSprop',
            'learning_rate': 0.1,
            'recurrent_init': 'glorot_uniform',
            'units_list': [144, 64, 48],
            'dropout_rate': 0.35,
            'hidden_units_list': [128, 112]
        },
        'CNN_RNN_category': {
            'kernel_init': 'he_normal',
            'bias_init': 'ones',
            'activation_fun': 'relu',
            'optimizer': 'Adamax',
            'learning_rate': 0.0075,
            'recurrent_init': 'orthogonal',
            'num_filter_list': [192, 224, 16],
            'kernel_size_list': [4, 2, 5],
            'pool_size_list': [3, 5, 5],
            'dropout_rate': 0.30,
            'rnn_units_list': [240, 112],
            'hidden_units_list': [192, 112]
        },
        'CNN_RNN_attention_category': {
            'kernel_init': 'lecun_normal',
            'bias_init': 'ones',
            'activation_fun': 'relu',
            'optimizer': 'Adamax',
            'learning_rate': 0.002,
            'recurrent_init': 'orthogonal',
            'num_filter_list': [144, 240],
            'kernel_size_list': [3, 3],
            'pool_size_list': [2, 5],
            'dropout_rate': 0.10,
            'rnn_units_list': [208, 240, 144],
            'hidden_units_list': [192, 32]
        }
    }

    ESG_params = {
        'MLP_category': {
            'kernel_init': 'lecun_normal',
            'bias_init': 'random_normal',
            'activation_fun': 'elu',
            'optimizer': 'Adam',
            'learning_rate': 0.025,
            'units_list': [16, 96, 16]
        },
        'CNN_category': {
            'kernel_init': 'lecun_normal',
            'bias_init': 'zeros',
            'activation_fun': 'sigmoid',
            'optimizer': 'Adam',
            'learning_rate': 0.00002,
            'num_filter_list': [208],
            'kernel_size_list': [3],
            'pool_size_list': [4],
            'dropout_rate': 0.20,
            'hidden_units_list': [64, 224]
        },
        'CNN_variant_category': {
            'kernel_init': 'he_normal',
            'bias_init': 'random_normal',
            'activation_fun': 'sigmoid',
            'optimizer': 'Adam',
            'learning_rate': 0.0002,
            'num_filter_list': [48],
            'kernel_size_list': [4],
            'pool_size_list': [4],
            'dropout_rate': 0.15,
            'hidden_units_list': [192, 256]
        },
        'TCN_category': {
            'kernel_init': 'lecun_normal',
            'bias_init': 'ones',
            'activation_fun': 'sigmoid',
            'optimizer': 'SGD',
            'learning_rate': 0.15,
            'num_filter_list': [32, 128],
            'kernel_size_list': [2, 5],
            'dropout_rate': 0.15,
            'hidden_units_list': [32, 160]
        },
        'RNN_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'ones',
            'activation_fun': 'sigmoid',
            'optimizer': 'Adamax',
            'learning_rate': 0.02,
            'recurrent_init': 'glorot_uniform',
            'units_list': [32],
            'dropout_rate': 0.45,
            'hidden_units_list': [144, 48]
        },
        'RNN_attention_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'ones',
            'activation_fun': 'sigmoid',
            'optimizer': 'RMSprop',
            'learning_rate': 0.001,
            'recurrent_init': 'lecun_normal',
            'units_list': [48],
            'dropout_rate': 0.45,
            'hidden_units_list': [128, 224]
        },
        'CNN_RNN_category': {
            'kernel_init': 'glorot_uniform',
            'bias_init': 'zeros',
            'activation_fun': 'sigmoid',
            'optimizer': 'Adamax',
            'learning_rate': 0.002,
            'recurrent_init': 'lecun_normal',
            'num_filter_list': [32, 224],
            'kernel_size_list': [4, 2],
            'pool_size_list': [2, 2],
            'dropout_rate': 0.40,
            'rnn_units_list': [64],
            'hidden_units_list': [176, 144]
        },
        'CNN_RNN_attention_category': {
            'kernel_init': 'he_normal',
            'bias_init': 'ones',
            'activation_fun': 'tanh',
            'optimizer': 'RMSprop',
            'learning_rate': 0.002,
            'recurrent_init': 'lecun_normal',
            'num_filter_list': [224, 32, 48],
            'kernel_size_list': [3, 5, 4],
            'pool_size_list': [4, 2, 4],
            'dropout_rate': 0.20,
            'rnn_units_list': [32],
            'hidden_units_list': [224, 96]
        }
    }


    if data_property == 'return':
        if model_category in return_params:
            model_parameters_dict.update(return_params[model_category])
        else:
            raise ValueError('Invalid model category for return data!')
    elif data_property == 'ESG':
        if model_category in ESG_params:
            model_parameters_dict.update(ESG_params[model_category])
        else:
            raise ValueError('Invalid model category for ESG data!')
    else:
        raise ValueError('Invalid data property!')

    return model_parameters_dict
