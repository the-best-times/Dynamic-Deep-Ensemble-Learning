
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")




from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import  Lasso, Ridge, ElasticNet, LinearRegression, SGDRegressor, BayesianRidge, \
    PassiveAggressiveRegressor, HuberRegressor, OrthogonalMatchingPursuit, ARDRegression, Lars, LassoLars
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from keras.losses import Huber



import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import get as get_initializer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax
from tensorflow.keras.layers import Dense, Reshape, Activation, Flatten, Dropout, Input, Multiply, Add, Layer,  Concatenate, ZeroPadding1D, BatchNormalization
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Attention
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tcn import TCN


def build_ML_model(model_type):

    model_mapping = {
        'Lasso': Lasso(alpha=1.0),

        'Ridge': Ridge(alpha=1.0),
        'Elasticnet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Linear': LinearRegression(),
        'SGD': MultiOutputRegressor(PassiveAggressiveRegressor()),
        'BayesianRidge': MultiOutputRegressor(BayesianRidge()),
        'PassiveAggressive': MultiOutputRegressor(PassiveAggressiveRegressor()),
        'Huber': MultiOutputRegressor(HuberRegressor()),
        'OMP': MultiOutputRegressor(OrthogonalMatchingPursuit()),
        'ARD': MultiOutputRegressor(ARDRegression()),
        'LassoLars': LassoLars(),
        'RandomForest': RandomForestRegressor(),
        'AdaBoost': MultiOutputRegressor(AdaBoostRegressor()),
        'ExtraTrees': MultiOutputRegressor(ExtraTreesRegressor()),
        'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor()),
        'HistGradientBoosting': MultiOutputRegressor(HistGradientBoostingRegressor()),
        'Bagging': MultiOutputRegressor(BaggingRegressor()),
        'Knearest': KNeighborsRegressor(),
        'SVR_linear': MultiOutputRegressor(SVR(kernel='linear')),
        'SVR_poly': MultiOutputRegressor(SVR(kernel='poly')),
        'SVR_rbf': MultiOutputRegressor(SVR(kernel='rbf')),
        'SVR_sigmoid': MultiOutputRegressor(SVR(kernel='sigmoid')),
        'PLS': PLSRegression(),
        'CCA': CCA(),
        'DecisionTree': MultiOutputRegressor(DecisionTreeRegressor()),
        'KernelRidge': MultiOutputRegressor(KernelRidge(alpha=1.0)),
        'XGBoost': XGBRegressor(),
        'CatBoost': MultiOutputRegressor(CatBoostRegressor(verbose=0)),
        'GBDT': MultiOutputRegressor(LGBMRegressor(verbose=0)),
        'MLP': MultiOutputRegressor(MLPRegressor())
    }


    if model_type in model_mapping:
        return model_mapping[model_type]
    else:
        raise ValueError('Invalid model type!')



def build_deep_ML_model(model_type, model_parameters_dict):


    model_type_mapping = {
        'MLP_deep':  (build_MLP_model,  (model_parameters_dict, )),

        'CNN':  (build_CNN_Attention_model,  (model_parameters_dict, None)),
        'CNN_channel': (build_CNN_Attention_model, (model_parameters_dict, 'channel')),
        'CNN_spatial': (build_CNN_Attention_model, (model_parameters_dict, 'spatial')),
        'CNN_cbam':    (build_CNN_Attention_model, (model_parameters_dict, 'cbam')),

        'RCNN': (build_CNN_variant_Attention_model, (model_parameters_dict, 'recursive', None)),
        'RCNN_channel': (build_CNN_variant_Attention_model, (model_parameters_dict, 'recursive', 'channel')),
        'RCNN_spatial': (build_CNN_variant_Attention_model, (model_parameters_dict, 'recursive', 'spatial')),
        'RCNN_cbam':    (build_CNN_variant_Attention_model, (model_parameters_dict, 'recursive', 'cbam')),
        'DenseNet': (build_CNN_variant_Attention_model, (model_parameters_dict, 'dense', None)),
        'DenseNet_channel': (build_CNN_variant_Attention_model, (model_parameters_dict, 'dense', 'channel')),
        'DenseNet_spatial': (build_CNN_variant_Attention_model, (model_parameters_dict, 'dense', 'spatial')),
        'DenseNet_cbam':    (build_CNN_variant_Attention_model, (model_parameters_dict, 'dense', 'cbam')),

        'TCN': (build_TCN_Attention_model, (model_parameters_dict, None)),
        'TCN_channel': (build_TCN_Attention_model, (model_parameters_dict, 'channel')),
        'TCN_spatial': (build_TCN_Attention_model, (model_parameters_dict, 'spatial')),
        'TCN_cbam':    (build_TCN_Attention_model, (model_parameters_dict, 'cbam')),

        'RNN':  (build_RNN_model,  ('RNN',  'unidirectional', None, model_parameters_dict)),
        'LSTM': (build_RNN_model,  ('LSTM', 'unidirectional', None, model_parameters_dict)),
        'GRU':  (build_RNN_model,  ('GRU',  'unidirectional', None, model_parameters_dict)),

        'BiRNN_sum':  (build_RNN_model,  ('RNN',  'bidirectional', 'sum', model_parameters_dict)),
        'BiLSTM_sum': (build_RNN_model,  ('LSTM', 'bidirectional', 'sum', model_parameters_dict)),
        'BiGRU_sum':  (build_RNN_model,  ('GRU',  'bidirectional', 'sum', model_parameters_dict)),
        'BiRNN_mul':  (build_RNN_model,  ('RNN',  'bidirectional', 'mul', model_parameters_dict)),
        'BiLSTM_mul': (build_RNN_model,  ('LSTM', 'bidirectional', 'mul', model_parameters_dict)),
        'BiGRU_mul':  (build_RNN_model,  ('GRU',  'bidirectional', 'mul', model_parameters_dict)),
        'BiRNN_ave':  (build_RNN_model,  ('RNN',  'bidirectional', 'ave', model_parameters_dict)),
        'BiLSTM_ave': (build_RNN_model,  ('LSTM', 'bidirectional', 'ave', model_parameters_dict)),
        'BiGRU_ave':  (build_RNN_model,  ('GRU',  'bidirectional', 'ave', model_parameters_dict)),
        'BiRNN_concat':  (build_RNN_model, ('RNN',  'bidirectional', 'concat', model_parameters_dict)),
        'BiLSTM_concat': (build_RNN_model, ('LSTM', 'bidirectional', 'concat', model_parameters_dict)),
        'BiGRU_concat':  (build_RNN_model, ('GRU',  'bidirectional', 'concat', model_parameters_dict)),

        'RNN_attention':  (build_RNN_Attention_model, ('RNN', 'unidirectional', None, model_parameters_dict)),
        'LSTM_attention': (build_RNN_Attention_model, ('LSTM', 'unidirectional', None, model_parameters_dict)),
        'GRU_attention':  (build_RNN_Attention_model, ('GRU', 'unidirectional', None, model_parameters_dict)),
        'BiRNN_sum_attention':    (build_RNN_Attention_model, ('RNN',  'bidirectional', 'sum', model_parameters_dict)),
        'BiLSTM_sum_attention':   (build_RNN_Attention_model, ('LSTM', 'bidirectional', 'sum', model_parameters_dict)),
        'BiGRU_sum_attention':    (build_RNN_Attention_model, ('GRU',  'bidirectional', 'sum', model_parameters_dict)),
        'BiRNN_mul_attention':    (build_RNN_Attention_model, ('RNN',  'bidirectional', 'mul', model_parameters_dict)),
        'BiLSTM_mul_attention':   (build_RNN_Attention_model, ('LSTM', 'bidirectional', 'mul', model_parameters_dict)),
        'BiGRU_mul_attention':    (build_RNN_Attention_model, ('GRU',  'bidirectional', 'mul', model_parameters_dict)),
        'BiRNN_ave_attention':    (build_RNN_Attention_model, ('RNN',  'bidirectional', 'ave', model_parameters_dict)),
        'BiLSTM_ave_attention':   (build_RNN_Attention_model, ('LSTM', 'bidirectional', 'ave', model_parameters_dict)),
        'BiGRU_ave_attention':    (build_RNN_Attention_model, ('GRU',  'bidirectional', 'ave', model_parameters_dict)),
        'BiRNN_concat_attention': (build_RNN_Attention_model, ('RNN',  'bidirectional', 'concat', model_parameters_dict)),
        'BiLSTM_concat_attention':(build_RNN_Attention_model, ('LSTM', 'bidirectional', 'concat', model_parameters_dict)),
        'BiGRU_concat_attention': (build_RNN_Attention_model, ('GRU',  'bidirectional', 'concat', model_parameters_dict)),

        'CNN_RNN':  (build_CNN_RNN_model, ('RNN',  'unidirectional', None, model_parameters_dict)),
        'CNN_LSTM': (build_CNN_RNN_model, ('LSTM', 'unidirectional', None, model_parameters_dict)),
        'CNN_GRU':  (build_CNN_RNN_model, ('GRU',  'unidirectional', None, model_parameters_dict)),
        'CNN_BiRNN_sum':    (build_CNN_RNN_model, ('RNN',  'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiLSTM_sum':   (build_CNN_RNN_model, ('LSTM', 'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiGRU_sum':    (build_CNN_RNN_model, ('GRU',  'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiRNN_mul':    (build_CNN_RNN_model, ('RNN',  'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiLSTM_mul':   (build_CNN_RNN_model, ('LSTM', 'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiGRU_mul':    (build_CNN_RNN_model, ('GRU',  'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiRNN_ave':    (build_CNN_RNN_model, ('RNN',  'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiLSTM_ave':   (build_CNN_RNN_model, ('LSTM', 'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiGRU_ave':    (build_CNN_RNN_model, ('GRU',  'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiRNN_concat': (build_CNN_RNN_model, ('RNN',  'bidirectional', 'concat', model_parameters_dict)),
        'CNN_BiLSTM_concat':(build_CNN_RNN_model, ('LSTM', 'bidirectional', 'concat', model_parameters_dict)),
        'CNN_BiGRU_concat': (build_CNN_RNN_model, ('GRU',  'bidirectional', 'concat', model_parameters_dict)),

        'CNN_RNN_attention':  (build_CNN_RNN_Attention_model, ('RNN', 'unidirectional', None, model_parameters_dict)),
        'CNN_LSTM_attention': (build_CNN_RNN_Attention_model, ('LSTM', 'unidirectional', None, model_parameters_dict)),
        'CNN_GRU_attention':  (build_CNN_RNN_Attention_model, ('GRU', 'unidirectional', None, model_parameters_dict)),
        'CNN_BiRNN_sum_attention':    (build_CNN_RNN_Attention_model, ('RNN',  'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiLSTM_sum_attention':   (build_CNN_RNN_Attention_model, ('LSTM', 'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiGRU_sum_attention':    (build_CNN_RNN_Attention_model, ('GRU',  'bidirectional', 'sum', model_parameters_dict)),
        'CNN_BiRNN_mul_attention':    (build_CNN_RNN_Attention_model, ('RNN',  'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiLSTM_mul_attention':   (build_CNN_RNN_Attention_model, ('LSTM', 'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiGRU_mul_attention':    (build_CNN_RNN_Attention_model, ('GRU',  'bidirectional', 'mul', model_parameters_dict)),
        'CNN_BiRNN_ave_attention':    (build_CNN_RNN_Attention_model, ('RNN',  'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiLSTM_ave_attention':   (build_CNN_RNN_Attention_model, ('LSTM', 'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiGRU_ave_attention':    (build_CNN_RNN_Attention_model, ('GRU',  'bidirectional', 'ave', model_parameters_dict)),
        'CNN_BiRNN_concat_attention': (build_CNN_RNN_Attention_model, ('RNN',  'bidirectional', 'concat', model_parameters_dict)),
        'CNN_BiLSTM_concat_attention':(build_CNN_RNN_Attention_model, ('LSTM', 'bidirectional', 'concat', model_parameters_dict)),
        'CNN_BiGRU_concat_attention': (build_CNN_RNN_Attention_model, ('GRU',  'bidirectional', 'concat', model_parameters_dict))
    }


    if model_type in model_type_mapping:
        model_function, parameters = model_type_mapping[model_type]
        model = model_function(*parameters)
    else:
        raise ValueError('Invalid model type!')
    return model


def DML_model_type_to_category(model_type):

    if model_type in ['MLP_deep']:
        model_category_type = 'MLP_category'
    elif model_type in ['CNN', 'CNN_channel', 'CNN_spatial', 'CNN_cbam']:
        model_category_type = 'CNN_category'
    elif model_type in ['RCNN', 'RCNN_channel', 'RCNN_spatial', 'RCNN_cbam', 'DenseNet', 'DenseNet_channel', 'DenseNet_spatial', 'DenseNet_cbam']:
        model_category_type = 'CNN_variant_category'
    elif model_type in ['TCN', 'TCN_channel', 'TCN_spatial', 'TCN_cbam']:
        model_category_type = 'TCN_category'
    elif model_type in ['RNN', 'LSTM', 'GRU', 'BiRNN_sum', 'BiLSTM_sum', 'BiGRU_sum', 'BiRNN_mul', 'BiLSTM_mul',
                        'BiGRU_mul', 'BiRNN_ave', 'BiLSTM_ave', 'BiGRU_ave', 'BiRNN_concat', 'BiLSTM_concat',
                        'BiGRU_concat']:
        model_category_type = 'RNN_category'
    elif model_type in ['RNN_attention', 'LSTM_attention', 'GRU_attention', 'BiRNN_sum_attention', 'BiLSTM_sum_attention', 'BiGRU_sum_attention',
                        'BiRNN_mul_attention', 'BiLSTM_mul_attention', 'BiGRU_mul_attention', 'BiRNN_ave_attention', 'BiLSTM_ave_attention', 'BiGRU_ave_attention',
                        'BiRNN_concat_attention', 'BiLSTM_concat_attention', 'BiGRU_concat_attention']:
        model_category_type = 'RNN_attention_category'
    elif model_type in ['CNN_RNN', 'CNN_LSTM', 'CNN_GRU', 'CNN_BiRNN_sum', 'CNN_BiLSTM_sum', 'CNN_BiGRU_sum',
                        'CNN_BiRNN_mul', 'CNN_BiLSTM_mul', 'CNN_BiGRU_mul', 'CNN_BiRNN_ave', 'CNN_BiLSTM_ave',
                        'CNN_BiGRU_ave', 'CNN_BiRNN_concat', 'CNN_BiLSTM_concat', 'CNN_BiGRU_concat']:
        model_category_type = 'CNN_RNN_category'
    elif model_type in ['CNN_RNN_attention', 'CNN_LSTM_attention', 'CNN_GRU_attention',
                        'CNN_BiRNN_sum_attention', 'CNN_BiLSTM_sum_attention', 'CNN_BiGRU_sum_attention',
                        'CNN_BiRNN_mul_attention', 'CNN_BiLSTM_mul_attention', 'CNN_BiGRU_mul_attention',
                        'CNN_BiRNN_ave_attention', 'CNN_BiLSTM_ave_attention', 'CNN_BiGRU_ave_attention',
                        'CNN_BiRNN_concat_attention', 'CNN_BiLSTM_concat_attention', 'CNN_BiGRU_concat_attention']:
        model_category_type = 'CNN_RNN_attention_category'
    return model_category_type



def get_common_parameters(model_parameters_dict):

    huber_loss = Huber(delta=1.0)


    timesteps = model_parameters_dict['timesteps']
    num_features = model_parameters_dict['num_features']


    kernel_init = get_initializer(model_parameters_dict.get('kernel_init', 'glorot_uniform'))
    bias_init = get_initializer(model_parameters_dict.get('bias_init', 'zeros'))


    activation_fun = model_parameters_dict.get('activation_fun', 'relu')
    optimizer = model_parameters_dict.get('optimizer', 'Adam')
    lr = model_parameters_dict.get('learning_rate', 1e-3)
    if optimizer == 'Adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=lr)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=lr)
    elif optimizer == 'Adamax':
        opt = Adamax(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    metric = model_parameters_dict.get('metric', ['mse', 'mae'])


    loss_fun = huber_loss

    return timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun





def build_MLP_model(model_parameters_dict):

    units_list = model_parameters_dict['units_list']
    output_units = model_parameters_dict['output_units']
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)


    model = Sequential()
    model.add(Flatten(input_shape = (timesteps, num_features)))
    for i in range(len(units_list)):
        model.add(Dense(units = units_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))

    model.add(Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init))


    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model




def build_CNN_Attention_model(model_parameters_dict, attention_type):

    num_filter_list = model_parameters_dict['num_filter_list']
    kernel_size_list = model_parameters_dict['kernel_size_list']
    pool_size_list = model_parameters_dict['pool_size_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)

    model = Sequential()

    for i in range(len(num_filter_list)):
        if i == 0:
            model.add(Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], input_shape = (timesteps, num_features), kernel_initializer = kernel_init,
                             bias_initializer = bias_init,  activation = activation_fun))
            model.add(MaxPooling1D(pool_size = pool_size_list[i]))
        else:
            model.add(Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))
            model.add(MaxPooling1D(pool_size = pool_size_list[i]))


    if attention_type == 'channel':
        model.add(ChannelAttention())
    elif attention_type == 'spatial':
        model.add(SpatialAttention())
    elif attention_type == 'cbam':
        model.add(CBAM())
    elif attention_type is None:
        pass

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for hidden_unit in hidden_units_list:
        model.add(Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activation_fun))
        model.add(Dropout(dropout_rate))
    model.add(Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init))

    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model




def build_CNN_variant_Attention_model(model_parameters_dict, variant_type, attention_type):


    num_filter_list = model_parameters_dict['num_filter_list']
    kernel_size_list = model_parameters_dict['kernel_size_list']
    pool_size_list = model_parameters_dict['pool_size_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)



    inputs = Input(shape = (timesteps, num_features))
    x = inputs

    if variant_type == 'recursive':
        for i in range(len(num_filter_list)):
            x = Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun, padding = 'same')(x)
            x = MaxPooling1D(pool_size = pool_size_list[i])(x)

            if i > 0:
                x = Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun, padding = 'same')(x)
                x = MaxPooling1D(pool_size = pool_size_list[i])(x)
    elif variant_type == 'dense':
        concatenated_layers = [x]
        for i in range(len(num_filter_list)):
            conv = Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun, padding = 'same')(x)
            pool = MaxPooling1D(pool_size = pool_size_list[i])(conv)
            x = pool
            concatenated_layers.append(x)
            if i > 0:

                max_len = max([layer.shape[1] for layer in concatenated_layers])
                padded_layers = [ZeroPadding1D(padding = (0, max_len - layer.shape[1]))(layer) for layer in concatenated_layers]

                x = Concatenate(axis = -1)(padded_layers)


    if attention_type == 'channel':
        x = ChannelAttention()(x)
    elif attention_type == 'spatial':
        x = SpatialAttention()(x)
    elif attention_type == 'cbam':
        x = CBAM()(x)
    elif attention_type is None:
        pass


    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    for hidden_unit in hidden_units_list:
        x = Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun)(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
    model = Model(inputs = inputs, outputs = outputs)

    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model



def build_TCN_Attention_model(model_parameters_dict, attention_type):

    num_filter_list = model_parameters_dict['num_filter_list']
    kernel_size_list = model_parameters_dict['kernel_size_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)


    inputs = Input(shape = (timesteps, num_features))
    x = inputs

    for i in range(len(num_filter_list)):
        x = TCN(nb_filters = num_filter_list[i], kernel_size = kernel_size_list[i], dilations = [2 ** i], kernel_initializer = kernel_init, return_sequences = True)(x)


    if attention_type == 'channel':
        x = ChannelAttention()(x)
    elif attention_type == 'spatial':
        x = SpatialAttention()(x)
    elif attention_type == 'cbam':
        x = CBAM()(x)
    elif attention_type is None:
        pass

    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)

    for hidden_unit in hidden_units_list:
        x = Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun)(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init)(x)
    model = Model(inputs = inputs, outputs = outputs)


    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model



def build_RNN_model(model_types, layer_direction, mode, model_parameters_dict):


    if model_types == 'RNN':
        layer_type = SimpleRNN
    elif model_types == 'LSTM':
        layer_type = LSTM
    elif model_types == 'GRU':
        layer_type = GRU
    else:
        raise ValueError('Invalid model type!')

    units_list = model_parameters_dict['units_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    recurrent_init = get_initializer(model_parameters_dict.get('recurrent_init', 'orthogonal'))
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)

    model = Sequential()
    if layer_direction  == 'unidirectional':
        for i in range(len(units_list)):
            if i == 0:
                model.add(layer_type(units = units_list[i], return_sequences = True, input_shape = (timesteps, num_features),
                                     kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init))
            else:
                model.add(layer_type(units = units_list[i], return_sequences = True,
                                     kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init))
    elif layer_direction == 'bidirectional':
        print('The connection mode is', mode)
        for i in range(len(units_list)):
            if i == 0:
                model.add(Bidirectional(layer_type(units = units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init),
                                        input_shape = (timesteps, num_features), merge_mode = mode ))
            else:
                model.add(Bidirectional(layer_type(units = units_list[i], return_sequences = True,  kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init),
                                        merge_mode = mode ))
    else:
        raise ValueError('Invalid layer_direction!')

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for hidden_unit in hidden_units_list:
        model.add(Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))
        model.add(Dropout(dropout_rate))
    model.add(Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init))

    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)



def build_RNN_Attention_model(model_types, layer_direction, mode, model_parameters_dict):

    if model_types == 'RNN':
        layer_type = SimpleRNN
    elif model_types == 'LSTM':
        layer_type = LSTM
    elif model_types == 'GRU':
        layer_type = GRU
    else:
        raise ValueError('Invalid model type!')

    units_list = model_parameters_dict['units_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    recurrent_init = get_initializer(model_parameters_dict.get('recurrent_init', 'orthogonal'))
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)


    inputs = Input(shape = (timesteps, num_features))
    x = inputs
    if layer_direction == 'unidirectional':
        for i in range(len(units_list)):
            x = layer_type(units = units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init)(x)
    elif layer_direction == 'bidirectional':
        print('The connection mode is', mode)
        for i in range(len(units_list)):
            x = Bidirectional(layer_type(units = units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init), merge_mode = mode)(x)
    else:
        raise ValueError('Invalid layer_direction!')

    x = Attention()([x, x])
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)

    for hidden_unit in hidden_units_list:
        x = Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init,  activation = activation_fun)(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init)(x)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(loss=loss_fun, optimizer=opt, metrics=metric)
    return model






def build_CNN_RNN_model(model_types, layer_direction, mode, model_parameters_dict):

    if model_types == 'RNN':
        layer_type = SimpleRNN
    elif model_types == 'LSTM':
        layer_type = LSTM
    elif model_types == 'GRU':
        layer_type = GRU
    else:
        raise ValueError('Invalid model type!')

    num_filter_list = model_parameters_dict['num_filter_list']
    kernel_size_list = model_parameters_dict['kernel_size_list']
    pool_size_list = model_parameters_dict['pool_size_list']
    rnn_units_list = model_parameters_dict['rnn_units_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    recurrent_init = get_initializer(model_parameters_dict.get('recurrent_init', 'orthogonal'))
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)

    model = Sequential()

    for i in range(len(num_filter_list)):
        if i == 0:
            model.add(Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], input_shape = (timesteps, num_features),
                             kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))
            model.add(MaxPooling1D(pool_size = pool_size_list[i]))
        else:
            model.add(Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))
            model.add(MaxPooling1D(pool_size = pool_size_list[i]))


    if layer_direction == 'unidirectional':
        for i in range(len(rnn_units_list)):
            model.add(layer_type(units = rnn_units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init))
    elif layer_direction == 'bidirectional':
        print('The connection mode is', mode)
        for i in range(len(rnn_units_list)):
            model.add(Bidirectional(layer_type(units = rnn_units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init,
                                               bias_initializer = bias_init), merge_mode = mode))
    else:
        raise ValueError('Invalid layer_direction!')

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    for hidden_unit in hidden_units_list:
        model.add(Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun))
        model.add(Dropout(dropout_rate))
    model.add(Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init))

    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model



def build_CNN_RNN_Attention_model(model_types, layer_direction, mode, model_parameters_dict):

    if model_types == 'RNN':
        layer_type = SimpleRNN
    elif model_types == 'LSTM':
        layer_type = LSTM
    elif model_types == 'GRU':
        layer_type = GRU
    else:
        raise ValueError('Invalid model type!')

    num_filter_list = model_parameters_dict['num_filter_list']
    kernel_size_list = model_parameters_dict['kernel_size_list']
    pool_size_list = model_parameters_dict['pool_size_list']
    rnn_units_list = model_parameters_dict['rnn_units_list']
    dropout_rate = model_parameters_dict['dropout_rate']
    hidden_units_list = model_parameters_dict['hidden_units_list']
    output_units = model_parameters_dict['output_units']
    recurrent_init = get_initializer(model_parameters_dict.get('recurrent_init', 'orthogonal'))
    timesteps, num_features, kernel_init, bias_init, activation_fun, opt, metric, loss_fun = get_common_parameters(model_parameters_dict)


    inputs = Input(shape = (timesteps, num_features))
    x = inputs


    for i in range(len(num_filter_list)):
        x = Conv1D(filters = num_filter_list[i], kernel_size = kernel_size_list[i], kernel_initializer = kernel_init, bias_initializer = bias_init, activation=activation_fun)(x)
        x = MaxPooling1D(pool_size = pool_size_list[i])(x)


    if layer_direction == 'unidirectional':
        for i in range(len(rnn_units_list)):
            x = layer_type(units = rnn_units_list[i], return_sequences = True, kernel_initializer = kernel_init, recurrent_initializer = recurrent_init, bias_initializer = bias_init)(x)
    elif layer_direction == 'bidirectional':
        print('The connection mode is', mode)
        for i in range(len(rnn_units_list)):
            x = Bidirectional(layer_type(units = rnn_units_list[i], return_sequences = True, kernel_initializer = kernel_init,
                                         recurrent_initializer = recurrent_init, bias_initializer = bias_init), merge_mode = mode)(x)
    else:
        raise ValueError('Invalid layer_direction!')

    x = Attention()([x, x])
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)

    for hidden_unit in hidden_units_list:
        x = Dense(units = hidden_unit, kernel_initializer = kernel_init, bias_initializer = bias_init, activation = activation_fun)(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(units = output_units, kernel_initializer = kernel_init, bias_initializer = bias_init)(x)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(loss = loss_fun, optimizer = opt, metrics = metric)
    return model




class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, trainable=True, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.trainable = trainable
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.shared_mlp = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=self.filters // self.ratio, kernel_size=1, activation='relu'),
            tf.keras.layers.Conv1D(filters=self.filters, kernel_size=1)
        ])
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avgout = tf.reduce_mean(inputs, axis=1, keepdims=True)
        maxout = tf.reduce_max(inputs, axis=1, keepdims=True)
        avgout = self.shared_mlp(avgout)
        maxout = self.shared_mlp(maxout)
        return tf.sigmoid(avgout + maxout)

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'trainable': self.trainable, 'ratio': self.ratio})
        return config

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super(SpatialAttention, self).__init__(trainable=trainable, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avgout = tf.reduce_mean(inputs, axis= -1, keepdims=True)
        maxout = tf.reduce_max( inputs, axis= -1, keepdims=True)
        concat = tf.concat([avgout, maxout], axis=-1)
        attention = self.conv1(concat)
        return inputs * attention

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        return config

class CBAM(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        return config


class TemporalBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate = 1, padding = 'causal', dropout_rate = 0.0, **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.dropout_rate = dropout_rate


        self.conv1 = Conv1D(filters, kernel_size, dilation_rate = dilation_rate, padding = padding)
        self.bn1 = BatchNormalization()
        self.activation1 = Activation('relu')
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = Conv1D(filters, kernel_size, dilation_rate = dilation_rate, padding = padding)
        self.bn2 = BatchNormalization()
        self.activation2 = Activation('relu')
        self.dropout2 = Dropout(dropout_rate)


        self.downsample = Conv1D(filters, 1) if dilation_rate > 1 else None
        self.activation3 = Activation('relu')

    def build(self, input_shape):

        if input_shape[-1] != self.filters:
            self.input_adjust = Conv1D(self.filters, 1, padding = 'same')
        else:
            self.input_adjust = None

    def call(self, inputs, training=None):
        y = self.conv1(inputs)
        y = self.bn1(y, training=training)
        y = self.activation1(y)
        y = self.dropout1(y, training=training)

        y = self.conv2(y)
        y = self.bn2(y, training=training)
        y = self.activation2(y)
        y = self.dropout2(y, training=training)


        if self.downsample is not None:
            inputs = self.downsample(inputs)


        if self.input_adjust is not None:
            inputs = self.input_adjust(inputs)

        y = Add()([y, inputs])
        y = self.activation3(y)
        return y


    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            "filters": self.conv1.filters,
            "kernel_size": self.conv1.kernel_size,
            "dilation_rate": self.conv1.dilation_rate,
            "padding": self.conv1.padding,
            "dropout_rate": self.dropout1.rate
        })
        return config