
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tcn import TCN
from tensorflow.keras.initializers import GlorotUniform, HeNormal, LecunNormal
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from joblib import dump, load
from keras.models import save_model, load_model
import threading


save_lock = threading.Lock()
load_lock = threading.Lock()



from process_time_series_data import transform_data_shape, transform_combine_data_shape,  split_data, split_and_combine_data,normalize_data, \
    inverse_normalize_data, evaluation_predict, dimensionality_reduction
from ML_algorithm import ChannelAttention, SpatialAttention, CBAM, TemporalBlock, build_ML_model, build_deep_ML_model, DML_model_type_to_category
from DML_structure_param import get_parameters



def get_predict_data(data_property, model_type, model_dir, ML_predict_param_dict, Train):

    ML_model_list = ML_predict_param_dict['ML_model_list']
    DML_model_list = ML_predict_param_dict['DML_model_list']
    plot = ML_predict_param_dict['plot']
    dim_reduction = ML_predict_param_dict['dim_reduction']
    reduced_method = ML_predict_param_dict['reduced_method']
    data_parameters_dict = ML_predict_param_dict['data_parameters']


    test_size = data_parameters_dict['test_size']
    valid_prob = data_parameters_dict['valid_prob']
    data_x = data_parameters_dict['data_x']
    data_y = data_parameters_dict['data_y']
    epoch = data_parameters_dict.get('epoch')
    batch_sizes = data_parameters_dict.get('batch_sizes')
    num_components = data_parameters_dict.get('num_components')

    if model_type in ML_model_list:
        data_x, data_y = transform_data_shape(data_x, data_y)
    elif model_type in DML_model_list:
        pass
    else:
        raise ValueError('Invalid model type!')

    print('The current model type is:', model_type)


    data_X, data_Y = split_data(data_x, data_y, test_size)

    predict_y_dict, real_y_dict, eval_dict = prediction_process(data_property, data_X, data_Y, dim_reduction, num_components, reduced_method,
                                                                Train, model_type, model_dir, ML_model_list, DML_model_list, plot, epoch, batch_sizes, valid_prob)


    return predict_y_dict, real_y_dict, eval_dict




def get_predict_data_multi_stock_task(data_property, model_type, model_dir, ML_predict_param_dict_list, Train):

    ML_model_list = ML_predict_param_dict_list[0]['ML_model_list']
    DML_model_list = ML_predict_param_dict_list[0]['DML_model_list']
    plot = ML_predict_param_dict_list[0]['plot']
    dim_reduction = ML_predict_param_dict_list[0]['dim_reduction']
    reduced_method = ML_predict_param_dict_list[0]['reduced_method']
    data_parameters_dict_ = ML_predict_param_dict_list[0]['data_parameters']


    test_size = data_parameters_dict_['test_size']
    valid_prob = data_parameters_dict_['valid_prob']
    epoch = data_parameters_dict_.get('epoch')
    batch_sizes = data_parameters_dict_.get('batch_sizes')
    num_components = data_parameters_dict_.get('num_components')


    num_assets = len(ML_predict_param_dict_list)
    data_x_list, data_y_list = [], []
    for i in range(num_assets):
        data_parameters_dict = ML_predict_param_dict_list[i]['data_parameters']
        data_x_list.append(data_parameters_dict['data_x'])
        data_y_list.append(data_parameters_dict['data_y'])

    if model_type in ML_model_list:
        data_x_list, data_y_list = transform_combine_data_shape(data_x_list, data_y_list)
    elif model_type in DML_model_list:
        pass
    else:
        raise ValueError('Invalid model type!')

    print('The current model type is:', model_type)



    concat_axis = 0
    data_X, data_Y = split_and_combine_data(data_x_list, data_y_list, test_size, concat_axis)

    predict_y_dict, real_y_dict, eval_dict = prediction_process(data_property, data_X, data_Y, dim_reduction, num_components, reduced_method,
                                                                Train, model_type, model_dir, ML_model_list, DML_model_list, plot, epoch, batch_sizes, valid_prob)

    return predict_y_dict, real_y_dict, eval_dict




def prediction_process(data_property, data_X, data_Y, dim_reduction, num_components, reduced_method, Train,
                       model_type, model_dir, ML_model_list, DML_model_list, plot, epoch, batch_sizes, valid_prob):


    def predict_(model, data, model_type, ML_model_list, DML_model_list):

        if model_type in ML_model_list:
            return model.predict(data)
        elif model_type in DML_model_list:
            return model.predict(data, verbose=0)
        else:
            raise ValueError(f"Unrecognized model type: {model_type}. Must be one of {ML_model_list} or {DML_model_list}.")


    norm_x_dict, scaler_x = normalize_data(data_X, normalize_type='MinMax')
    norm_y_dict, scaler_y = normalize_data(data_Y, normalize_type='MinMax')


    if dim_reduction and num_components and reduced_method:
        norm_x_dict = dimensionality_reduction(data_property, norm_x_dict, num_components, reduced_method, valid_prob)



    model = None
    if model_type in ML_model_list:
        if Train:
            model = build_ML_model(model_type)
            X_train, X_valid, y_train, y_valid = train_test_split(norm_x_dict['train_data'], norm_y_dict['train_data'], test_size=valid_prob, random_state=42)
            his = model.fit(X_train, y_train)
            save_model_to_file(model, model_type, model_dir, ML_model_list, DML_model_list)
        else:
            model = load_model_from_file(model_type, model_dir, ML_model_list, DML_model_list, custom_objects=None)

    elif model_type in DML_model_list:

        data_x_ = norm_x_dict['train_data']
        data_y_ = norm_y_dict['train_data']
        model_parameters_dict = get_parameters(data_property, model_type, data_x_, data_y_)

        if Train:
            model = build_deep_ML_model(model_type, model_parameters_dict)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            his = model.fit(norm_x_dict['train_data'], norm_y_dict['train_data'], epochs=epoch, batch_size=batch_sizes,
                            validation_split=valid_prob, callbacks=[early_stopping], shuffle=True, verbose=0)
            save_model_to_file(model, model_type, model_dir, ML_model_list, DML_model_list)
        else:

            custom_objects_dict = {
                'kernel_initializer': model_parameters_dict.get('kernel_init'),
                'bias_initializer': model_parameters_dict.get('bias_init'),
                'activation': model_parameters_dict.get('activation_fun'),
                'optimizer': model_parameters_dict.get('optimizer'),
                'recurrent_initializer': model_parameters_dict.get('recurrent_init') if 'recurrent_init' in model_parameters_dict else None,
                'ChannelAttention': ChannelAttention, 'SpatialAttention': SpatialAttention, 'CBAM': CBAM, 'TCN': TCN}

            model = load_model_from_file(model_type, model_dir, ML_model_list, DML_model_list, custom_objects=custom_objects_dict)


    num_predictions = 10
    predict_norm_y_list = []

    for _ in range(num_predictions):
        try:
            predict_norm_y_dict = {
                'train_data': predict_(model, norm_x_dict['train_data'], model_type, ML_model_list, DML_model_list),
                'last_data':  predict_(model, norm_x_dict['last_data'], model_type, ML_model_list, DML_model_list),
                'test_data':  predict_(model, norm_x_dict['test_data'], model_type, ML_model_list, DML_model_list) }
            predict_norm_y_list.append(predict_norm_y_dict)
        except ValueError as e:
            print(f"Error during prediction: {e}")


    avg_predict_norm_y_dict = {
        'train_data': np.mean([pred['train_data'] for pred in predict_norm_y_list], axis=0),
        'last_data': np.mean([pred['last_data'] for pred in predict_norm_y_list], axis=0),
        'test_data': np.mean([pred['test_data'] for pred in predict_norm_y_list], axis=0) }


    predict_y_dict = inverse_normalize_data(avg_predict_norm_y_dict, scaler_y)
    real_y_dict = inverse_normalize_data(norm_y_dict, scaler_y)


    if Train and plot:
        plot_learning_process(predict_y_dict, real_y_dict, his)


    eval_dict = evaluation_predict(predict_y_dict, real_y_dict)

    return predict_y_dict, real_y_dict, eval_dict






def save_model_to_file(model, model_type, model_dir, ML_model_list, DML_model_list):

    if model_type in ML_model_list:

        ext = 'joblib'
    elif model_type in DML_model_list:
        ext = 'h5'
    else:
        raise ValueError('Invalid model type!')

    os.makedirs(model_dir, exist_ok = True)
    model_filename = os.path.join(model_dir, f'model_{model_type}.{ext}')

    with save_lock:
        if os.path.exists(model_filename):
            os.remove(model_filename)

        if model_type in ML_model_list:
            dump(model, model_filename)
        elif model_type in DML_model_list:
            model.save(model_filename, save_format='h5', overwrite=True)


    print(f'The model has been saved!')



def load_model_from_file(model_type, model_dir, ML_model_list, DML_model_list, custom_objects=None):


    if model_type in ML_model_list:

        ext = 'joblib'
        load_func = load
    elif model_type in DML_model_list:

        ext = 'h5'
        load_func = load_model
    else:
        raise ValueError('Invalid model type!')

    model_filename = os.path.join(model_dir, f'model_{model_type}.{ext}')


    if custom_objects is None:
        custom_objects = {}


    if model_type in DML_model_list:
        model_category = DML_model_type_to_category(model_type)
        if model_category == 'TCN_category':

            custom_objects['GlorotUniform'] = GlorotUniform
            custom_objects['HeNormal'] = HeNormal
            custom_objects['LecunNormal'] = LecunNormal


    if os.path.exists(model_filename):
        with load_lock:
            if model_type in ML_model_list:
                model = load_func(model_filename)
            else:
                model = load_func(model_filename, custom_objects=custom_objects)

            print(f'The model has been loaded!')
            return model


    else:
        raise FileNotFoundError(f'Model file {model_filename} not found!')


def plot_learning_process(predict_y_dict, real_y_dict, his):

    plt.figure(figsize=(24, 8))



    if 'loss' in his.history and 'val_loss' in his.history:

        plt.subplot(211)
        plt.plot(his.history['loss'][0:], label = 'train_loss')

        plt.grid(axis='y')
        plt.legend()
    else:
        print(" Training history does not contain 'loss' and 'val_loss'. ")




    plt.subplot(212)
    plt.grid(axis='x')
    if 'valid_data' in real_y_dict:
        real_combined = [i for i in real_y_dict['train_data'].reshape(-1)] + [i for i in real_y_dict['valid_data'].reshape(-1)] + [i for i in real_y_dict['test_data'].reshape(-1)]
        predict_combined = [i for i in predict_y_dict['train_data'].reshape(-1)] + [i for i in predict_y_dict['valid_data'].reshape(-1)] + [i for i in predict_y_dict['test_data'].reshape(-1)]
    else:
        real_combined = [i for i in real_y_dict['train_data'].reshape(-1)] + [i for i in real_y_dict['test_data'].reshape(-1)]
        predict_combined = [i for i in predict_y_dict['train_data'].reshape(-1)] + [i for i in predict_y_dict['test_data'].reshape(-1)]

    plt.plot(real_combined, '-o', label = 'real Y')
    plt.plot(predict_combined, '-o', label = 'predict Y')
    plt.legend()
    plt.show()







