
import numpy as np
import pandas as pd


from keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax


def create_dataset(data_x, data_y, look_back, look_forward, delay):

    batch_size = determine_batch_size(data_x)

    data_gen = data_generator(data_x, data_y, look_back, look_forward, delay, batch_size)

    num_samples = len(data_x) - look_back - look_forward - delay + 1
    num_batches = (num_samples + batch_size - 1) // batch_size

    X_combined, Y_combined = combine_batches(data_gen, num_batches)

    return X_combined, Y_combined



def transform_data_shape(data_x, data_y):


    num_samples, timesteps, num_features = data_x.shape
    X = data_x.reshape(num_samples, -1)

    num_sample, prediction_steps, features = data_y.shape
    Y = data_y.reshape(num_sample, -1)
    Y_flatten = Y.flatten()
    return X, Y


def transform_combine_data_shape(data_x_list, data_y_list):


    X_list = []
    Y_list = []

    for data_x, data_y in zip(data_x_list, data_y_list):
        X, Y = transform_data_shape(data_x, data_y)
        X_list.append(X)
        Y_list.append(Y)

    return X_list, Y_list




def split_data(data_x, data_y, test_size):

    train_size = data_x.shape[0] - test_size

    train_x = data_x[0:train_size, :]
    test_x = data_x[train_size:, :]

    last_x = data_x[-1:, :]


    train_y = data_y[0:train_size, :]
    test_y = data_y[train_size:, :]

    data_X_dict = {'train_data': train_x, 'test_data': test_x, 'last_data': last_x }
    data_Y_dict = {'train_data': train_y, 'test_data': test_y}

    return data_X_dict, data_Y_dict




def split_and_combine_data(data_x_list, data_y_list, test_size, concat_axis):

    combined_X_dict = {'train_data': [], 'test_data': [], 'last_data': []}
    combined_Y_dict = {'train_data': [], 'test_data': []}

    for data_x, data_y in zip(data_x_list, data_y_list):
        data_X_dict, data_Y_dict = split_data(data_x, data_y, test_size)

        for key in combined_X_dict:
            if data_X_dict[key] is not None:
                combined_X_dict[key].append(data_X_dict[key])

        for key in combined_Y_dict:
            if data_Y_dict[key] is not None:
                combined_Y_dict[key].append(data_Y_dict[key])


    for key in combined_X_dict:
        combined_X_dict[key] = np.concatenate(combined_X_dict[key], axis=concat_axis) if combined_X_dict[key] else None


    for key in combined_Y_dict:
        combined_Y_dict[key] = np.concatenate(combined_Y_dict[key], axis=0) if combined_Y_dict[key] else None


    return combined_X_dict, combined_Y_dict


def normalize_data(data_dict, normalize_type):

    scalers_mapping = {
        'MinMax': MinMaxScaler(),
        'Standard': StandardScaler(),
        'MaxAbs': MaxAbsScaler(),
        'Robust': RobustScaler(),
        'Norm': Normalizer(),
        'Quantile': QuantileTransformer(),
        'Power': PowerTransformer()
    }

    if normalize_type not in scalers_mapping:
        raise ValueError('Invalid normalization type!')


    scaler = scalers_mapping[normalize_type]

    norm_data_dict = {}


    train_data = data_dict['train_data']
    if train_data.size > 0:
        train_data_norm = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1))
        train_data_norm = train_data_norm.reshape(train_data.shape)
    else:
        train_data_norm = None
    norm_data_dict['train_data'] = train_data_norm


    test_data = data_dict.get('test_data')
    if test_data is not None and test_data.size > 0:
        test_data_norm = scaler.transform(test_data.reshape(test_data.shape[0], -1))
        test_data_norm = test_data_norm.reshape(test_data.shape)
    else:
        test_data_norm = None
    norm_data_dict['test_data'] = test_data_norm


    if 'last_data' in data_dict:
        last_data = data_dict['last_data']
        last_data_norm = scaler.transform(last_data.reshape(last_data.shape[0], -1))
        last_data_norm = last_data_norm.reshape(last_data.shape)
        norm_data_dict['last_data'] = last_data_norm
    else:
        norm_data_dict['last_data'] = None

    return norm_data_dict, scaler



def inverse_normalize_data(norm_data_dict, scaler):


    inv_norm_data_dict = {}


    train_data_norm = norm_data_dict['train_data']
    if train_data_norm is not None and train_data_norm.size > 0:
        train_data = scaler.inverse_transform(train_data_norm.reshape(train_data_norm.shape[0], -1))
        train_data = train_data.reshape(train_data_norm.shape)
    else:
        train_data = None
    inv_norm_data_dict['train_data'] = train_data



    test_data_norm = norm_data_dict.get('test_data')
    if test_data_norm is not None and test_data_norm.size > 0:
        test_data = scaler.inverse_transform(test_data_norm.reshape(test_data_norm.shape[0], -1))
        test_data = test_data.reshape(test_data_norm.shape)
    else:
        test_data = None
    inv_norm_data_dict['test_data'] = test_data


    last_data_norm = norm_data_dict.get('last_data')
    if last_data_norm is not None and last_data_norm.size > 0:
        last_data = scaler.inverse_transform(last_data_norm.reshape(last_data_norm.shape[0], -1))
        last_data = last_data.reshape(last_data_norm.shape)
    else:
        last_data = None
    inv_norm_data_dict['last_data'] = last_data

    return inv_norm_data_dict



def evaluation_predict(predict_y_dict, real_y_dict):

    eval_dict = {}


    eval_dict['train'] = calculate_scores(real_y_dict['train_data'], predict_y_dict['train_data'])


    if 'test_data' in predict_y_dict:
        eval_dict['test'] = calculate_scores(real_y_dict['test_data'], predict_y_dict['test_data'])
    else:
        eval_dict['test'] = None
    return eval_dict



def calculate_scores(real, predicted):

    if real.ndim == 3:
        real = real.reshape(real.shape[0], -1)
        predicted = predicted.reshape(predicted.shape[0], -1)


    scores_dict = {
        'RMSE': [],
        'MSE': [],
        'R2_score': [],
        'MAE': [],
        'MAPE': [],
        'SMAPE': [],
        'Huber_loss': [],
        'residuals': []
    }


    for i in range(real.shape[1]):
        step_real = real[:, i]
        step_predicted = predicted[:, i]

        residual = step_real - step_predicted
        scores_dict['residuals'].append(residual)

        scores_dict['RMSE'].append(np.sqrt(mean_squared_error(step_real, step_predicted)))
        scores_dict['MSE'].append(mean_squared_error(step_real, step_predicted))
        scores_dict['R2_score'].append(r2_score(step_real, step_predicted))
        scores_dict['MAE'].append(mean_absolute_error(step_real, step_predicted))


        mape = np.mean(np.abs((step_real - step_predicted) / np.where(step_real != 0, step_real, np.nan))) * 100
        scores_dict['MAPE'].append(mape)

        smape = 2.0 * np.mean(np.abs(step_predicted - step_real) / (np.abs(step_predicted) + np.abs(step_real) + 1e-10)) * 100
        scores_dict['SMAPE'].append(smape)

        huber_loss_fn = Huber(delta=1.0)
        huber_loss_value = huber_loss_fn(step_real, step_predicted)
        scores_dict['Huber_loss'].append(huber_loss_value.numpy())


    mean_scores_dict = {metric: np.mean(scores_dict[metric]) for metric in scores_dict}
    return mean_scores_dict




def dimensionality_reduction(data_property, norm_x_dict, num_components, reduced_method, valid_prob):

    train_data_shape = norm_x_dict['train_data'].shape
    if len(train_data_shape) == 3:
        return dimensionality_reduction_with_3D(data_property, norm_x_dict, num_components, reduced_method, valid_prob)
    elif len(train_data_shape) == 2:
        return dimensionality_reduction_with_2D(data_property, norm_x_dict, num_components, reduced_method, valid_prob)
    else:
        raise ValueError(f'Unsupported data dimensions {train_data_shape}. Expected 2D or 3D arrays.')



def dimensionality_reduction_with_2D(data_property, norm_x_dict, num_components, reduced_method, valid_prob):

    key_list = ['last_data', 'test_data']
    train_data_x = norm_x_dict['train_data']
    data_reduced_dict = {}

    if reduced_method == 'PCA':
        pca = PCA(n_components=num_components)
        data_reduced_dict['train_data'] = pca.fit_transform(train_data_x)
        for key in  key_list:
            if key in norm_x_dict:
                data_reduced_dict[key] = pca.transform(norm_x_dict[key])

    elif reduced_method == 'Autoencoders':
        input_dim = train_data_x.shape[1]
        data_shape = '2D'
        model_parameters_dict = get_dim_model_param_dict(data_property, data_shape)
        epochs_ = model_parameters_dict.get('epochs')
        batch_size_ = model_parameters_dict.get('batch_size')

        num_predictions = 10
        predictions = []

        for _ in range(num_predictions):
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            autoencoder, encoder = build_autoencoders_model(input_dim, num_components, model_parameters_dict, data_shape)
            autoencoder.fit(train_data_x, train_data_x, epochs=epochs_, batch_size=batch_size_,
                            validation_split=valid_prob, callbacks=[early_stopping], shuffle=True, verbose=0)


            reduced_data = encoder.predict(train_data_x, verbose=0)
            predictions.append(reduced_data)


        data_reduced_dict['train_data'] = np.mean(predictions, axis=0)

        for key in key_list:
            if key in norm_x_dict:
                key_data = norm_x_dict[key]
                if key_data.ndim != 2:
                    raise ValueError(f"{key} must be a 2D array with shape [num_samples, num_features].")

                reduced_key_data = []
                for _ in range(num_predictions):
                    reduced_key_data.append(encoder.predict(key_data, verbose=0))
                data_reduced_dict[key] = np.mean(reduced_key_data, axis=0)

    else:
        raise ValueError('Invalid reduction method specified. Choose from "PCA" or "Autoencoders".')

    return data_reduced_dict



def dimensionality_reduction_with_3D(data_property, norm_x_dict, num_components, reduced_method, valid_prob):

    key_list = ['last_data', 'test_data']

    train_data_x = norm_x_dict['train_data']
    if train_data_x.ndim != 3:
        raise ValueError("Training data must be a 3D array with shape [num_samples, look_back, num_features].")
    num_samples, look_back, num_features = train_data_x.shape

    data_reduced_dict = {}

    if reduced_method == 'PCA':

        reshaped_train_data = train_data_x.reshape(-1, num_features)
        pca = PCA(n_components=num_components)
        reduced_data = pca.fit_transform(reshaped_train_data)

        data_reduced_dict['train_data'] = reduced_data.reshape(num_samples, look_back, num_components)

        for key in  key_list:
            if key in norm_x_dict:
                key_data = norm_x_dict[key]
                num_samples_key, look_back_key, num_features_key = key_data.shape
                reshaped_key_data = key_data.reshape(-1, num_features_key)
                reduced_key_data = pca.transform(reshaped_key_data)
                data_reduced_dict[key] = reduced_key_data.reshape(num_samples_key, look_back_key, num_components)

    elif reduced_method == 'Autoencoders':
        data_shape = '3D'
        model_parameters_dict = get_dim_model_param_dict(data_property, data_shape)
        epochs_ = model_parameters_dict.get('epochs')
        batch_size_ = model_parameters_dict.get('batch_size')

        input_dim = (look_back, num_features)


        num_predictions = 1
        predictions = []
        for _ in range(num_predictions):
            autoencoder, encoder = build_autoencoders_model(input_dim, num_components, model_parameters_dict, data_shape)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


            autoencoder.fit(train_data_x, train_data_x, epochs=epochs_, batch_size=batch_size_, validation_split=valid_prob, callbacks=[early_stopping], shuffle=True, verbose=0)


            reduced_data = encoder.predict(train_data_x, verbose=0)
            predictions.append(reduced_data)


        avg_reduced_data = np.mean(predictions, axis=0)
        data_reduced_dict['train_data'] = avg_reduced_data

        for key in key_list:
            if key in norm_x_dict:
                key_data = norm_x_dict[key]
                if key_data.ndim != 3:
                    raise ValueError(f"{key} must be a 3D array with shape [num_samples, look_back, num_features].")


                reduced_key_data = []
                for _ in range(num_predictions):
                    reduced_key_data.append(encoder.predict(key_data, verbose=0))


                data_reduced_dict[key] = np.mean(reduced_key_data, axis=0)

    else:
        raise ValueError('Invalid reduction method specified. Choose from "PCA" or "Autoencoders".')

    return data_reduced_dict


def get_dim_model_param_dict(data_property, data_shape):

    if data_property == 'return':
        if data_shape == '2D':
            model_parameters_dict = {
                'optimizer': 'Adam',
                'activation_fun': 'relu',
                'encoded_units_list': [256, 128],
                'dropout_rate': 0.40,
                'learning_rate': 0.001,
                'epochs': 10,
                'batch_size': 336}
        elif data_shape == '3D':
            model_parameters_dict = {
                'optimizer': 'Adamax',
                'activation_fun': 'tanh',
                'kernel_size_list': [4, 3],
                'dropout_rate': 0.45,
                'learning_rate': 0.02,
                'epochs': 160,
                'batch_size': 112}
        else:
            raise ValueError("data_shape must be either '2D' or '3D'.")

    elif data_property == 'ESG':
        if data_shape == '2D':
            model_parameters_dict = {
                'optimizer': 'Adam',
                'activation_fun': 'relu',
                'encoded_units_list': [112, 56],
                'dropout_rate': 0.15,
                'learning_rate': 0.00001,
                'epochs': 30,
                'batch_size': 416}
        elif data_shape == '3D':
            model_parameters_dict = {
                'optimizer': 'RMSprop',
                'activation_fun': 'selu',
                'kernel_size_list': [3, 3, 3],
                'dropout_rate': 0.40,
                'learning_rate': 0.002,
                'epochs': 150,
                'batch_size': 32}
        else:
            raise ValueError("data_shape must be either '2D' or '3D'.")

    return model_parameters_dict



def build_autoencoders_model(input_dim, encoding_dim, model_parameters_dict, data_shape):

    optimizer = model_parameters_dict.get('optimizer', 'Adam')
    activation_fun = model_parameters_dict.get('activation_fun', 'relu')
    dropout_rate = model_parameters_dict.get('dropout_rate')
    lr = model_parameters_dict.get('learning_rate')


    if data_shape not in ['2D', '3D']:
        raise ValueError("data_shape must be either '2D' or '3D'.")

    if data_shape == '2D':
        encoded_units_list = model_parameters_dict.get('encoded_units_list')
        input_layer = Input(shape=(input_dim,))


        encoded = input_layer
        for units in encoded_units_list:
            encoded = Dense(units, activation=activation_fun)(encoded)
            encoded = Dropout(dropout_rate)(encoded)
        encoded = Dense(encoding_dim, activation=activation_fun)(encoded)


        decoded = encoded
        decoded_units_list = list(reversed(encoded_units_list))
        for units in decoded_units_list:
            decoded = Dense(units, activation=activation_fun)(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

    elif data_shape == '3D':

        input_layer = Input(shape=(input_dim[0], input_dim[1]))

        kernel_size_list = model_parameters_dict.get('kernel_size_list')

        encoded = input_layer
        for i in range(len(kernel_size_list)):
            encoded = Conv1D(filters=encoding_dim, kernel_size=kernel_size_list[i], padding='same', activation=activation_fun)(encoded)
            encoded = MaxPooling1D(pool_size=1)(encoded)
            encoded = Dropout(dropout_rate)(encoded)


        decoded = encoded
        for i in range(len(kernel_size_list)):
            decoded = UpSampling1D(size=1)(decoded)
            decoded = Conv1D(filters=input_dim[1], kernel_size=kernel_size_list[-(i + 1)], padding='same', activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)


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



    huber_loss = Huber(delta=1.0)
    autoencoder.compile(optimizer=opt, loss=huber_loss)
    return autoencoder, encoder




def determine_batch_size(data_x, max_batch_size=1024):

    num_samples = len(data_x)
    sample_size = np.prod(data_x.shape[1:])


    estimated_memory_per_sample = sample_size * 4
    estimated_memory_per_batch = estimated_memory_per_sample * max_batch_size


    if estimated_memory_per_batch > 2 * 1024**3:
        while estimated_memory_per_batch > 2 * 1024**3 and max_batch_size > 1:
            max_batch_size //= 2
            estimated_memory_per_batch = estimated_memory_per_sample * max_batch_size

    return max_batch_size





def data_generator(data_x, data_y, look_back, look_forward, delay, batch_size):

    num_samples = len(data_x) - look_back - look_forward - delay + 1

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = range(start, end)

        X_batch = np.empty((len(batch_indices), look_back, data_x.shape[1]), dtype=np.float32)
        Y_batch = np.empty((len(batch_indices), look_forward, data_y.shape[1]), dtype=np.float32)

        for i, idx in enumerate(batch_indices):
            X_batch[i] = data_x[idx:(idx + look_back), :]
            Y_batch[i] = data_y[(idx + delay + look_back):(idx + delay + look_back + look_forward), :]

        yield X_batch, Y_batch





def combine_batches(data_gen, num_batches):

    X_list = []
    Y_list = []

    for _ in range(num_batches):
        X_batch, Y_batch = next(data_gen)
        X_list.append(X_batch)
        Y_list.append(Y_batch)

    X_combined = np.concatenate(X_list, axis=0)
    Y_combined = np.concatenate(Y_list, axis=0)

    return X_combined, Y_combined


