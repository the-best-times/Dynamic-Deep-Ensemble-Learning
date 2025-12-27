
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import warnings
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


from process_time_series_data import create_dataset, split_and_combine_data, normalize_data, transform_combine_data_shape



def create_predict_param_dict(basis_param_dict, code_item, data_property):

    path_basis_dict = basis_param_dict['path_basis_dict']
    original_file_path = path_basis_dict['original_file_path']
    data_folder_path = path_basis_dict['data_folder_path']

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    predict_basis_param_dict = basis_param_dict['predict_basis_param_dict']

    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    look_forward = structural_basis_param_dict['look_forward']
    hyperparameter_dict = predict_basis_param_dict['hyperparameter_dict']
    hyperparams = hyperparameter_dict[data_property]

    predict_param_dict_dict = {}


    X_Y_dict_dict = process_sample_data(original_file_path, data_folder_path, date_split_node_list, look_forward, code_item, data_property)

    for period_item, data_dict in X_Y_dict_dict.items():
        data_parameters_dict = create_data_parameters_dict(data_dict, structural_basis_param_dict, hyperparams)

        predict_param_dict_dict[period_item] = {
            'ML_model_list': predict_basis_param_dict.get('ML_model_list', []),
            'DML_model_list': predict_basis_param_dict.get('DML_model_list', []),
            'plot': predict_basis_param_dict.get('plot_train', False),
            'dim_reduction': predict_basis_param_dict.get('dim_reduction', False),
            'reduced_method': predict_basis_param_dict.get('reduced_method', None),
            'data_parameters': data_parameters_dict}

    return predict_param_dict_dict



def create_data_parameters_dict(original_data_dict, structural_basis_param_dict, hyperparams):

    look_back = structural_basis_param_dict['look_back']
    delay = structural_basis_param_dict['delay']
    look_forward = structural_basis_param_dict['look_forward']


    original_data_x = original_data_dict['data_x']
    original_data_y = original_data_dict['data_y'].reshape(-1, 1)


    current_data_x, current_data_y = create_dataset(original_data_x, original_data_y, look_back, look_forward, delay)


    current_observed_data = current_data_y[-1:]
    observed_data = current_observed_data.reshape(-1)

    previous_observed_data = current_data_y[-2: -1]
    previous_observed_data = previous_observed_data.reshape(-1)
    previous_data = np.array([previous_observed_data[0]])




    data_parameters_dict = {
        'data_x':  current_data_x,
        'data_y': current_data_y,
        'observed_data': observed_data,
        'previous_data': previous_data,

        'test_size': structural_basis_param_dict['test_size'],
        'valid_prob': structural_basis_param_dict['valid_prob'],
        'epoch': hyperparams.get('epoch'),
        'batch_sizes': hyperparams.get('batch_sizes'),
        'num_components': hyperparams.get('num_components') }

    return data_parameters_dict



def process_sample_data(original_file_path, folder_path, date_split_node_list, look_forward, code_item, data_property):

    extended_date_split_node_list = generate_extended_date_split_node_list(original_file_path, date_split_node_list, look_forward)
    data_with_periods_dict_dict = load_and_split_data(folder_path, extended_date_split_node_list, code_item)

    X_Y_dict_dict = {}

    period_dict = data_with_periods_dict_dict[data_property]
    for period_item, df in period_dict.items():
        data_x, data_y = determine_target_and_features_value(df, data_property)
        X_Y_dict_dict[period_item] = {'data_x': data_x, 'data_y': data_y}
    return X_Y_dict_dict




def determine_target_and_features_value(df, data_property):

    df = df.copy()

    if '交易日期' in df.columns:
        df.drop(columns=['交易日期'], inplace=True)

    if data_property == 'return':
        if '百分比回报' not in df.columns:
            raise KeyError("'百分比回报' column is missing from the DataFrame.")
        data_y = df['百分比回报'].values
        data_x = df.drop(columns=['百分比回报']).values

    elif data_property == 'ESG':
        if 'ESG总分' not in df.columns:
            raise KeyError("'ESG总分' column is missing from the DataFrame.")
        data_y = df['ESG总分'].values
        data_x = df.drop(columns=['ESG总分']).values

    else:
        raise ValueError("data_property must be one of 'return' or 'ESG'.")

    return data_x, data_y



def determine_features_target(df, data_property):

    df = df.copy()
    if '交易日期' in df.columns:
        df.drop(columns=['交易日期'], inplace=True)

    if data_property == 'return':
        if '百分比回报' not in df.columns:
            raise KeyError("'百分比回报' column is missing from the DataFrame.")
        data_x = df.drop(columns=['百分比回报'])
        data_y = df[['百分比回报']]

    elif data_property == 'ESG':
        if 'ESG总分' not in df.columns:
            raise KeyError("'ESG总分' column is missing from the DataFrame.")
        data_x = df.drop(columns=['ESG总分'])
        data_y = df[['ESG总分']]
    else:
        raise ValueError("data_property must be one of 'return' or 'ESG'.")

    return data_x, data_y

# ------------------------------------ load and creat stocks data df -------------------------------------------------

def load_and_split_data(folder_path, date_split_node_list, code_item):

    data_dict = {}
    try:
        if 'market' in code_item:
            file_path = os.path.join(folder_path, 'market_portfolio.xlsx')
            return_df = pd.read_excel(file_path, sheet_name='return')
            if '超额回报' in return_df.columns:
                return_df.drop(columns=['超额回报'], inplace=True)
            data_dict['return'] = return_df

        else:
            file_path = os.path.join(folder_path, f'{code_item}.xlsx')
            return_df = pd.read_excel(file_path, sheet_name='return')
            if '超额回报' in return_df.columns:
                return_df.drop(columns=['超额回报'], inplace=True)
            data_dict['return'] = return_df

            ESG_df = pd.read_excel(file_path, sheet_name='ESG')
            data_dict['ESG'] = ESG_df


        data_with_periods_dict_dict = {}
        data_with_periods_dict_dict['return'] = split_df_with_periods(data_dict['return'], date_split_node_list)
        if 'ESG' in data_dict:
            data_with_periods_dict_dict['ESG'] = split_df_with_periods(data_dict['ESG'], date_split_node_list)

    except FileNotFoundError:
        print(f'Error: The file for {code_item} could not be found.')
    except pd.errors.ExcelFileNotFoundError:
        print(f'Error: The sheet in the Excel file for {code_item} could not be found.')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

    return data_with_periods_dict_dict



def split_df_with_periods(df, date_split_node_list):

    df['交易日期'] = pd.to_datetime(df['交易日期']).dt.date
    df_dict = {}

    start_date = pd.to_datetime(date_split_node_list[0]).date()

    for period_item in range(1, len(date_split_node_list)):
        end_date = pd.to_datetime(date_split_node_list[period_item]).date()
        df_dict[f'periods_{period_item - 1}'] = df[(df['交易日期'] >= start_date) & (df['交易日期'] < end_date)]

    return df_dict




def generate_extended_date_split_node_list(original_file_path, observation_date_split_node_list, look_forward):


    observation_date_split_node_list = [pd.to_datetime(date).date() for date in observation_date_split_node_list]

    extended_date_split_node_list = []
    extended_date_split_node_list.append(observation_date_split_node_list[0])
    observation_num = len(observation_date_split_node_list)

    trading_dates_df = pd.read_excel(original_file_path, sheet_name='return')
    trading_dates_df['交易日期'] = pd.to_datetime(trading_dates_df['交易日期']).dt.date

    for i in range(1, observation_num):
        date_node = observation_date_split_node_list[i]


        if date_node not in trading_dates_df['交易日期'].values:
            later_dates = trading_dates_df[trading_dates_df['交易日期'] > date_node]
            if not later_dates.empty:
                date_node = later_dates['交易日期'].iloc[0]
            else:
                raise ValueError(f"No later trading date found for {date_node}")


        date_index = trading_dates_df[trading_dates_df['交易日期'] == date_node].index[0]
        if date_index + look_forward < len(trading_dates_df):
            new_date_node = trading_dates_df.iloc[date_index + look_forward]['交易日期']
        else:
            new_date_node = trading_dates_df.iloc[-1]['交易日期']

        extended_date_split_node_list.append(new_date_node)

    extended_date_split_node_list = [date.strftime('%Y-%m-%d') for date in extended_date_split_node_list]
    return extended_date_split_node_list



def load_historical_data(folder_path, date_split_node_list, code_item):


    def load_and_filter_data(file_path, sheet_name_, date_col, target_col, start_date, end_date):

        df = pd.read_excel(file_path, sheet_name=sheet_name_)
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df_filtered = df[(df[date_col] >= start_date) & (df[date_col] < end_date)]
        if target_col not in df_filtered.columns:
            raise KeyError(f"'{target_col}' column is missing from the {sheet_name_} sheet.")
        return df_filtered[target_col].values


    historical_data_dict = {}
    start_date = pd.to_datetime(date_split_node_list[0]).date()
    end_date = pd.to_datetime(date_split_node_list[-1]).date()

    try:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Error: The folder path '{folder_path}' does not exist.")

        file_path = os.path.join(folder_path, 'market_portfolio.xlsx' if 'market' in code_item else f'{code_item}.xlsx')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' for {code_item} could not be found.")


        historical_data_dict['return'] = load_and_filter_data(file_path, 'return', '交易日期', '百分比回报', start_date, end_date)

        if 'market' not in code_item:
            historical_data_dict['ESG'] = load_and_filter_data(file_path, 'ESG', '交易日期', 'ESG总分', start_date, end_date)

    except FileNotFoundError as e:
        print(e)
        return None
    except KeyError as e:
        print(f"KeyError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    return historical_data_dict



def load_list_from_text(folder_path, file_name):

    file_path = os.path.join(folder_path, file_name)
    code_list = []
    with open(file_path, 'r') as file:
        for line in file:
            code_list.append(line.strip())
    return code_list




def report_column_name(folder_path, code_list):

    columns_dict = {}

    market_file_path = os.path.join(folder_path, 'market_portfolio.xlsx')
    try:
        market_df = pd.read_excel(market_file_path, sheet_name='return')
        if '超额回报' in market_df.columns:
            market_df.drop(columns=['超额回报'], inplace=True)
        columns_dict['market_portfolio'] = market_df.columns.tolist()
    except Exception as e:
        print(f'Error loading market data: {e}')
        columns_dict['market_portfolio'] = []

    if code_list:
        first_code = code_list[0]
        stock_file_path = os.path.join(folder_path, f'stock_{first_code}.xlsx')
        try:
            stock_return_df = pd.read_excel(stock_file_path, sheet_name='return')
            if '超额回报' in stock_return_df.columns:
                stock_return_df.drop(columns=['超额回报'], inplace=True)
            columns_dict['stock_return'] = stock_return_df.columns.tolist()


            stock_ESG_df = pd.read_excel(stock_file_path, sheet_name='ESG')
            columns_dict['stock_ESG'] = stock_ESG_df.columns.tolist()
        except Exception as e:
            print(f'Error loading stock data for code {first_code}: {e}')
            columns_dict['stock_return'] = []
            columns_dict['stock_ESG'] = []


    parent_dir = os.path.dirname(folder_path)
    columns_file_path = os.path.join(parent_dir, 'columns_name_report.txt')
    with open(columns_file_path, 'w') as file:
        for key, columns in columns_dict.items():
            file.write(f"{key}:\n")
            for column in columns:
                file.write(f"  - {column}\n")
            file.write("\n")

    print(f'Column names have been written!')








def generate_ML_predict_params(basis_param_dict, code_list, data_property):

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    total_periods_num = len(date_split_node_list) - 1


    ML_predict_param_dict_dict_dict = {code_item: {} for code_item in code_list}

    for code_item in code_list:
        predict_param_dict_dict = create_predict_param_dict(basis_param_dict, code_item, data_property)


        for t in range(total_periods_num):
            period_key = f'periods_{t}'
            if period_key in predict_param_dict_dict:
                ML_predict_param_dict_dict_dict[code_item][period_key] = predict_param_dict_dict[period_key]

    return ML_predict_param_dict_dict_dict


def extract_ML_predict_params_by_period(ML_predict_param_dict_dict_dict, period_item):

    period_item = str(period_item)


    parameters_by_period_dict = {}

    for code_item, params_dict in ML_predict_param_dict_dict_dict.items():
        period_key = f'periods_{period_item}'
        if period_key in params_dict:
            parameters_by_period_dict[code_item] = params_dict[period_key]


    ML_predict_param_dict_market = None
    ML_predict_param_dict_list = []


    for key, ML_predict_param_dict in parameters_by_period_dict.items():
        if 'market' in key:
            ML_predict_param_dict_market = ML_predict_param_dict
        else:
            ML_predict_param_dict_list.append(ML_predict_param_dict)

    return ML_predict_param_dict_market, ML_predict_param_dict_list




def generate_statistics(basis_param_dict, code_list, plot_code_list):


    def calculate_statistics(df):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        statistics = {
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            '25%': df.quantile(0.25),
            '50%': df.quantile(0.50),
            '75%': df.quantile(0.75),
            'max': df.max(),
            'nums': df.shape[0]
        }
        return pd.DataFrame(statistics)

    def calculate_correlation(df):

        if df.shape[1] > 1:
            return df.corr(method='pearson')
        return pd.DataFrame()


    def plot_correlation_heatmap(data, data_property, period_item, saved_dir):

        df = pd.DataFrame(data)
        correlation_matrix = df.corr(method='pearson')

        plt.figure(figsize=(10, 8))


        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linewidths=0.5)


        num = period_item.split('_')[-1]
        number = str(int(num) + 1)
        plt.title(f'Correlation with {data_property} in period {number}', fontsize=16)
        plt.xticks(ticks=np.arange(len(correlation_matrix.columns)) + 0.5, labels=correlation_matrix.columns, rotation=80, fontsize=11)
        plt.yticks(ticks=np.arange(len(correlation_matrix.index)) + 0.5, labels=correlation_matrix.index, rotation=0, fontsize=11)
        plt.tight_layout()


        plot_file_dir = os.path.join(saved_dir, 'heatmap', f'{data_property}_{period_item}')
        os.makedirs(plot_file_dir, exist_ok=True)

        png_filename = os.path.join(plot_file_dir, 'correlation_heatmap.png')
        pdf_filename = os.path.join(plot_file_dir, 'correlation_heatmap.pdf')


        plt.savefig(png_filename, dpi=600, format='png')
        plt.savefig(pdf_filename, dpi=600, format='pdf')

        plt.close()


    def plot_histogram(data_y, data_property, period_item, saved_dir, bins):

        plt.figure(figsize=(10, 6))
        plt.hist(data_y, bins=bins, color='blue', alpha=0.6, edgecolor='black')

        if data_property == 'return':
            x_label = 'Return'
        elif data_property == 'ESG':
            x_label = 'ESG score'


        num = period_item.split('_')[-1]
        number = str(int(num) + 1)
        plt.title(f'Distribution of {x_label} in period {number}', fontsize=16)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.tight_layout()


        plot_file_dir = os.path.join(saved_dir, 'histogram', f'{data_property}_{period_item}')
        os.makedirs(plot_file_dir, exist_ok=True)
        png_filename = os.path.join(plot_file_dir, 'histogram.png')
        pdf_filename = os.path.join(plot_file_dir, 'histogram.pdf')


        plt.savefig(png_filename, dpi=600, format='png')
        plt.savefig(pdf_filename, dpi=600, format='pdf')


        plt.close()


    path_basis_dict = basis_param_dict['path_basis_dict']
    data_folder_path = path_basis_dict['data_folder_path']
    results_saved_folder_statistics = path_basis_dict['results_saved_folder_statistics']

    structural_basis_param_dict = basis_param_dict['structural_basis_param_dict']
    date_split_node_list = structural_basis_param_dict['date_split_node_list']
    data_property_list = structural_basis_param_dict['data_property_list']

    total_items_num = len(code_list)


    combined_data_dict = {data_property: {} for data_property in data_property_list}
    all_stats_dict = {data_property: {} for data_property in data_property_list}

    with tqdm(total=total_items_num, desc='Calculating statistical info', unit='asset') as pbar:
        for code_item in code_list:
            data_with_periods_dict_dict = load_and_split_data(data_folder_path, date_split_node_list, code_item)

            saved_dir = os.path.join(results_saved_folder_statistics, code_item)
            os.makedirs(saved_dir, exist_ok=True)

            for data_property in data_property_list:
                period_dict = data_with_periods_dict_dict[data_property]


                file_path = os.path.join(saved_dir, f'{data_property}_result.xlsx')
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    for period_item, df in period_dict.items():

                        if period_item not in combined_data_dict[data_property]:
                            combined_data_dict[data_property][period_item] = []
                            all_stats_dict[data_property][period_item] = []
                        new_column_names = rename_col(code_item, data_property)
                        data_x, data_y = determine_features_target(df, data_property)


                        combined_df = pd.concat([data_y, data_x], axis=1)
                        combined_df = combined_df.rename(columns=new_column_names)


                        combined_data_dict[data_property][period_item].append(combined_df)



                        stats = calculate_statistics(combined_df)
                        correlations = calculate_correlation(combined_df)

                        all_stats_dict[data_property][period_item].append(stats)


                        stats.to_excel(writer, sheet_name=f'stats_{period_item}')


                        if code_item in plot_code_list:
                            plot_correlation_heatmap(combined_df, data_property, period_item, saved_dir)
                            plot_histogram(data_y, data_property, period_item, saved_dir, bins=50)

                pbar.update(1)


    for data_property, period_dict in all_stats_dict.items():
        mean_file_path = os.path.join(results_saved_folder_statistics, f'mean_{data_property}.xlsx')
        with pd.ExcelWriter(mean_file_path, engine='xlsxwriter') as writer:
            for period_item, stats_list in period_dict.items():
                if stats_list:
                    mean_stats_df = pd.DataFrame()
                    for column in stats_list[0].columns:
                        column_means = pd.concat([df[column] for df in stats_list], axis=1).mean(axis=1)
                        mean_stats_df[column] = column_means


                    mean_stats_df.to_excel(writer, sheet_name=f'mean_stats_{period_item}', index=True)






def rename_col(code_item, data_property):

    new_column_names = {}


    if 'market' in code_item:
        new_column_names = {
            '百分比回报': 'Return',
            '开盘指数': 'OPI',
            '最高指数': 'HPI',
            '最低指数': 'LPI',
            '收盘指数': 'CPI',
            '指数样本股成交金额(亿元)': 'ISTV',
            '指数流通股本(亿股)': 'ISSO',
            '指数样本股组合流通市值(亿元)': 'ICMV',
            '市场风险溢价因子(总市值加权)_五因子': 'RP',
            '市值因子(总市值加权)_五因子': 'SMB',
            '账面市值比因子(总市值加权)_五因子': 'HML',
            '盈利能力因子(总市值加权)_五因子': 'RMW',
            '投资模式因子(总市值加权)_五因子': 'CMA',
            '市场规模_上市股票数': 'N_Stocks',
            '市场规模_流通市值(亿元)': 'MCV',
            '市场规模_成交金额(亿元)': 'MTV',
            '人民币元对美元汇率': 'USD/CNY',
            '居民消费价格指数': 'CPI',
            '宏观经济景气指数': 'Macro',
            '国内生产总值(万亿元)': 'GDP',
            '经济政策不确定性': 'EPU'   }

    else:
        if data_property == 'return':
            new_column_names = {
                '百分比回报': 'Return',
                '开盘价': 'OP',
                '收盘价': 'CP',
                '最高价': 'HP',
                '最低价': 'LP',
                '换手率': 'TR',
                '总股本(亿股)': 'TS',
                '流通股本(亿股)': 'CS',
                '市盈率': 'PE',
                '市净率': 'PB',
                '市销率': 'PS',
                '流动性指标': 'Liquidity',
                '一周最高价(后复权)': 'WHP',
                '一月最高价(后复权)': 'MHP',
                '一周最低价(后复权)': 'WLP',
                '一月最低价(后复权)': 'MLP',
                '股票价值得分': 'OVS',
                '股票成长得分': 'OGS',
                'RSI': 'RSI',
                'KDJ': 'KDJ',
                'MACD': 'MACD',
                '机构持股比例': 'Holding_Prop',
                '人民币元对美元汇率': 'USD/CNY',
                '居民消费价格指数': 'CPI',
                '宏观经济景气指数': 'Macro',
                '国内生产总值(万亿元)': 'GDP',
                '经济政策不确定性': 'EPU',
                'log_连涨天数': 'Log_UD',
                'log_连跌天数': 'Log_DD',
                'log_连续放量天数': 'Log_UVD',
                'log_连续缩量天数': 'Log_DVD',
                'log_新闻数量': 'Log_News',
                'log_被分析师关注数': 'Log_Analyst',
                'log_被研报关注数': 'Log_Report'  }
        elif data_property == 'ESG':
            new_column_names = {
                'ESG总分': 'ESG_Score',
                '治理维度得分': 'G_Score',
                '社会维度得分': 'S_Score',
                '环境维度得分': 'E_Score',
                'ESG争议事件得分': 'Controversy_Score',
                'ESG管理实践得分': 'Manage_Score',
                'log_客户及消费者权益保护_披露数': 'Log_DCCRP',
                'log_职工权益保护_披露数': 'Log_DERP',
                'log_安全生产内容_披露数': 'Log_DSP',
                'log_供应商权益保护_披露数': 'Log_DSRP',
                'log_股东权益保护_披露数': 'Log_DSHRP',
                'log_公共关系和社会公益事业_披露数': 'Log_DPRSW',
                'log_环境绩效_披露数': 'Log_EPD',
                'log_资源消耗_披露数': 'Log_RCD',
                'log_环境排放物_披露数': 'Log_EED',
                '机构持股比例': 'Holding_Prop',
                '人民币元对美元汇率': 'USD/CNY',
                '居民消费价格指数': 'CPI',
                '宏观经济景气指数': 'Macro',
                '国内生产总值(万亿元)': 'GDP',
                '经济政策不确定性': 'EPU',
                'log_新闻数量': 'Log_News',
                'log_被分析师关注数': 'Log_Analyst',
                'log_被研报关注数': 'Log_Report' }

    return new_column_names



