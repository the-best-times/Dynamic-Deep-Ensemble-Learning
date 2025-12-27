
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_1samp



current_working_dir = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(current_working_dir))


results_saved_folder_path = os.path.join(parent_dir, 'Data_and_Results', 'Results', 'Part_asset_construction')
data_property_list = ['return', 'ESG']
save_excel_path = os.path.join(results_saved_folder_path, 't_test_result.xlsx')






def significance_star(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''


def t_test_with_diff_model(save_excel_path, results_saved_folder_path):

    target_metrics = ['MSE', 'RMSE', 'RMSPE', 'MAE', 'MAPE', 'SMAPE']
    model_dirs = {
        "DDEL": os.path.join(results_saved_folder_path,
                             'forward_info_pure_DML',
                             'nodes_5_models_10_removeProb_0.2'),
        "DSEL": os.path.join(results_saved_folder_path,
                             'forward_info_pure_ML',
                             'nodes_5_models_10_removeProb_0.2'),
        "SL": os.path.join(results_saved_folder_path,
                           'forward_info_pure_DML',
                           'nodes_5_models_1_removeProb_0.2'),
        "SO": os.path.join(results_saved_folder_path,
                           'forward_info_pure_DML',
                           'nodes_1_models_10_removeProb_0.2'),
    }

    with pd.ExcelWriter(save_excel_path, engine='xlsxwriter') as writer:

        for data_property in data_property_list:

            if data_property == 'return':
                sheet_base = 'Return Metrics'
            elif data_property == 'ESG':
                sheet_base = 'ESG Metrics'
            else:
                continue


            df_ddel = pd.read_excel(
                os.path.join(model_dirs["DDEL"], 'eval_forward_info.xlsx'),
                sheet_name=sheet_base
            ).set_index('Code')

            metric_names = [m for m in target_metrics if m in df_ddel.columns]


            for baseline in ["DSEL", "SL", "SO"]:

                df_base = pd.read_excel(
                    os.path.join(model_dirs[baseline], 'eval_forward_info.xlsx'),
                    sheet_name=sheet_base
                ).set_index('Code')

                rows = []

                for metric in metric_names:
                    diff = df_base[metric] - df_ddel[metric]
                    diff = diff.dropna()

                    t_stat, p_value = ttest_1samp(diff.values, 0.0, alternative='greater')

                    rows.append({
                        'Metric': metric,
                        'Mean Diff': diff.mean(),
                        't-stat': t_stat,
                        'p-value': p_value,
                        'Significance': significance_star(p_value),
                        'N': diff.shape[0]
                    })

                result_df = pd.DataFrame(rows)

                sheet_name = f"{data_property}_DDEL_vs_{baseline}"
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)


    print(f"T-test results saved to: {save_excel_path}")



t_test_with_diff_model(save_excel_path, results_saved_folder_path)