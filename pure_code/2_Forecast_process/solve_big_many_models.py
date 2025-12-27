
import numpy as np
import os
import datetime
from scipy.stats import norm, multivariate_normal




def remove_model(models_dict, remove_model_prob):

    model_list = np.array(models_dict['model_list'])
    model_age_list = np.array(models_dict['model_age_list'])
    post_weight_list = np.array(models_dict['post_weights_list'])

    remove_model_num = max(round(len(model_list) * remove_model_prob), 1)


    survival_model_list = []
    survival_model_age_list = []
    survival_weight_list = []

    death_model_list = []
    death_model_age_list = []
    death_weight_list = []


    zero_weight_indices = np.where(post_weight_list == 0)[0]
    for index in zero_weight_indices:
        death_model_list.append(model_list[index])
        death_model_age_list.append(model_age_list[index])
        death_weight_list.append(0)


    non_zero_weight_indices = np.where(post_weight_list > 0)[0]
    for index in non_zero_weight_indices:
        survival_model_list.append(model_list[index])
        survival_model_age_list.append(model_age_list[index])
        survival_weight_list.append(post_weight_list[index])



    remaining_removals = remove_model_num - len(death_model_list)
    if remaining_removals > 0:

        sorted_indices = np.lexsort((-np.array(survival_model_age_list), np.array(survival_weight_list)))


        for index in sorted_indices[:remaining_removals]:
            death_model_list.append(survival_model_list[index])
            death_model_age_list.append(survival_model_age_list[index])
            death_weight_list.append(0)


        survival_model_list = [model for i, model in enumerate(survival_model_list) if i not in sorted_indices[:remaining_removals]]
        survival_model_age_list = [age for i, age in enumerate(survival_model_age_list) if i not in sorted_indices[:remaining_removals]]
        survival_weight_list = [weight for i, weight in enumerate(survival_weight_list) if i not in sorted_indices[:remaining_removals]]



    survival_weight_list = np.array(survival_weight_list)
    survival_weight_sum = np.sum(survival_weight_list)
    normalized_survival_weights = survival_weight_list / survival_weight_sum if survival_weight_sum > 0 else np.zeros(len(survival_weight_list))


    survival_model_dict = {
        'model_list': [],
        'model_age_list': [],
        'weights_list': [],
        'indices': []
    }

    death_model_dict = {
        'model_list': [],
        'model_age_list': [],
        'weights_list': [],
        'indices': []
    }


    for i in range(len(model_list)):
        if model_list[i] in survival_model_list:
            index = survival_model_list.index(model_list[i])
            survival_model_dict['model_list'].append(model_list[i])
            survival_model_dict['model_age_list'].append(model_age_list[i])
            survival_model_dict['weights_list'].append(normalized_survival_weights[index])
            survival_model_dict['indices'].append(i)
        else:
            death_model_dict['model_list'].append(model_list[i])
            death_model_dict['model_age_list'].append(model_age_list[i])
            death_model_dict['weights_list'].append(0)
            death_model_dict['indices'].append(i)

    return survival_model_dict, death_model_dict






def add_model_update_probability(models_dict, add_model_list, current_time, nu):

    model_list = np.array(models_dict['model_list'])
    model_age_list = np.array(models_dict['model_age_list'])
    weight_list = np.array(models_dict['weights_list'])
    add_model_list = np.array(add_model_list)

    pi = 0
    for i in range(len(model_list)):
        pi += (1 / (current_time + 1 - model_age_list[i] )) * weight_list[i]

    psi = calculate_psi(models_dict, add_model_list, nu)

    updated_weight_list = (1 - pi) * weight_list
    updated_age_list = model_age_list + 1


    new_model_weights = pi * psi
    new_model_ages = np.zeros(len(add_model_list))


    combined_model_list = np.concatenate((model_list, add_model_list))
    combined_age_list = np.concatenate((updated_age_list, new_model_ages))
    combined_weight_list = np.concatenate((updated_weight_list, new_model_weights))

    combined_weight_list = np.clip(combined_weight_list, 0, None)


    total_weight = np.sum(combined_weight_list)
    if total_weight > 0:
        combined_weight_list = combined_weight_list / total_weight


    updated_models_dict = {
        'model_list': combined_model_list.tolist(),
        'model_age_list': combined_age_list.tolist(),
        'weights_list': combined_weight_list.tolist()}

    return updated_models_dict




def calculate_psi(models_dict, add_model_list, nu):

    existing_model_list = np.array(models_dict['model_list'])
    existing_weight_list = np.array(models_dict['weights_list'])
    add_model_num = len(add_model_list)

    psi = np.zeros(add_model_num)


    for i, new_model in enumerate(add_model_list):

        same_model_indices = np.where(existing_model_list == new_model)[0]


        if same_model_indices.size > 0:
            psi[i] = np.sum(existing_weight_list[same_model_indices])

    epsilon = 1e-6
    bar_psi_vector = (nu / (add_model_num + epsilon)) + ( 1 - nu) * psi
    return bar_psi_vector






def random_select_models(model_list, model_num):

    model_list = np.array(model_list)
    selected_models = np.random.choice(model_list, size=model_num, replace=False)
    return selected_models.tolist()




def convert_to_log_probabilities(prob):

    prob = np.array(prob)



    prob = np.where(prob == 0, 1e-10, prob)

    log_prob = np.log(prob)
    return log_prob


def convert_to_probabilities(log_prob):

    log_prob = np.array(log_prob)
    prob = np.exp(log_prob)
    prob /= np.sum(prob)
    return prob






















