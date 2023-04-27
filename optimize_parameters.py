# This script is used to optimize the parameters of the keyword selection
# algorithm. It uses the CMA-ES algorithm to find the optimal parameters.
# The parameters are saved in the config/params.yaml file.

import cma
from cma import transformations
import numpy as np
import pandas as pd
# from ruamel.yaml import YAML
import yaml
from main import process_messages
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from multiprocessing import freeze_support

# run file it as: python3 optimize_parameters.py -Xfrozen_modules=off

# Set the number of cores to use for multiprocessing
n_cores = 7

config_path = "config/params.yaml"

with open(config_path, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    return df["freq"].to_dict()

# preloading the word frequency dictionary
word_freq_dict = load_word_freq_dict()

# Define the objective function
def run_process_messages(x):
    # Update the parameters with the new values from the optimization
    ksp = parameters['keyword_selection_parameters']
    ksp['spacy_keywords_weight'] = str(x[0])
    ksp['rake_keywords_weight'] = str(x[1])
    ksp['tf_IDF_keywords_weight'] = str(x[2])
    ksp['LDA_keywords_weight'] = str(x[3])
    ksp['NMF_keywords_weight'] = str(x[4])
    ksp['frequency_threshold1'] = str(round(x[5]))
    ksp['frequency_threshold2'] = str(round(x[6]))
    ksp['frequency_weight1'] = str(x[7])
    ksp['frequency_weight2'] = str(x[8])
    ksp['digit_weight'] = str(x[9])
    ksp['highest_rank'] = str(round(x[10]))
    ksp['rank_weight'] = str(x[11])
    ksp['position_weight'] = str(x[12])
    ksp['position_ratio_weight'] = str(x[13])
    parameters['spacy']['LOC'] = str(round(x[14]))
    parameters['spacy']['ORG'] = str(round(x[15]))
    parameters['spacy']['MISC'] = str(round(x[16]))
    parameters['rake']['max_length'] = str(round(x[17]))
    parameters['tf_IDF']['min_keywords'] = str(round(x[18]))
    parameters['tf_IDF']['max_keywords'] = str(round(x[19]))
    parameters['tf_IDF']['keywords_multiplier'] = str(x[20])
    parameters['LDA']['num_topics_multiplier'] = str(x[21])
    parameters['LDA']['passes'] = str(round(x[22]))
    parameters['NMF']['num_topics_multiplier'] = str(x[23])

    # Convert dictionary to JSON object
    #json_obj = json.dumps(parameters)

    # Convert JSON object to YAML and save to file
    # Save the updated parameters back to the YAML file without removing comments
    with open(config_path, "w") as file:
        yaml.dump(parameters, file)

    # Call the process_messages function
    performance = process_messages(word_freq_dict)
    print("-performance['common_topics']", -performance['common_topics'])

    # Return the performance score for optimization
    return -performance['common_topics']

# Set the initial values and bounds for the optimization
initial_values = [
    parameters['keyword_selection_parameters']['spacy_keywords_weight'],
    parameters['keyword_selection_parameters']['rake_keywords_weight'],
    parameters['keyword_selection_parameters']['tf_IDF_keywords_weight'],
    parameters['keyword_selection_parameters']['LDA_keywords_weight'],
    parameters['keyword_selection_parameters']['NMF_keywords_weight'],
    parameters['keyword_selection_parameters']['frequency_threshold1'],
    parameters['keyword_selection_parameters']['frequency_threshold2'],
    parameters['keyword_selection_parameters']['frequency_weight1'],
    parameters['keyword_selection_parameters']['frequency_weight2'],
    parameters['keyword_selection_parameters']['digit_weight'],
    parameters['keyword_selection_parameters']['highest_rank'],
    parameters['keyword_selection_parameters']['rank_weight'],
    parameters['keyword_selection_parameters']['position_weight'],
    parameters['keyword_selection_parameters']['position_ratio_weight'],
    parameters['spacy']['LOC'],
    parameters['spacy']['ORG'],
    parameters['spacy']['MISC'],
    parameters['rake']['max_length'],
    parameters['tf_IDF']['min_keywords'],
    parameters['tf_IDF']['max_keywords'],
    parameters['tf_IDF']['keywords_multiplier'],
    parameters['LDA']['num_topics_multiplier'],
    parameters['LDA']['passes'],
    parameters['NMF']['num_topics_multiplier']
]


lower_bounds = [
    0, 0, 0, 0, 0, 0.4, 1.0, -300, -300, -500, 10, 0, 0, 0, 
    0, 0, 0, 1, 1, 1, 0, 0, 5, 0
]

upper_bounds = [
    20, 20, 20, 20, 20, 1.0, 1.5, 0, -50, 0, 50, 5, 500, 5, 
    1, 1, 1, 5, 5, 10, 2.0, 10, 30, 10
]

myoptions = {
    "bounds": [lower_bounds, upper_bounds],
    "popsize": 16,
    "verb_disp": 100,
    "tolx": 1e-6,
    "tolfun": 1e-4,
    "maxiter": 100000,
}

# Set the initial standard deviation for the optimization
sigma0 = 0.5

# Call the CMA-ES optimization function with transformations
#initial_values = [float(val) for val in initial_values]
es = cma.fmin(run_process_messages, initial_values, sigma0, options=myoptions)

# if __name__ == '__main__':
#     freeze_support()
#     es = cma.CMAEvolutionStrategy(initial_values, sigma0, myoptions)
#     # Initialize the multiprocessing pool
#     pool = mp.Pool(n_cores)
#     while not es.stop():
#         X = es.ask()
#         f_values = pool.map_async(run_process_messages, X).get()
#         # use chunksize parameter as es.popsize/len(pool)?
#         es.tell(X, f_values)
#         es.disp()
#         es.logger.add()
#         es.logger.disp()

    # print('termination:', es.stop())
    # cma.pprint(es.best.__dict__)
    # print("Optimized parameters:", es.result.xbest)
    # Close the multiprocessing pool
    # pool.close()
    # pool.join()

