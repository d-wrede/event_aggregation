# This script is used to optimize the parameters of the keyword selection
# algorithm. It uses the CMA-ES algorithm to find the optimal parameters.
# The parameters are saved in the config/params.yaml file.

import cma
import os
#from cma import transformations
import numpy as np
import pandas as pd
import spacy
# from ruamel.yaml import YAML
import yaml
import json
from main import process_messages
import time
import multiprocessing as mp
from multiprocessing import freeze_support
import cProfile
import pstats

profile_filename = "profile_results.prof"
cProfile_switch = False

# run file as: python3 optimize_parameters.py -Xfrozen_modules=off

# Set the number of cores to use for multiprocessing
n_cores = 16

config_path = "config/params.yaml"
with open(config_path, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    print("Word frequency dictionary loaded in optimize_parameters.py")
    return df["freq"].to_dict()

# preloading the word frequency dictionary
word_freq_dict = load_word_freq_dict()

json_filename = "chosen_topics.json"
with open(json_filename, "r", encoding="utf-8") as f:
    messages = json.load(f)

nlp_spacy = spacy.load("de_core_news_lg", disable=["parser", "tagger"])


# Define the objective function
def run_process_messages(x):
    """Run the process_messages function with the given parameters"""
    
    # Update the parameters with the new values from the optimization
    ksp = parameters['keyword_selection_parameters']
    ksp['spacy_keywords_weight'] = x[0]
    ksp['rake_keywords_weight'] = x[1]
    ksp['tf_IDF_keywords_weight'] = x[2]
    ksp['LDA_keywords_weight'] = x[3]
    ksp['NMF_keywords_weight'] = x[4]
    ksp['frequency_threshold1'] = round(x[5])
    ksp['frequency_threshold2'] = round(x[6])
    ksp['frequency_weight1'] = x[7]
    ksp['frequency_weight2'] = x[8]
    ksp['digit_weight'] = x[9]
    ksp['highest_rank'] = round(x[10])
    ksp['rank_weight'] = x[11]
    ksp['position_weight'] = x[12]
    ksp['position_ratio_weight'] = x[13]
    parameters['spacy']['LOC'] = round(x[14])
    parameters['spacy']['ORG'] = round(x[15])
    parameters['spacy']['MISC'] = round(x[16])
    parameters['rake']['max_length'] = round(x[17])
    parameters['tf_IDF']['min_keywords'] = round(x[18])
    parameters['tf_IDF']['max_keywords'] = round(x[19])
    parameters['tf_IDF']['keywords_multiplier'] = x[20]
    parameters['LDA']['num_topics_multiplier'] = x[21]
    parameters['LDA']['passes'] = round(x[22])
    parameters['NMF']['num_topics_multiplier'] = x[23]
    parameters['NMF']['max_iter'] = round(x[24])

    # Convert dictionary to JSON object
    #json_obj = json.dumps(parameters)

    # Convert JSON object to YAML and save to file
    # Save the updated parameters back to the YAML file without removing comments
    # with open(config_path, "w") as file:
    #     yaml.dump(parameters, file)

    start_time = time.time()

    # Call the process_messages function
    performance = process_messages(word_freq_dict, parameters, messages, nlp_spacy)
    end_time = time.time()
    perf_score = performance['common_topics'] - (end_time - start_time) * 50

    print("performance['common_topics']", performance['common_topics'])
    print("iteration time: ", end_time - start_time)
    print("combined perf_score: ", perf_score)
    # Return the performance score for optimization
    return -perf_score

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
    parameters['NMF']['num_topics_multiplier'],
    parameters['NMF']['max_iter']
]


lower_bounds = [
    0, 0, 0, 0, 0, 0.4, 1.0, -300, -300, -500, 10, 0, 0, 0, 
    0, 0, 0, 1, 1, 1, 0, 0, 5, 0, 1
]

upper_bounds = [
    20, 20, 20, 20, 20, 1.0, 1.5, 0, -50, 0, 50, 5, 500, 5, 
    1, 1, 1, 5, 5, 10, 2.0, 10, 30, 2, 200
]
# Set the index positions of the integer parameters in the input vector
idx_integers = [10, 14, 15, 16, 17, 18, 19, 22, 24]

# Convert the parameters to floats
initial_values = [float(val) for val in initial_values]

myoptions = {
    "bounds": [lower_bounds, upper_bounds],
    "popsize": 15,
    "verb_disp": 100,
    "tolx": 1e-6,
    "tolfun": 1e-4,
    "maxiter": 100000,
}

# Set the initial standard deviation for the optimization
sigma0 = 0.5

if cProfile_switch:
    # Run the function with profiling and save the results to a file
    cProfile.run("run_process_messages(initial_values)", filename=profile_filename)

    # Load the results from the file and sort them by cumulative time
    stats = pstats.Stats(profile_filename)
    stats.sort_stats("cumulative").print_stats(40)
    exit()
# Call the CMA-ES optimization function with transformations
#initial_values = [float(val) for val in initial_values]
# es = cma.fmin(run_process_messages, initial_values, sigma0, options=myoptions)

if __name__ == '__main__':
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    freeze_support()
    es = cma.CMAEvolutionStrategy(initial_values, sigma0, myoptions)
    # Initialize the multiprocessing pool
    pool = mp.Pool(n_cores)
    counter = 1
    while not es.stop():
        X = es.ask()
        f_values = pool.map_async(run_process_messages, X).get()
        # use chunksize parameter as es.popsize/len(pool)?
        es.tell(X, f_values)
        if counter % 10 == 0:
            # Get the worst individual
            worst_individual = X[0] #np.argmax(f_values)
            # Run the function with profiling and save the results to a file
            cProfile.run("run_process_messages(worst_individual)", filename=profile_filename)
            # Load the results from the file and sort them by cumulative time
            stats = pstats.Stats(profile_filename)
            # print the top 20 functions sorted by cumulative time
            stats.sort_stats("cumulative").print_stats(40)
        es.disp()
        es.logger.add()
        es.logger.disp()
        #print(f"round {counter} of {myoptions['maxiter']}")
        counter += 1
    print('termination:', es.stop())
    cma.pprint(es.best.__dict__)
    print("Optimized parameters:", es.result.xbest)
    # Close the multiprocessing pool
    pool.close()
    pool.join()

