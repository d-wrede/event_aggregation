# This script is used to optimize the parameters of the keyword selection
# algorithm. It uses the CMA-ES algorithm to find the optimal parameters.
# The parameters are saved in the config/params.yaml file.

import cma
import os

# from cma import transformations
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
import warnings
import csv
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set the path to the file containing the most recent best parameters
xrecentbest_path = "outcmaes/xrecentbest.dat"
profile_filename = "profile_results.prof"
cProfile_switch = False

# run file as: python3 optimize_parameters.py -Xfrozen_modules=off

# Set the number of cores to use for multiprocessing
n_cores = 4

config_path = "config/params.csv"
with open(config_path, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    # print("Word frequency dictionary loaded in optimize_parameters.py")
    return df["freq"].to_dict()


# preloading the word frequency dictionary
word_freq_dict = load_word_freq_dict()

json_filename = "chosen_topics.json"
with open(json_filename, "r", encoding="utf-8") as f:
    messages = json.load(f)

nlp_spacy = spacy.load("de_core_news_lg", disable=["parser", "tagger"])


# Define the objective function
def run_process_messages(parameters):
    """Run the process_messages function with the given parameters"""
    start_time = time.time()

    # Call the process_messages function
    performance = process_messages(word_freq_dict, parameters, messages, nlp_spacy)
    end_time = time.time()
    perf_score = performance["common_topics"] - (end_time - start_time) * 25

    # print("performance['common_topics']", performance['common_topics'])
    # print("iteration time: ", end_time - start_time)
    # print("combined perf_score: ", perf_score)
    # Return the performance score for optimization
    return performance["common_topics"], -perf_score, end_time - start_time


def lists_to_dicts(X, param_keys, data_types):
    dicts = []  # Initialize an empty list to store the dictionaries

    # Iterate through the list of parameter value lists
    for x in X:  
        # Initialize an empty dictionary to store the current parameter values
        params = {}

        # Iterate through the parameter values and their corresponding keys
        for i, (key1, key2) in enumerate(param_keys):
            # If the first key is not in the dictionary, create a new entry
            if (key1 not in params):  
                params[key1] = {}

            # Assign the current parameter value to the appropriate key in the dictionary
            params[key1][key2] = x[i] if data_types[i] == "float" else int(x[i])

        dicts.append(params)
    return dicts


def read_parameter_file(file_path):
    """Read the parameter file and return the parameter keys, initial values,
    lower and upper bounds"""
    # Initialize empty lists
    initial_values = []
    lower_bounds = []
    upper_bounds = []
    param_keys = []
    data_types = []

    # Read the CSV file
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header
        next(csvreader)

        # Iterate through each row in the CSV file
        for row in csvreader:
            # Remove spaces from each element of the row
            row = [element.strip() for element in row]
            param_keys.append((row[0], row[1]))
            initial_values.append(float(row[2]))
            lower_bounds.append(float(row[3]))
            upper_bounds.append(float(row[4]))
            data_types.append(row[5])
    return param_keys, initial_values, lower_bounds, upper_bounds, data_types


def get_best_opt_pars(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Remove the first line (header)
    data_lines = lines[1:]

    # Extract the fitness values and corresponding xbest values
    fitness_values = []
    xbest_values = []
    for line in data_lines:
        parts = line.strip().split()
        fitness = float(parts[4])
        xbest = [float(x) for x in parts[5:]]
        fitness_values.append(fitness)
        xbest_values.append(xbest)

    # Find the index of the lowest fitness value
    best_index = fitness_values.index(min(fitness_values))

    # Return the corresponding xbest values
    return xbest_values[best_index]


# Read the parameter file
param_keys, initial_values, lower_bounds, upper_bounds, data_types = read_parameter_file(
    config_path
)

initial_values = get_best_opt_pars(xrecentbest_path)

options = {
    "bounds": [lower_bounds, upper_bounds],
    "popsize": 4,
    "verb_disp": 1,
    "tolx": 1e-6,
    "tolfun": 1e-4,
    "maxiter": 10000,
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
# initial_values = [float(val) for val in initial_values]
# es = cma.fmin(run_process_messages, initial_values, sigma0, options=myoptions)

if __name__ == "__main__":
    # disable file validation to suppress warning messages
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    # improve debugging accuracy
    freeze_support()
    es = cma.CMAEvolutionStrategy(initial_values, sigma0, options)
    # Initialize the multiprocessing pool
    pool = mp.Pool(n_cores)
    counter = 1
    while not es.stop():
        # Request new list of candidate solutions
        X = es.ask()

        # Turn list into parameter dictionaries
        param_dicts = lists_to_dicts(X, param_keys, data_types)

        # initialize timing after cold start phase
        if counter == 2:
            main_start_time = time.time()

        # evaluate function in parallel
        results = pool.map_async(run_process_messages, param_dicts).get()

        # Print the best performance and longest runtime every iteration
        if counter >= 2:
            # Print the best performance and longest runtime every iteration
            print("Best performance:", max(performance))
            print("longest runtime:", max(runtimes))
            print(
                "sec/f_eval: ",
                (time.time() - main_start_time) / (counter * options["popsize"]),
            )

        performance, f_values, runtimes = zip(*results)
        es.tell(X, f_values)

        if counter % 10 == 0:
            max_idx = np.argmax(runtimes)
            worst_individual = param_dicts[max_idx]
            #lists_to_dicts(X, param_keys, data_types)
            # Run the function with profiling and save the results to a file
            cProfile.run(
                "run_process_messages(worst_individual)", filename=profile_filename
            )
            # Load the results from the file and sort them by cumulative time
            stats = pstats.Stats(profile_filename)
            # print the top 20 functions sorted by cumulative time
            stats.sort_stats("cumulative").print_stats(40)
        es.disp()
        # es.logger.add()
        # es.logger.disp()
        # print(f"round {counter} of {myoptions['maxiter']}")
        counter += 1
    print("termination:", es.stop())
    es.result_pretty()
    print("Optimized parameters:", es.result.xbest)
    # Close the multiprocessing pool
    pool.close()
    pool.join()

    # Convert the optimized parameters to a dictionary
    #parameters = update_parameters(es.result.xbest)
    # Save the updated parameters in yaml format
    with open("config/params_optimized.yaml", "w") as file:
        yaml.dump(parameters, file)
