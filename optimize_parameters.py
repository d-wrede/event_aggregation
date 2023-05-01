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
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set the path to the file containing the most recent best parameters
xrecentbest_path = "outcmaes/xrecentbest_lastrun.dat"
profile_filename = "profile_results.prof"
# only profile the objective function / keyword selection algorithm
cProfile_switch = False
# use the most recent best parameters as starting point
best_switch = True

# run file as: python3 optimize_parameters.py -Xfrozen_modules=off

# Set the number of cores to use for multiprocessing
n_cores = 15

config_path = "config/params_indexed.csv"
with open(config_path, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    return df["freq"].to_dict()


# preloading the word frequency dictionary
word_freq_dict = load_word_freq_dict()

json_filename = "chosen_topics.json"
with open(json_filename, "r", encoding="utf-8") as f:
    messages = json.load(f)

nlp_spacy = spacy.load("de_core_news_lg", disable=["parser", "tagger"])


# Define the objective function
def objective_function(parameters):
    """Run the process_messages function with the given parameters"""
    start_time = time.time()

    # Call the process_messages function
    performance = process_messages(word_freq_dict, parameters, messages, nlp_spacy)
    end_time = time.time()
    perf_score = performance["common_topics"] - (end_time - start_time) * 25

    return performance["common_topics"], -perf_score, end_time - start_time


def lists_to_dicts(X, param_keys, data_types):
    dicts = []  # Initialize an empty list to store the dictionaries

    # If there is only one set of parameter values, convert it to a list of a list
    if isinstance(X[0], float):
        X = [X]

    # Iterate through the list of parameter value lists
    for x in X:  
        params = {}

        # Iterate through the parameter values and their corresponding keys
        for i, (key1, key2) in enumerate(param_keys):
            # If the first key is not in the dictionary, create a new entry
            if (key1 not in params):  
                params[key1] = {}

            # Assign the current parameter value to the appropriate key in the dictionary
            params[key1][key2] = x[i] if data_types[i] == "float" else int(x[i])

        # ensure that tf_IDF ngram_range1 <= ngram_range2
        params["tf_IDF"]["ngram_range1"] = min(
            params["tf_IDF"]["ngram_range1"], params["tf_IDF"]["ngram_range2"]
        )
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
            param_keys.append((row[1], row[2]))
            initial_values.append(float(row[3]))
            lower_bounds.append(float(row[4]))
            upper_bounds.append(float(row[5]))
            data_types.append(row[6])
    return param_keys, initial_values, lower_bounds, upper_bounds, data_types


def get_best_opt_pars(file_path, initial_values):
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

    # Return the xbest values corresponding to the lowest fitness value
    # if the number of parameters hasn't changed.
    if len(initial_values) == len(xbest_values[best_index]):
        print("set initial values to best parameters from previous run:", initial_values)
        return xbest_values[best_index]
    else:
        print("number of parameters has changed, using initial values:", initial_values)
        return initial_values


def scale_variables(initial_values, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound):
    """Scale the initial values to the range [cma_lower_bound, cma_upper_bound]"""
    scaled_lower_bounds = [cma_lower_bound] * len(lower_bounds)
    scaled_upper_bounds = [cma_upper_bound] * len(upper_bounds)
    
    scaled_initial_values = [((value - lb) / (ub - lb)) * (sub - slb) + slb for value, lb, ub, slb, sub in zip(initial_values, lower_bounds, upper_bounds, scaled_lower_bounds, scaled_upper_bounds)]

    return scaled_initial_values


def unscale_variables(scaled_variables, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound):
    """Unscale the variables from the range [cma_lower_bound, cma_upper_bound] to the range [lower_bound, upper_bound]"""
    unscaled_variables = [
        lb + (scaled_value - cma_lower_bound) * (ub - lb) / (cma_upper_bound - cma_lower_bound)
        for scaled_value, lb, ub in zip(scaled_variables, lower_bounds, upper_bounds)
    ]
    return unscaled_variables



# Read the parameter file
param_keys, initial_values, lower_bounds, upper_bounds, data_types = read_parameter_file(
    config_path
)

# use switch to load the best parameters from the previous run
if best_switch:
    initial_values = get_best_opt_pars(xrecentbest_path, initial_values)

# choose the standard bounds for the CMA-ES optimization
cma_lower_bound = 0
cma_upper_bound = 10

# Set the initial standard deviation for the optimization
# The optimum should lie within the scaled bounds, approximately within x0 ± 3*sigma0.
sigma0 = 0.1 * (cma_upper_bound - cma_lower_bound)

options = {
    "bounds": [[cma_lower_bound] * len(initial_values), [cma_upper_bound] * len(initial_values)],
    "popsize": 30,
    "verb_disp": 1,
    "tolx": 1e-6,
    "tolfun": 1e-4,
    # "maxiter": 10,
    #'CMA_diagonal': True,
}

if cProfile_switch:
    # Convert the LIST (only one here) of parameter values to a dictionary
    param_dict = lists_to_dicts(initial_values, param_keys, data_types)[0]

    # Run the function with profiling and save the results to a file
    cProfile.run("objective_function(param_dict)", filename=profile_filename)

    # Load the results from the file and sort them by cumulative time
    stats = pstats.Stats(profile_filename)
    stats.sort_stats("cumulative").print_stats(40)
    exit()

if __name__ == "__main__":
    # disable file validation to suppress warning messages
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    # improve debugging accuracy
    freeze_support()
    # create scale coordinates
    scaled_initial_values = scale_variables(initial_values, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound)
    # Instantiate the CMAEvolutionStrategy with the scaled initial values
    es = cma.CMAEvolutionStrategy(scaled_initial_values, sigma0, options)
    # Initialize the multiprocessing pool
    pool = mp.Pool(n_cores)
    counter = 1
    while not es.stop():
        # Request new list of candidate solutions
        X = es.ask()

        # unscale the candidates to translate into objective function space
        unscaled_candidates = [
            unscale_variables(candidate, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound)
            for candidate in X
            ]

        # Turn list into parameter dictionaries
        param_dicts = lists_to_dicts(unscaled_candidates, param_keys, data_types)

        # initialize timing after cold start phase
        if counter == 2:
            main_start_time = time.time()
        
        # evaluate function in parallel
        results = pool.map_async(objective_function, param_dicts).get()

        # Extract the performance and runtime values from the results
        performance, f_values, runtimes = zip(*results)
        # Update the CMA-ES with the new objective function returns
        es.tell(X, f_values)

        # Print the best performance and longest runtime every iteration
        if counter >= 2:
            # Print the best performance and longest runtime every iteration
            print("Best performance:", max(performance))
            print("longest runtime:", max(runtimes))
            print(
                "sec/f_eval: ",
                (time.time() - main_start_time) / (counter * options["popsize"]),
            )

        if counter % 10 == 0:
            max_idx = np.argmax(runtimes)
            worst_individual = param_dicts[max_idx]

            # Run the function with profiling and save the results to a file
            cProfile.run(
                "objective_function(worst_individual)", filename=profile_filename
            )
            # Load the results from the file and sort them by cumulative time
            stats = pstats.Stats(profile_filename)
            # print the top 20 functions sorted by cumulative time
            stats.sort_stats("cumulative").print_stats(40)
        #es.disp()
        es.logger.add()
        es.logger.disp()

        counter += 1
    es.result_pretty()
    print("Optimization time: ", time.time() - main_start_time, "seconds")
    # Generate plots from the logged data
    cma.plot()
    plt.show()
    input("Look at the plots and press enter to continue.")


    # Close the multiprocessing pool
    pool.close()
    pool.join()
