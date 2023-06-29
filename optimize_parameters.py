# This script is used to optimize the parameters of the keyword selection
# algorithm. It uses the CMA-ES algorithm to find the optimal parameters.
# The parameters are saved in the config/params.yaml file.

import pprint
import cma
import os
import shutil

# from cma import transformations
import numpy as np
import pandas as pd
import json
from main import process_messages
import time
import multiprocessing as mp
from multiprocessing import freeze_support
from multiprocessing import Pool
import cProfile
import pstats
import warnings
import csv
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt
from jsonmerge import merge
import seaborn as sns
import matplotlib.pyplot as plt

# Set the number of cores to use for multiprocessing
n_cores = 2
# cmaes population size, typically n_cores * integer
popsize = n_cores * 3
maxiter = 3
# Set the time penalty factor for the objective function
time_penalty_factor = 0

# Set the path to the file containing the most recent best parameters
xrecentbest_path = "outcmaes/xrecentbest.dat"
profile_filename = "profile_results.prof"
config_path = "config/params_tuned_20230520_2.csv"

# only profile the objective function / keyword selection algorithm
cProfile_switch = False
# use the most recent best parameters as starting point
best_switch = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    return df["freq"].to_dict()


# preloading the word frequency dictionary
word_freq_dict = load_word_freq_dict()

json_filename = "chosen_topics.json"
with open(json_filename, "r", encoding="utf-8") as f:
    messages = json.load(f)


# Define the objective function
def objective_function(parameters):
    """Run the process_messages function with the given parameters"""
    start_time = time.time()

    # Call the process_messages function
    performance = process_messages(word_freq_dict, parameters, messages)
    end_time = time.time()
    print("performance: ", performance)
    perf_score = (
        performance["common_topics"] - (end_time - start_time) * time_penalty_factor
    )

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
            if key1 not in params:
                params[key1] = {}
            # Assign the current parameter value to the appropriate key in the dictionary
            params[key1][key2] = x[i] if data_types[i] == "float" else int(x[i])

        dicts.append(params)
    return dicts


def check_ngrams(par_dicts):
    """Check the ngram parameters and set the ngram parameters to 1 if they are not valid"""
    for par_dict in par_dicts:
        # ensure that tf_IDF ngram_range1 <= ngram_range2
        par_dict["tf_IDF"]["ngram_range1"] = min(
            par_dict["tf_IDF"]["ngram_range1"], par_dict["tf_IDF"]["ngram_range2"]
        )


def read_parameter_file(file_path):
    """Read the parameter file and return the parameter keys, initial values,
    lower and upper bounds"""
    # Initialize empty dicts of lists
    opt_vars = {
        "param_keys": [],
        "initial_values": [],
        "lower_bounds": [],
        "upper_bounds": [],
        "data_types": [],
    }
    const_params = {
        "param_keys": [],
        "param_values": [],
        "data_types": [],
    }

    # Read the CSV file
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header
        next(csvreader)

        # Iterate through each row in the CSV file
        for row in csvreader:
            # Remove spaces from each element of the row
            row = [element.strip() for element in row]
            param_key = (row[1], row[2])
            initial_value = float(row[3])
            data_type = row[6]
            opt_switch = row[7]

            if opt_switch == "const":
                const_params["param_keys"].append(param_key)
                const_params["param_values"].append(initial_value)
                const_params["data_types"].append(data_type)

            elif opt_switch == "opt":
                opt_vars["param_keys"].append(param_key)
                opt_vars["initial_values"].append(initial_value)
                opt_vars["lower_bounds"].append(float(row[4]))
                opt_vars["upper_bounds"].append(float(row[5]))
                opt_vars["data_types"].append(data_type)
    return opt_vars, const_params


def get_best_opt_pars(file_path, initial_values):
    with open(file_path, "r") as f:
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
        print(
            "set initial values to best parameters from previous run:", initial_values
        )
        return xbest_values[best_index]
    else:
        print("number of parameters has changed, using initial values:", initial_values)
        return initial_values


def scale_variables(opt_vars, cma_bounds):
    """Scale the initial values to the range [cma_lower_bound, cma_upper_bound]"""

    scaled_initial_values = [
        ((value - lb) / (ub - lb)) * (sub - slb) + slb
        for value, lb, ub, slb, sub in zip(
            opt_vars["initial_values"],
            opt_vars["lower_bounds"],
            opt_vars["upper_bounds"],
            [cma_bounds[0]] * len(opt_vars["lower_bounds"]),
            [cma_bounds[1]] * len(opt_vars["upper_bounds"]),
        )
    ]

    return scaled_initial_values


def unscale_variables(scaled_variables, opt_vars, cma_bounds):
    """Unscale the variables from the range [cma_lower_bound, cma_upper_bound] to the range [lower_bound, upper_bound]"""
    unscaled_variables = [
        lb
        + (scaled_value - cma_bounds[0]) * (ub - lb) / (cma_bounds[1] - cma_bounds[0])
        for scaled_value, lb, ub in zip(
            scaled_variables, opt_vars["lower_bounds"], opt_vars["upper_bounds"]
        )
    ]
    return unscaled_variables


def optimize_parameters(es, opt_vars, const_params, cma_bounds):
    """Optimize the parameters using the CMA-ES algorithm"""

    # Request new list of candidate solutions
    X = es.ask()

    # Apply the scale_coordinates transformation to the objective function
    unscaled_candidates = [
        unscale_variables(candidate, opt_vars, cma_bounds) for candidate in X
    ]

    # Turn optimization variable lists into parameter dictionaries lists
    opt_var_dicts = lists_to_dicts(
        unscaled_candidates, opt_vars["param_keys"], opt_vars["data_types"]
    )
    # Turn constant parameter lists into parameter dictionaries lists
    const_par_dict = lists_to_dicts(
        [const_params["param_values"]],
        const_params["param_keys"],
        const_params["data_types"],
    )
    # Combine the constant and optimization variable dictionaries
    par_dicts = [
        merge(opt_var_dict, const_par_dict[0]) for opt_var_dict in opt_var_dicts
    ]

    # # par_dicts = [dict(opt_var_dict, **const_par_dict[0]) for opt_var_dict in opt_var_dicts]
    # print("par_dicts[0]:\n")
    # pprint.pprint(par_dicts[0])

    # evaluate function in parallel
    results = pool.map_async(objective_function, par_dicts).get()
    # with Pool(processes=4) as pool:  # Substitute 4 with number of desired processes
    #     for par_dict in par_dicts:
    #         result = pool.apply(objective_function, args=(par_dict,))
    #         print(result)

    # Extract the performance and runtime values from the results
    performance, f_values, runtimes = zip(*results)
    # Update the CMA-ES with the new objective function returns
    es.tell(X, f_values)
    return performance, runtimes, par_dicts


def run_cProfile(param_dict, top_n=20):
    """Run the objective function with profiling, save the results to a file
    and print the top_n functions sorted by cumulative time"""
    profile_filename = "profile_results.prof"
    # parameters = parameters.copy()

    # Run the function with profiling and save the results to a file
    # cProfile.run(
    #     f"exec(lambda: objective_function(param_dict))", filename=profile_filename
    # )
    # Create a profiler object
    profiler = cProfile.Profile()

    # Run the function with profiling using runcall method
    profiler.runcall(objective_function, param_dict)

    # Save the results to a file
    profiler.dump_stats(profile_filename)

    # Load the results from the file and sort them by cumulative time
    stats = pstats.Stats(profile_filename)
    # print the top_n functions sorted by cumulative time
    stats.sort_stats("cumulative").print_stats(top_n)


def print_performance(performance, runtimes, param_dicts):
    # Print the best performance and longest runtime every iteration
    print("Best performance:", max(performance))
    print("longest runtime:", max(runtimes))
    print(
        "sec/f_eval: ",
        (time.time() - main_start_time) / (counter * options["popsize"]),
    )


def backup_results(cov_matrix):
    # save the outcmaes files to the results folder
    timestamp = time.strftime("%Y%m%d_%H%M")
    src_folder = "outcmaes"
    new_folder_name = "outcmaes_" + timestamp
    destination_folder = os.path.join("results", new_folder_name)
    # os.makedirs(destination_folder, exist_ok=True)
    # for file in os.listdir("outcmaes"):
    #     shutil.copy(os.path.join("outcmaes", file), destination_folder)

    # Copy the entire directory tree to the new location
    shutil.copytree(src_folder, destination_folder)

    # # save the plots to the results folder
    # fig1.savefig(os.path.join(destination_folder, "result_diagram.png"))
    # fig2.savefig(os.path.join(destination_folder, "covariance_map.png"))

    # write the covariance matrix to a csv file
    cov_matrix_df = pd.DataFrame(cov_matrix)
    cov_matrix_df.to_csv(os.path.join(destination_folder, "covariance_matrix.csv"))

    print("backed up outcmaes files to:", destination_folder)
    return destination_folder


# Read the parameter file
opt_vars, const_params = read_parameter_file(config_path)

# choose the standard bounds for the CMA-ES optimization
cma_bounds = (0, 10)

# Set the initial standard deviation for the optimization
# The optimum should lie within the scaled bounds, approximately within x0 ± 3*sigma0.
sigma0 = 0.4 * (cma_bounds[1] - cma_bounds[0])

options = {
    "bounds": [
        [cma_bounds[0]] * len(opt_vars["initial_values"]),
        [cma_bounds[1]] * len(opt_vars["initial_values"]),
    ],
    "popsize": popsize,
    "verb_disp": 1,
    "tolx": 1e-6,
    # "tolfun": 3,
    "maxiter": maxiter,
    #'CMA_diagonal': True, performance:  {'spacy_NER': 1548, 'rake_keywords': 2794, 'tf_IDF': 45, 'LDA': 457, 'NMF': 925, 'common_topics': 3053}
}

# if cProfile_switch:
#     # TODO: Remove or update to handle const_params split
#     # Convert the LIST (only one here) of parameter values to a dictionary
#     param_dict = lists_to_dicts(initial_opt_values, param_keys, data_types)[0]
#     run_cProfile(param_dict, 20)
#     exit()

if __name__ == "__main__":
    # disable file validation to suppress warning messages
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    # improve debugging accuracy
    freeze_support()

    # create scale coordinates
    scaled_init_opt_values = scale_variables(opt_vars, cma_bounds)

    # Instantiate the CMAEvolutionStrategy with the scaled initial values
    es = cma.CMAEvolutionStrategy(scaled_init_opt_values, sigma0, options)

    # Initialize the multiprocessing pool
    pool = mp.Pool(n_cores)

    counter = 1
    try:
        print("starting optimization")
        while not es.stop():
            # initialize timing after cold start phase
            if counter == 2:
                main_start_time = time.time()

            performance, runtimes, par_dicts = optimize_parameters(
                es, opt_vars, const_params, cma_bounds
            )
            print("runtimes:", runtimes)

            if counter % 10 == 0:
                # print par_dicts with best performance
                min_idx = np.argmin(performance)
                print("best parameters:")
                pprint.pprint(par_dicts[min_idx])

            # Print the best performance and longest runtime every iteration
            if counter >= 2:
                print_performance(performance, runtimes, counter)

            if counter % 20 == 0:
                max_idx = np.argmax(runtimes)
                param_dict_max_runtimes = par_dicts[max_idx]
                run_cProfile(param_dict_max_runtimes, 20)

            es.logger.add()
            es.logger.disp()

            counter += 1
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        es.result_pretty()
        print("Optimization time: ", time.time() - main_start_time, "seconds")

        # access and plot the covariance matrix
        cov_matrix = es.C

        # backup outcmaes files
        dest_folder = backup_results(cov_matrix)

        # Generate plots from the logged data
        cma.plot()
        plt.savefig(f"{dest_folder}/result_diagram.png")
        plt.show(block=True)
        # input("Look at the plots and press enter to continue.")

        # plt.imshow(cov_matrix, cmap='coolwarm', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # input("Look at the plot and press enter to continue.")
        # plot the covariance matrix as a heatmap
        variable_names = [f"{key[0]}:{key[1]}" for key in opt_vars["param_keys"]]
        plt.figure(figsize=(50, 50))
        sns.heatmap(
            cov_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=variable_names,
            yticklabels=variable_names,
        )
        # variable_names = [...]  # list of variable names
        # cmap='coolwarm'
        # sns.heatmap(cov_matrix, annot=True, fmt=".2f",
        plt.savefig(f"{dest_folder}/covariance_map.png")
        # plt.show(block=True)
        # input("Look at the plot and press enter to continue.")

        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        pprint.pprint(corr_matrix)
        # sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=variable_names, yticklabels=variable_names)
        # plt.savefig(f"{dest_folder}/correlation_map.png")
        # plt.show(block=True)
        plt.figure(figsize=(50, 50))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=variable_names,
            yticklabels=variable_names,
        )
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.savefig(f"{dest_folder}/correlation_map.png")
        plt.show(block=True)
        # input("Look at the plot and press enter to continue.")

        # save optimization results to file
        with open("outcmaes/optimization_summary.json", "w") as f:
            summary_data = {
                "stop": es.stop(),
                "result": {
                    "x_best": es.result[0].tolist(),
                    "f_best": es.result[1],
                    "evaluations": es.result[3],
                    "stop": es.result[6].tolist(),
                },
            }
            json.dump(summary_data, f, indent=4)

        # Close the multiprocessing pool
        pool.terminate()  # pool.close()
        pool.join()

        # Obtain the result dictionary
        # result_dict = es.result()

        # Pretty print the result dictionary
        # pprint.pprint(result_dict)
