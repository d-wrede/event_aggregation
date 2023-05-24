import time
import pandas as pd
import re
import os
from optimize_parameters import read_parameter_file, unscale_variables

fn_readparams = "config\params_tuned_20230520_2.csv"

# generate new filename for the tuned parameters
n = 0
while True:
    fn_params_tuned = f"config/params_tuned_{time.strftime('%Y%m%d')}_{n}.csv"
    if not os.path.isfile(fn_params_tuned):
        break
    n += 1

def unscale_row(row):
    """
    Unscales a row of data.

    This function takes a row of data, extracts the variables to be unscaled, unscales the variables, 
    and replaces the original scaled variables with the unscaled variables.

    Args:
        row (pd.Series or similar): A row of data containing variables to be unscaled.

    Returns:
        row: The row of data with unscaled variables.
    """
    # Extract the variables to be unscaled
    scaled_values = row[column_names[5:]]
    # print("scaled_values:", scaled_values)

    # Unscaled the variables
    cma_bounds = (cma_lower_bounds, cma_upper_bounds)
    unscaled_candidates = unscale_variables(scaled_values, opt_vars, cma_bounds)
    #     candidate, opt_vars["lower_bounds"], opt_vars["upper_bounds"], cma_lower_bounds, cma_upper_bounds
    # )
    # unscale_variables(candidate, opt_vars, cma_bounds)

    # Replace the scaled variables with the unscaled variables
    print("unscaled_candidates:", unscaled_candidates)
    print("row[column_names[5:]]:", row[column_names[5:]])
    row[column_names[5:]] = unscaled_candidates

    return row


def calculate_proposed_bounds(df_in, stddevs):
    """
    Calculates the proposed bounds for the optimization parameters based on statistical analysis.

    The function analyzes the input DataFrame and standard deviations of the parameters. It calculates proposed bounds 
    for each parameter by adjusting its range based on its standard deviation ratio. It then corrects these proposed bounds 
    to ensure they aren't too close to values from the previous optimization run. If the original bound was zero, it will 
    be adjusted accordingly. Remarks are generated when bounds are adjusted.

    Args:
        df_in (pd.DataFrame): Input DataFrame with data of the parameters.
        stddevs (list): List of standard deviations of the parameters.

    Returns:
        pd.DataFrame: A DataFrame with the ratio of standard deviations and the proposed lower and upper bounds for 
        each parameter, along with remarks for any adjustments made.
    """

    def scale_factors(P1, P2):
        """Calculate the scale factor between two points."""
        x1, y1 = P1
        x2, y2 = P2
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b

    def update_bounds(
        dist, level, bounds, adjusted_range, ratio_stddevs, a_bound, b_bound
    ):
        """Calculate the proposed bounds for each variable."""
        result = []
        for d, b, ar, rs in zip(dist, bounds, adjusted_range, ratio_stddevs):
            if d < 0.1 * ar and level == "min":
                result.append(b - d * (a_bound * rs + b_bound))
            elif d < 0.2 * ar and level == "opt_val":
                result.append(b - d * (a_bound * rs + b_bound))
            elif d < 0.1 * ar and level == "max":
                result.append(b + d * (a_bound * rs + b_bound))
            else:
                result.append(b)

        return result

    df = df_in.copy()

    max_dev = float(max(stddevs))
    ratio_stddevs = [0] * len(stddevs)
    for i in range(len(stddevs)):
        ratio_stddevs[i] = float(stddevs[i].replace("\n", "")) / max_dev
    df["ratio_stddev"] = ratio_stddevs

    df["range"] = df["upper_bounds"] - df["lower_bounds"]

    # Calculate the scale factors for the adjusted range
    major_scale = 1
    # low ratio point (ratio, scale_factor):
    P1 = (0.25, 1.2) * major_scale
    # high ratio point:
    P2 = (1, 0.7) * major_scale
    a_range, b_range = scale_factors(P1, P2)

    # Adjust the range based on the standard deviation ratio and the scale factor
    # df['adjusted_range'] = df.apply(lambda row: row['range'] * (a_range * row['ratio_stddev'] + b_range), axis=1)
    df["adjusted_range"] = df["range"] * (a_range * df["ratio_stddev"] + b_range)

    # Calculate the weighted center considering the mean, max and min values
    df["weighted_center"] = (2 * df["opt_val"] + df["mean"] + df["max"] + df["min"]) / 5
    # TODO: add global scaling factor
    
    # the following is not weighted, but
    # df['weighted_center'] = (df['u_bound'] + df['l_bound']) / 2

    # Calculate the proposed lower and upper bounds
    df["p_l_bound_init"] = df["weighted_center"] - df["adjusted_range"] / 2
    df["p_u_bound_init"] = df["weighted_center"] + df["adjusted_range"] / 2

    # Calculate the scale factors for the bounds
    # low ratio point (ratio, scale_factor):
    P1 = (0.25, 1.7) * major_scale
    # high ratio point:
    P2 = (1, 1) * major_scale
    a_bound, b_bound = scale_factors(P1, P2)
    
    def create_remark(row, bound_type, dir):
        if dir == "min":
            column_name = row.idxmin()
        elif dir == "max":
            column_name = row.idxmax()

        if column_name == f"p_{bound_type}_bound_opt":
            return "opt"
        elif column_name == f"p_{bound_type}_bound_min":
            return "min"
        elif column_name == f"p_{bound_type}_bound_max":
            return "max"
        else:
            return "standard"

    # # use the maximum of the calculated bounds
    # df['p_u_bound'] = df[['p_u_bound_opt', 'p_u_bound_max', 'p_u_bound']].max(axis=1)
    # Correct the lower bounds, if necessary
    # high values will lead to redefining the bounds earlier
    optscale = 0.3
    minmaxscale = 0.2

    df["opt_dist"] = df["opt_val"] - df["p_l_bound_init"]
    df["accepted_distance"] = optscale * df["adjusted_range"]
    # if optimized value is too close to the lower bound
    # condition is true, if distance is accepted
    accepted_condition1 = df["opt_dist"] > df["accepted_distance"]

    # reevaluate the lower bound
    df.loc[~accepted_condition1, "p_l_bound_opt"] = df["p_l_bound_init"] - df[
        "opt_dist"
    ] * (a_bound * df["ratio_stddev"] + b_bound)

    df["min_dist"] = df["min"] - df["p_l_bound_init"]
    df["accepted_distance"] = minmaxscale * df["adjusted_range"]
    accepted_condition2 = df["min_dist"] > df["accepted_distance"]

    df.loc[~accepted_condition2, "p_l_bound_min"] = df["p_l_bound_init"] - df[
        "min_dist"
    ] * (a_bound * df["ratio_stddev"] + b_bound)

    # Use the minimum of the calculated bounds
    df["p_l_bound"] = df[["p_l_bound_opt", "p_l_bound_min", "p_l_bound_init"]].min(
        axis=1
    )

    # leave a remark, which bound was used
    df["remark_l"] = df[["p_l_bound_opt", "p_l_bound_min", "p_l_bound_init"]].apply(
        lambda row: create_remark(row, "l", "min"), axis=1
    )

    # Correct the upper bounds, if necessary
    # If optimized value is too close to the upper bound
    df["opt_dist"] = df["p_u_bound_init"] - df["opt_val"]  # positive
    df["accepted_distance"] = optscale * df["adjusted_range"]

    accepted_condition3 = df["opt_dist"] > df["accepted_distance"]

    df.loc[~accepted_condition3, "p_u_bound_opt"] = df["upper_bounds"] + df["opt_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # If maximum value is too close to the upper bound
    df["max_dist"] = df["p_u_bound_init"] - df["max"]  # positive
    df["accepted_distance"] = minmaxscale * df["adjusted_range"]

    accepted_condition4 = df["max_dist"] > df["accepted_distance"]
    df.loc[~accepted_condition4, "p_u_bound_max"] = df["upper_bounds"] + df["max_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # Use the maximum of the calculated bounds
    df["p_u_bound"] = df[["p_u_bound_opt", "p_u_bound_max", "p_u_bound_init"]].max(
        axis=1
    )

    # leave a remark, which bound was used
    df["remark_u"] = df[["p_u_bound_opt", "p_u_bound_max", "p_u_bound_init"]].apply(
        lambda row: create_remark(row, "u", "max"), axis=1
    )

    # correct the bounds, if original bound was zero
    df["p_l_bound"] = df[["p_l_bound", "lower_bounds"]].max(axis=1)
    df["p_u_bound"] = df[["p_u_bound", "upper_bounds"]].min(axis=1)

    return df[["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]].round(2)


def update_params_file(df_analysed, fn_read, fn_write):
    """
    Update parameters from an analysis DataFrame based on a parameters file.

    This function reads the parameters file into a DataFrame, matches rows in the analysed DataFrame based on 
    the parameter name, and updates the 'initial_value', 'lower_bound', and 'upper_bound' columns. 
    The updated DataFrame is then saved back to the parameters file.

    Args:
        df_analysed (pd.DataFrame): A DataFrame containing analysed data.
        fn_read (str): The filepath of the parameters file to be read.
        fn_write (str): The filepath of the parameters file to be updated.

    Returns:
        None
    """
    # Read the parameter file into a DataFrame
    params_indexed_df = pd.read_csv(fn_read)
    # print("params_indexed_df:\n", params_indexed_df.head())
    # drop old index column
    params_indexed_df = params_indexed_df.drop(params_indexed_df.columns[0], axis=1)
    # print("params_indexed_df:\n", params_indexed_df.head())

    # remove whitespace from column names
    params_indexed_df.columns = params_indexed_df.columns.str.strip()
    params_indexed_df["parameter1"] = params_indexed_df.iloc[:, 0].str.strip()
    params_indexed_df["parameter2"] = params_indexed_df.iloc[:, 1].str.strip()
    # to do it for all columns, consider
    # for col in params_indexed_df.columns:
    #     params_indexed_df[col] = params_indexed_df[col].str.strip()

    
    # print("params_indexed_df:\n", params_indexed_df.head())

    # Loop through the rows of the 'df_analysed' DataFrame
    for index, row in df_analysed.iterrows():
        # print()
        # print(f"index: {index}, row:\n{row}")
        # print()
        # Find the corresponding row in 'params_indexed_df' based on the parameter name
        match1 = params_indexed_df["parameter1"] == index.split(":")[0]
        match2 = params_indexed_df["parameter2"] == index.split(":")[-1]
        match = match1 & match2
        # print("match:", match)

        # Update the 'initial_value', 'lower_bound', and 'upper_bound' columns
        params_indexed_df.loc[match, "initial_value"] = round(row["opt_val"], 2)
        params_indexed_df.loc[match, "lower_bound"] = round(row["p_l_bound"], 2)
        params_indexed_df.loc[match, "upper_bound"] = round(row["p_u_bound"], 2)

    # print("params_indexed_df:\n", params_indexed_df.head())

    # updated_params_df = params_indexed_df[['index', 'parameter1', 'parameter2', 'initial_value', 'lower_bound', 'upper_bound', 'data_type', 'description']]
    # column_order = ['index', 'parameter1', 'parameter2', 'initial_value', 'lower_bound', 'upper_bound', 'data_type', 'description']
    # updated_params_df = params_indexed_df[column_order]

    # updated_params_df = params_indexed_df.iloc[:, [0, 1, 8, 9, 10, 11, 6, 7]]

    # print("updated_params_df:\n", params_indexed_df.head())
    # Save the updated 'params_indexed_df' back to the 'params_indexed.csv' file
    params_indexed_df.to_csv(fn_write, index=True)


###                                       ###
#   Read parameter names from config file   #
###                                       ###

# Read the parameter names from the 'params_indexed.csv' file
params = pd.read_csv(fn_readparams, index_col=None, skipinitialspace=True)
# Take all rows, and all columns starting from the second column)
params = params.iloc[:, 1:]
# optimized value parameters
params_opt = params[params['opt_or_not'] == 'opt']
# constant value parameters
params_const = params[params['opt_or_not'] == 'const']

params_opt.columns = params_opt.columns.str.strip()
# parameter names of the optimized parameters
param_names = (
    params_opt["parameter1"].str.strip() + ":" + params_opt["parameter2"].str.strip()
).tolist()
print("param names:\n", param_names)
print("len(params_opt): ", len(params_opt))
# print("Parameter names:", param_names)
# param_names = (params_opt["parameter1"].strip() + "_" + params_opt["parameter2"].strip()).tolist()

# Read the parameter file
opt_vars, const_params = read_parameter_file(
    fn_readparams
)


###                                      ###
#   Read optimization results recentbest   #
###                                      ###

# Read the first line of the file to extract column names
with open("outcmaes/xrecentbest.dat") as f:
    # gets "% # columns="iter, evals, sigma, 0, fitness, xbest" seed=540356, Thu May 18 15:02:09 2023, <python>{}</python>"
    first_line = f.readline()

# Extract column names from the first line
# gets "iter, evals, sigma, 0, fitness, xbest"
column_names_str = first_line.split('"')[1]
# since xbest comprises the parameter values, the column names are the parameter names
column_names = column_names_str.split(", ")[:5] + param_names

# Read the data from the file, skipping the first row (header)
df_scaled = pd.read_csv(
    "outcmaes/xrecentbest.dat",
    delim_whitespace=True,
    skiprows=1,
    header=None,
    index_col=False,
    names=column_names,
)

# evaluate the last 50% of the optimization data
threshold = df_scaled["evals"].quantile(0.5)
df_scaled = df_scaled[df_scaled["evals"] >= threshold]

###                                    ###
#   Analyse optimization results         #
###                                    ###
#   Goal: mean, min, max, stddev, fitness,
#         opt_val, lower_bounds, upper_bounds,

# cma scaling bounds
cma_lower_bounds = 0
cma_upper_bounds = 10

# for each row in the data, unscale the variables (columns 5 to end)
# Apply the unscale function to each row of the data DataFrame
print("df_scaled head:\n", df_scaled.head())
df_unscaled = df_scaled.apply(unscale_row, axis=1)
print("unscaled_data head:\n", df_unscaled.head())

# get mean, min, max values for each variable
df_analysed = df_unscaled[column_names[5:]].describe().T.round(3)
df_analysed.index.name = "parameter_name"
df_analysed = df_analysed.drop(["count", "std", "25%", "50%", "75%"], axis=1)
print("df_analysed head:\n", df_analysed.head())

# add column with optimized values, taken from the best result (min fitness)
min_fitness_row = df_unscaled.loc[df_unscaled["fitness"].idxmin()]
df_analysed["opt_val"] = min_fitness_row.T.round(3)
print("df_analysed:\n", df_analysed)

# add columns with applied lower and upper bounds
df_analysed["lower_bounds"] = opt_vars["lower_bounds"]
df_analysed["upper_bounds"] = opt_vars["upper_bounds"]


###                                   ###
#   Calculate proposed bounds           #
#   based on the stddev of the results  #
###                                   ###

# read stddev from file
with open("outcmaes/stddev.dat") as f:
    last_line = f.readlines()[-1]

stddevs = last_line.split(" ")[5:]

# calculate proposed bounds
df_analysed[
    ["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]
] = calculate_proposed_bounds(df_analysed, stddevs)

# reorder columns
df_analysed = df_analysed[
    [
        "mean",
        "min",
        "max",
        "opt_val",
        "lower_bounds",
        "upper_bounds",
        "ratio_stddev",
        "p_l_bound",
        "p_u_bound",
        "remark_l",
        "remark_u",
    ]
]
print("variable analysis:\n", df_analysed.sort_values(by=['ratio_stddev'], ascending=False))

df_analysed.to_csv("analyze_data/variable_analysis.csv", index=True)

# update the parameter file
print("Do you want to update the parameter file with the proposed bounds and opt values as initial values?")
update = input(f"They will be written to {fn_params_tuned} (y/n): ")
if update == "y":
    update_params_file(df_analysed, fn_readparams, fn_params_tuned)
    print("\nParameter file updated. Please check the updated parameter file,\n\
update read param_filename and run the optimization again.")
else:
    print("Parameter file not updated.")



exit()