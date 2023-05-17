import pandas as pd
import re
from optimize_parameters import read_parameter_file, unscale_variables


def unscale_row(row):
    # Extract the variables to be unscaled
    candidate = row[column_names[5:]]

    # Unscaled the variables
    cma_bounds = (cma_lower_bounds, cma_upper_bounds)
    unscaled_candidates = unscale_variables(candidate, opt_vars, cma_bounds)
    #     candidate, opt_vars["l_bound"], opt_vars["u_bound"], cma_lower_bounds, cma_upper_bounds
    # )
    # unscale_variables(candidate, opt_vars, cma_bounds)

    # Replace the scaled variables with the unscaled variables
    row[column_names[5:]] = unscaled_candidates

    return row


def calculate_proposed_bounds(df_in, stddevs):
    """Calculate the proposed bounds for each variable."""

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

    df["range"] = df["u_bound"] - df["l_bound"]

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

    df.loc[~accepted_condition3, "p_u_bound_opt"] = df["u_bound"] + df["opt_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # If maximum value is too close to the upper bound
    df["max_dist"] = df["p_u_bound_init"] - df["max"]  # positive
    df["accepted_distance"] = minmaxscale * df["adjusted_range"]

    accepted_condition4 = df["max_dist"] > df["accepted_distance"]
    df.loc[~accepted_condition4, "p_u_bound_max"] = df["u_bound"] + df["max_dist"] * (
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
    df["p_l_bound"] = df[["p_l_bound", "l_bound"]].max(axis=1)
    df["p_u_bound"] = df[["p_u_bound", "u_bound"]].min(axis=1)

    return df[["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]].round(2)


def update_params_indexed(df_analysed, params_indexed_file):
    # Read the 'params_indexed.csv' file into a DataFrame
    params_indexed_df = pd.read_csv(params_indexed_file)

    print("params_indexed_df:\n", params_indexed_df.head())
    params_indexed_df["parameter2"] = params_indexed_df.iloc[:, 2].str.strip()

    # Loop through the rows of the 'df_analysed' DataFrame
    for index, row in df_analysed.iterrows():
        print()
        print(f"index: {index}, row:\n{row}")
        print()
        # Find the corresponding row in 'params_indexed_df' based on the parameter name
        match = params_indexed_df["parameter2"] == index.split(":")[-1]
        print("match:", match)

        # Update the 'initial_value', 'lower_bound', and 'upper_bound' columns
        params_indexed_df.loc[match, "initial_value"] = round(row["opt_val"], 2)
        params_indexed_df.loc[match, "lower_bound"] = round(row["p_l_bound"], 2)
        params_indexed_df.loc[match, "upper_bound"] = round(row["p_u_bound"], 2)

    print("params_indexed_df:\n", params_indexed_df.head())

    # updated_params_df = params_indexed_df.iloc[:, 1]
    # updated_params_df = params_indexed_df[['parameter2', 'initial_value', 'lower_bound', 'upper_bound']]
    # updated_params_df = params_indexed_df.iloc[:, 6]
    # updated_params_df = params_indexed_df.iloc[:, 7]

    # updated_params_df = params_indexed_df[['index', 'parameter1', 'parameter2', 'initial_value', 'lower_bound', 'upper_bound', 'data_type', 'description']]
    updated_params_df = params_indexed_df.iloc[:, [0, 1, 8, 9, 10, 11, 6, 7]]

    print("updated_params_df:\n", updated_params_df.head())
    # Save the updated 'params_indexed_df' back to the 'params_indexed.csv' file
    updated_params_df.to_csv("config/params_tuned.csv", index=False)


# def update_params_indexed(df_analysed, params_indexed_file):
#     # Read the 'params_indexed.csv' file into a DataFrame
#     params_indexed_df = pd.read_csv(params_indexed_file)

#     # Store the lengths of the entries in the 'parameter2' column
#     entry_lengths = pd.DataFrame()
#     entry_lengths['parameter2'] = params_indexed_df.iloc[:, 2].str.len()
#     entry_lengths['initial_value'] = params_indexed_df.iloc[:, 3].len()
#     entry_lengths['lower_bound'] = params_indexed_df.iloc[:, 4].len()
#     entry_lengths['upper_bound'] = params_indexed_df.iloc[:, 5].len()

#     # Strip whitespaces from the strings in the 'parameter2' column
#     params_indexed_df['parameter2'] = params_indexed_df.iloc[:, 2].str.strip()
#     params_indexed_df['initial_value'] = params_indexed_df.iloc[:, 3]
#     params_indexed_df['lower_bound'] = params_indexed_df.iloc[:, 4]
#     params_indexed_df['upper_bound'] = params_indexed_df.iloc[:, 5]

#     print("params_indexed_df:\n", params_indexed_df.head())

#     # Loop through the rows of the 'df_analysed' DataFrame
#     for index, row in df_analysed.iterrows():
#         print()
#         print(f"index: {index}, row:\n{row}")
#         print()
#         # Find the corresponding row in 'params_indexed_df' based on the parameter name
#         match = params_indexed_df['parameter2'] == index.split(":")[-1]
#         print("match:", match)
#         # Find the entry length for the current parameter
#         entry_length = entry_lengths.loc[match, 'entry_length'].values[0]

#         # Update the 'initial_value', 'lower_bound', and 'upper_bound' columns
#         # Format the updated values with the proper length and whitespaces
#         params_indexed_df.loc[match, 'initial_value'] = " {:.2f}".format(row['opt_val']).ljust(entry_length)
#         params_indexed_df.loc[match, 'lower_bound'] = " {:.2f}".format(row['p_l_bound']).ljust(entry_length)
#         params_indexed_df.loc[match, 'upper_bound'] = " {:.2f}".format(row['p_u_bound']).ljust(entry_length)

#         # Update the 'initial_value', 'lower_bound', and 'upper_bound' columns with the needed number of spaces
#         params_indexed_df.loc[match, 'initial_value'] = " {:.2f}".format(row['opt_val']).ljust(entry_length + 1)
#         params_indexed_df.loc[match, 'lower_bound'] = " {:.2f}".format(row['p_l_bound']).ljust(entry_length + 1)
#         params_indexed_df.loc[match, 'upper_bound'] = " {:.2f}".format(row['p_u_bound']).ljust(entry_length + 1)

#     updated_params_df = params_indexed_df.iloc[:, 1]
#     updated_params_df = params_indexed_df['parameter2']
#     updated_params_df = params_indexed_df['initial_value']
#     updated_params_df = params_indexed_df['lower_bound']
#     updated_params_df = params_indexed_df['upper_bound']
#     updated_params_df = params_indexed_df.iloc[:, 6]
#     updated_params_df = params_indexed_df.iloc[:, 7]

#     print("updated_params_df:\n", updated_params_df.head())


#     # Save the updated 'params_indexed_df' back to the 'params_indexed.csv' file
#     updated_params_df.to_csv("config/funny.csv", index=False)


# Read the parameter names from the 'params_indexed.csv' file
params = pd.read_csv("config/params_indexed.csv")
params.columns = params.columns.str.strip()
param_names = (
    params["parameter1"].str.strip() + ":" + params["parameter2"].str.strip()
).tolist()
# print("Parameter names:", param_names)
# param_names = (params["parameter1"].strip() + "_" + params["parameter2"].strip()).tolist()

# Read the first line of the file to extract column names
with open("outcmaes/xrecentbest.dat") as f:
    first_line = f.readline()

# Extract column names from the first line
column_names_str = first_line.split('"')[1]
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

# Exclude the initial iterations (e.g., rows with 'evals' less than 5000)
# df_scaled = df_scaled[df_scaled["evals"] >= 10000]
threshold = df_scaled["evals"].quantile(0.5)
df_scaled = df_scaled[df_scaled["evals"] >= threshold]


# Read the parameter file
config_path = "config/params_tuned_230515.csv"
opt_vars, const_params = read_parameter_file(
    config_path
)

# param_keys, initial_values, l_bound, u_bound, data_types

cma_lower_bounds = 0
cma_upper_bounds = 10
# for each row in data, unscale the variables (columns 5 to end)
# unscaled_candidates = unscale_variables(candidate, l_bound, u_bound, cma_lower_bounds, cma_upper_bounds)

# Apply the unscale function to each row of the data DataFrame
print("df_scaled head:\n", df_scaled.head())
df_unscaled = df_scaled.apply(unscale_row, axis=1)
print("unscaled_data head:\n", df_unscaled.head())


df_analysed = df_unscaled[column_names[5:]].describe().T.round(2)
df_analysed.index.name = "parameter_name"
df_analysed = df_analysed.drop(["count", "std", "25%", "50%", "75%"], axis=1)

# add row with min fitness value to describe_unscaled_data
min_fitness_row = df_unscaled.loc[df_unscaled["fitness"].idxmin()]
df_analysed["opt_val"] = min_fitness_row.T.round(2)
df_analysed["u_bound"] = opt_vars["u_bound"]
df_analysed["l_bound"] = opt_vars["l_bound"]


# read stddev from file
with open("outcmaes/stddev.dat") as f:
    last_line = f.readlines()[-1]

stddevs = last_line.split(" ")[5:]

# calculate proposed bounds
df_analysed[
    ["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]
] = calculate_proposed_bounds(df_analysed, stddevs)

df_analysed = df_analysed[
    [
        "mean",
        "min",
        "max",
        "opt_val",
        "l_bound",
        "u_bound",
        "ratio_stddev",
        "p_l_bound",
        "p_u_bound",
        "remark_l",
        "remark_u",
    ]
]
print("variable analysis:\n", df_analysed.sort_values(by=['ratio_stddev'], ascending=False))

df_analysed.to_csv("analyze_data/variable_analysis.csv", index=True)

# Usage example:
update = input("Do you want to update the parameter file? (y/n): ")
if update == "y":
    update_params_indexed(df_analysed, "config/params_indexed.csv")
    print("Parameter file updated.")
else:
    print("Parameter file not updated.")



exit()
# Calculate min, max, and average values for each variable
# min_values = data[column_names[5:]].min()
# max_values = data[column_names[5:]].max()
# average_values = data[column_names[5:]].mean()

# # Create a DataFrame to store the calculated values
# summary = pd.DataFrame(
#     {"Min": min_values, "Max": max_values, "Average": average_values}
# )

# print("data head3:\n", data[column_names[5:]].head())
# # Calculate the tendency using a histogram
# tendency = (
#     data[column_names[5:]].apply(pd.Series.value_counts, bins=10, normalize=True).T
# )
# print("tendency head:\n", tendency.head())
print("describe data:", df_scaled.describe())
# Add the tendency information to the summary DataFrame
for idx, col_name in enumerate(tendency.columns):
    summary[f"Tendency_{(idx+1)*10}%"] = tendency[col_name].values

# Display the results
print("summary head1:\n", summary.head())

# Calculate additional statistics
additional_stats = (
    df_scaled[column_names[5:]].describe().loc[["25%", "50%", "75%", "std"]]
)
summary = summary.join(additional_stats.T)
print("summary head2:\n", summary.head())
summary.to_csv("analyze_data/summary.csv", index=True)

# Convert the summary DataFrame to a readable string
summary_str = summary.to_string()

# Print the summary DataFrame
print("Summary:")
print(summary_str)
