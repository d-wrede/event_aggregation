import pandas as pd
import re
from optimize_parameters import read_parameter_file, unscale_variables


def unscale_row(row):
    # Extract the variables to be unscaled
    candidate = row[column_names[5:]]

    # Unscaled the variables
    unscaled_candidates = unscale_variables(
        candidate, l_bound, u_bound, cma_lower_bounds, cma_upper_bounds
    )

    # Replace the scaled variables with the unscaled variables
    row[column_names[5:]] = unscaled_candidates

    return row


def calculate_proposed_bounds(df_in, stddevs):
    """Calculate the proposed bounds for each variable."""

    def scale_factors(x1, y1, x2, y2):
        """Calculate the scale factor between two points."""
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
    a_range, b_range = scale_factors(0.25, 1.2, 1, 0.7)

    # Adjust the range based on the standard deviation ratio and the scale factor
    # df['adjusted_range'] = df.apply(lambda row: row['range'] * (a_range * row['ratio_stddev'] + b_range), axis=1)
    df["adjusted_range"] = df["range"] * (a_range * df["ratio_stddev"] + b_range)

    # Calculate the weighted center considering the mean, max and min values
    df["weighted_center"] = (df["mean"] + df["max"] + df["min"]) / 3

    # Calculate the proposed lower and upper bounds
    df["p_l_bound_init"] = df["weighted_center"] - df["adjusted_range"] / 2
    df["p_u_bound_init"] = df["weighted_center"] + df["adjusted_range"] / 2

    # Calculate the scale factors for the bounds
    a_bound, b_bound = scale_factors(0.25, 1.7, 1, 1.2)

    # # Correct the lower bounds, if necessary
    # df['min_dist'] = df['opt_val'] - df['u_bound']
    # df.loc[df['min_dist'] < 0.2 * df['adjusted_range'], 'p_l_bound_opt'] = df['u_bound'] - df['min_dist'] * (a_bound * df['ratio_stddev'] + b_bound)
    # df['min_dist'] = df['min'] - df['u_bound']
    # df.loc[df['min_dist'] < 0.1 * df['adjusted_range'], 'p_l_bound_min'] = df['u_bound'] - df['min_dist'] * (a_bound * df['ratio_stddev'] + b_bound)

    # # use the minimum of the calculated bounds
    # df['p_l_bound'] = df[['p_l_bound_opt', 'p_l_bound_min', 'p_l_bound']].min(axis=1)

    # # Correct the upper bounds, if necessary
    # # if optimized value is too close to the upper bound
    # df['max_dist'] = df['opt_val'] - df['u_bound']
    # df.loc[df['max_dist'] < 0.2 * df['adjusted_range'], 'p_u_bound_opt'] = df['u_bound'] + df['max_dist'] * (a_bound * df['ratio_stddev'] + b_bound)
    # # if maximum value is too close to the upper bound
    # df['min_dist'] = df['max'] - df['u_bound']
    # df.loc[df['max_dist'] < 0.1 * df['adjusted_range'], 'p_u_bound_max'] = df['u_bound'] + df['max_dist'] * (a_bound * df['ratio_stddev'] + b_bound)

    def create_remark(row, bound_type):
        column_name = row.idxmin()
        if column_name == f"p_{bound_type}_bound_opt":
            return "opt"
        elif column_name == f"p_{bound_type}_bound_min":
            return "min"
        else:
            return "standard"

    # # use the maximum of the calculated bounds
    # df['p_u_bound'] = df[['p_u_bound_opt', 'p_u_bound_max', 'p_u_bound']].max(axis=1)
    # Correct the lower bounds, if necessary
    optscale = 0.02
    minmaxscale = 0.01
    df["min_dist"] = df["opt_val"] - df["p_l_bound_init"]
    condition1 = df["min_dist"] < optscale * df["adjusted_range"]
    print("condition1:", condition1)
    df.loc[condition1, "p_l_bound_opt"] = df["p_l_bound_init"] - df["min_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )


    df["min_dist"] = df["min"] - df["p_l_bound_init"]
    condition2 = df["min_dist"] < minmaxscale * df["adjusted_range"]
    print("condition2:", condition2)
    df.loc[condition2, "p_l_bound_min"] = df["p_l_bound_init"] - df["min_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # Use the minimum of the calculated bounds
    df["p_l_bound"] = df[["p_l_bound_opt", "p_l_bound_min", "p_l_bound_init"]].min(axis=1)
    # leave message in column 'remarks', which bound was used
    # df["remark"] = (
    #     df[["p_l_bound_opt", "p_l_bound_min", "p_l_bound_init"]]
    #     .idxmin(axis=1)
    #     .apply(
    #         lambda x: "opt"
    #         if x == "p_l_bound_opt"
    #         else ("min" if x == "p_l_bound_min" else "standard")
    #     )
    # )

    df["remark_l"] = df[["p_l_bound_opt", "p_l_bound_min", "p_l_bound_init"]].apply(lambda row: create_remark(row, 'l'), axis=1)





    # Correct the upper bounds, if necessary
    # If optimized value is too close to the upper bound
    df["max_dist"] = df["p_u_bound_init"] - df["opt_val"]
    condition3 = df["max_dist"] < optscale * df["adjusted_range"]
    print("condition3:", condition3)
    df.loc[condition3, "p_u_bound_opt"] = df["u_bound"] + df["max_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # If maximum value is too close to the upper bound
    df["max_dist"] = df["u_bound"] - df["max"]
    condition4 = df["max_dist"] < minmaxscale * df["adjusted_range"]
    print("condition4:", condition4)
    df.loc[condition4, "p_u_bound_max"] = df["u_bound"] + df["max_dist"] * (
        a_bound * df["ratio_stddev"] + b_bound
    )

    # Use the maximum of the calculated bounds
    df["p_u_bound"] = df[["p_u_bound_opt", "p_u_bound_max", "p_u_bound_init"]].max(axis=1)
    df["remark_u"] = df[["p_u_bound_opt", "p_u_bound_max", "p_u_bound_init"]].apply(lambda row: create_remark(row, 'u'), axis=1)
    
    # # Correct the lower bounds, if necessary
    # # if optimized value is too close to the lower bound
    # df['min_dist'] = df['opt_val'] - df['u_bound']

    # # if minimum value is too close to the lower bound
    # df['min_dist'] = df['min'] - df['u_bound']
    # if df['min_dist'] < 0.1 * df['adjusted_range']:
    #     df['p_l_bound_min'] = df['u_bound'] - df['min_dist'] * (a_bound * df['ratio_stddev'] + b_bound)

    # # use the minimum of the calculated bounds
    # df['p_l_bound'] = df[['p_l_bound_opt', 'p_l_bound_min', 'p_l_bound']].min(axis=1)

    # # Correct the upper bounds, if necessary
    # # if optimized value is too close to the upper bound
    # df['max_dist'] = df['u_bound'] - df['opt_val']
    # if df['max_dist'] < 0.2 * df['adjusted_range']:
    #     df['p_u_bound_opt'] = df['u_bound'] + df['max_dist'] * (a_bound * df['ratio_stddev'] + b_bound)
    # # if maximum value is too close to the upper bound
    # df['max_dist'] = df['u_bound'] - df['max']
    # if df['max_dist'] < 0.1 * df['adjusted_range']:
    #     df['p_u_bound_max'] = df['u_bound'] + df['max_dist'] * (a_bound * df['ratio_stddev'] + b_bound)

    # df['p_u_bound'] = df[['p_u_bound_opt', 'p_u_bound_max', 'p_u_bound']].max(axis=1)

    return df[["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]].round(3)


# Read the parameter names from the 'params_indexed.csv' file
params = pd.read_csv("config/params_indexed.csv")
params.columns = params.columns.str.strip()
param_names = (
    params["parameter1"].str.strip() + "_" + params["parameter2"].str.strip()
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
df_scaled = df_scaled[df_scaled["evals"] >= 10000]

# Read the parameter file
config_path = "config/params_indexed.csv"
param_keys, initial_values, l_bound, u_bound, data_types = read_parameter_file(
    config_path
)

cma_lower_bounds = 0
cma_upper_bounds = 10
# for each row in data, unscale the variables (columns 5 to end)
# unscaled_candidates = unscale_variables(candidate, l_bound, u_bound, cma_lower_bounds, cma_upper_bounds)

# Apply the unscale function to each row of the data DataFrame
df_unscaled = df_scaled.apply(unscale_row, axis=1)
print("unscaled_data head:\n", df_unscaled.head())


df_analysed = df_unscaled[column_names[5:]].describe().T.round(3)
df_analysed = df_analysed.drop(["count", "std", "25%", "50%", "75%"], axis=1)

# add row with min fitness value to describe_unscaled_data
min_fitness_row = df_unscaled.loc[df_unscaled["fitness"].idxmin()]
df_analysed["opt_val"] = min_fitness_row.T.round(3)
df_analysed["u_bound"] = u_bound
df_analysed["l_bound"] = l_bound


# read stddev from file
with open("outcmaes/stddev.dat") as f:
    last_line = f.readlines()[-1]

stddevs = last_line.split(" ")[5:]

# calculate proposed bounds
df_analysed[
    ["ratio_stddev", "p_l_bound", "p_u_bound", "remark_l", "remark_u"]
] = calculate_proposed_bounds(df_analysed, stddevs)

df_analysed = df_analysed[['mean', 'min', 'max', 'opt_val', 'l_bound', 'u_bound', 'ratio_stddev', 'p_l_bound', 'p_u_bound', 'remark_l', 'remark_u']]
print("variable analysis:\n", df_analysed)

df_analysed.to_csv("analyze_data/variable_analysis.csv", index=True)

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
