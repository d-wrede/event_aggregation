import pandas as pd
import re
from optimize_parameters import read_parameter_file, unscale_variables


def unscale_row(row):
    # Extract the variables to be unscaled
    candidate = row[column_names[5:]]
    
    # Unscaled the variables
    unscaled_candidates = unscale_variables(candidate, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound)
    
    # Replace the scaled variables with the unscaled variables
    row[column_names[5:]] = unscaled_candidates
    
    return row


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

column_names_str = re.search(r'"(.*?)"', first_line).group(1)
column_names = column_names_str.split(", ")[:5] + param_names
# alternatively:
# column_names_str = first_line.split('"')[1]
# column_names = column_names_str.split(', ')[:5] + param_names

print("Column names:", column_names)
# Read the data from the file, skipping the first row (header)
data = pd.read_csv(
    "outcmaes/xrecentbest.dat",
    delim_whitespace=True,
    skiprows=1,
    header=None,
    index_col=False,
    names=column_names,
)

print("data head1:\n", data.head())

# Exclude the initial iterations (e.g., rows with 'evals' less than 5000)
data = data[data["evals"] >= 5000]

# Read the parameter file
config_path = "config/params_indexed.csv"
param_keys, initial_values, lower_bounds, upper_bounds, data_types = read_parameter_file(
    config_path
)

cma_lower_bound = 0
cma_upper_bound = 10
# for each row in data, unscale the variables (columns 5 to end)
#unscaled_candidates = unscale_variables(candidate, lower_bounds, upper_bounds, cma_lower_bound, cma_upper_bound)

# Apply the unscale function to each row of the data DataFrame
data = data.apply(unscale_row, axis=1)

print("describe data:\n", data[column_names[5:]].describe().T)
data[column_names[5:]].describe().T.round(3).to_csv('analyze_data/data_analyzed.csv', index=True)

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
print("describe data:", data.describe())
# Add the tendency information to the summary DataFrame
for idx, col_name in enumerate(tendency.columns):
    summary[f"Tendency_{(idx+1)*10}%"] = tendency[col_name].values

# Display the results
print("summary head1:\n", summary.head())

# Calculate additional statistics
additional_stats = data[column_names[5:]].describe().loc[["25%", "50%", "75%", "std"]]
summary = summary.join(additional_stats.T)
print("summary head2:\n", summary.head())
summary.to_csv('analyze_data/summary.csv', index=True)

# Convert the summary DataFrame to a readable string
summary_str = summary.to_string()

# Print the summary DataFrame
print("Summary:")
print(summary_str)
