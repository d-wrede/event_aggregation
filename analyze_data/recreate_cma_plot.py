import cma

# Replace 'filename_prefix' with the appropriate prefix used during optimization
filename_prefix = "../outcmaes"

# Load the CMADataLogger object with the saved data
data_logger = cma.CMADataLogger(filename_prefix).load()

# Recreate the plot
data_logger.plot()
