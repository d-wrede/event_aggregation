import os
import re
import numpy as np

log_files = [
    "fit.dat",
    "xrecentbest.dat",
    "axlen.dat",
    "xmean.dat",
    "axlencorr.dat",
    "sigvec.dat",
    "axlenprec.dat",
    "stddev.dat",
]

optimization_options_file = "optimize_parameters.py"
optimization_summary_file = 'outcmaes/optimization_summary.json'
output_file = "analyze_data/summary.txt"

text_intro = "I have these optimization results, see below. It terminated on stagnation. I would like you to analyse and interpret the optimization-performance and give specific advice, for how to improve the optimization performance. Please be as specific as possible."

def extract_first_last_lines(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        first_line = lines[0].strip()
        last_line = lines[-1].strip()

    if file_path.endswith("fit.dat") or file_path.endswith("xrecentbest.dat"):
        data = np.loadtxt(file_path, skiprows=1)

        if file_path.endswith("fit.dat"):
            best_ever_fitness = data[:, 4]
            best_fitness = data[:, 5]
            median_fitness = data[:, 6]

            min_best_ever = np.min(best_ever_fitness)
            max_best_ever = np.max(best_ever_fitness)
            mean_best_ever = np.mean(best_ever_fitness)
            median_best_ever = np.median(best_ever_fitness)

            min_best = np.min(best_fitness)
            max_best = np.max(best_fitness)
            mean_best = np.mean(best_fitness)
            median_best = np.median(best_fitness)

            min_median = np.min(median_fitness)
            max_median = np.max(median_fitness)
            mean_median = np.mean(median_fitness)
            median_median = np.median(median_fitness)

            return (
                first_line,
                last_line,
                (min_best_ever, max_best_ever, mean_best_ever, median_best_ever),
                (min_best, max_best, mean_best, median_best),
                (min_median, max_median, mean_median, median_median),
            )

        elif file_path.endswith("xrecentbest.dat"):
            fitness_values = data[:, 4]

            min_value = np.min(fitness_values)
            max_value = np.max(fitness_values)
            mean_value = np.mean(fitness_values)
            median_value = np.median(fitness_values)

            return (
                first_line,
                last_line,
                (min_value, max_value, mean_value, median_value),
                None,
                None,
            )
    else:
        return first_line, last_line, None, None, None


def extract_options(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        options_pattern = re.compile(
            r"options\s*=\s*\{[^}]*\}", re.MULTILINE | re.DOTALL
        )
        match = options_pattern.search(content)
        if match:
            return match.group()
        else:
            return None


def read_optimization_summary(file_path):
    with open(file_path, 'r') as file:
        optimization_summary = json.load(file)
    return optimization_summary


if os.path.exists(optimization_summary_file):
    optimization_summary = read_optimization_summary(optimization_summary_file)
else:
    optimization_summary = None

with open(output_file, 'w') as outfile:

    outfile.write(text_intro + "\n\n")
    if optimization_summary:
        outfile.write("Optimization summary:\n")
        outfile.write(f"Stop: {optimization_summary['stop']}\n")
        outfile.write(f"Result: {optimization_summary['result']}\n\n")
    for log_file in log_files:
        file_path = os.path.join("outcmaes", log_file)
        if os.path.exists(file_path):
            (
                first_line,
                last_line,
                best_ever_stats,
                best_stats,
                median_stats,
            ) = extract_first_last_lines(file_path)
            outfile.write(f"{log_file}:\n")
            outfile.write(f"First line: {first_line}\n")
            outfile.write(f"Last line: {last_line}\n")
            if best_ever_stats:
                outfile.write(
                    f"Best-ever fitness (min, max, mean, median): {best_ever_stats}\n"
                )
            if best_stats:
                outfile.write(f"Best fitness (min, max, mean, median): {best_stats}\n")
            if median_stats:
                outfile.write(
                    f"Median fitness (min, max, mean, median): {median_stats}\n"
                )

            outfile.write("\n")
        else:
            print(f"{file_path} not found.")

    options_text = extract_options(optimization_options_file)
    if options_text:
        outfile.write(f"Options from {optimization_options_file}:\n{options_text}\n")
    else:
        print(f"Options not found in {optimization_options_file}.")
