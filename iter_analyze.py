"""
iter_analyze.py
===============

Analyze iterative experiment results and generate summary CSV files.

Reads the conversation logs and submission results from iter_exp.py,
extracts key metrics (status codes, test case counts), and produces
a CSV table for further analysis.

Output format:
    model, language, problem_name, loop_1, loop_2, ..., loop_10,
    testcase@1, testcase@2, ..., testcase@10, total_testcase

Usage:
    python iter_analyze.py --prob_list <file> --output <csv_path>
"""

import os
import argparse
import pandas as pd
import leetcode_api as lc


MAX_ITERATIONS = 10


def analyze_results(prob_list, lang_list, model_list, input_dir, output_csv):
    """Process all experiment results and generate a summary CSV.

    Args:
        prob_list: List of problem titleSlugs.
        lang_list: List of programming languages.
        model_list: List of model identifiers.
        input_dir: Base directory containing experiment results.
        output_csv: Path for the output CSV file.
    """
    rows = []

    for model_name in model_list:
        for lang in lang_list:
            for prob_name in prob_list:
                results_file = os.path.join(
                    input_dir, model_name, lang, f"{prob_name}_results.json"
                )

                if not os.path.exists(results_file):
                    print(f"[WARNING] Missing results: {model_name}/{lang}/{prob_name}")
                    continue

                json_data = lc.file_to_json(results_file)
                if not json_data:
                    continue

                # Initialize row: status codes and testcase counts for each iteration
                status_codes = [""] * MAX_ITERATIONS
                testcase_counts = [""] * MAX_ITERATIONS
                total_testcases = 0

                # Parse the results file
                for i in range(len(json_data)):
                    if i >= MAX_ITERATIONS * 2:
                        break

                    # Each iteration has a {"loop": N} marker followed by result data
                    if "loop" in json_data[i]:
                        loop_num = json_data[i]["loop"]

                        # Check if this iteration has result data
                        if i + 1 >= len(json_data):
                            print(f"[WARNING] Incomplete iteration {loop_num} for "
                                  f"{model_name}/{lang}/{prob_name}")
                            break

                        # Extract submission details
                        details = json_data[i + 1].get("data", {}).get("submissionDetails", {})
                        if details:
                            status_codes[loop_num - 1] = str(details.get("statusCode", ""))
                            testcase_counts[loop_num - 1] = str(details.get("totalCorrect", ""))
                            total_testcases = details.get("totalTestcases", 0)

                # Build the row
                row = {
                    "model": model_name,
                    "language": lang,
                    "problem_name": prob_name,
                }

                # Add loop status columns
                for i in range(MAX_ITERATIONS):
                    row[f"loop_{i+1}"] = status_codes[i]

                # Add testcase count columns
                for i in range(MAX_ITERATIONS):
                    row[f"testcase@{i+1}"] = testcase_counts[i]

                row["total_testcase"] = total_testcases
                rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print(f"Total problems analyzed: {len(rows)}")
    print(df.head())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze iterative experiment results and generate CSV summary."
    )
    parser.add_argument("--prob_list", required=True,
                        help="Path to problem list file")
    parser.add_argument("--lang", nargs="+", default=["python3", "java"],
                        help="Programming languages to analyze")
    parser.add_argument("--model", nargs="+", default=["deepseek_v3"],
                        help="Model names to analyze")
    parser.add_argument("--input_dir", default="data/iter_exp/",
                        help="Base directory containing experiment results")
    parser.add_argument("--output", default="data/analysis/iter_results.csv",
                        help="Output CSV file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prob_list = lc.file_to_list(args.prob_list)
    if not prob_list:
        print(f"ERROR: Could not load problem list from {args.prob_list}")
        exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    analyze_results(
        prob_list=prob_list,
        lang_list=args.lang,
        model_list=args.model,
        input_dir=args.input_dir,
        output_csv=args.output,
    )
