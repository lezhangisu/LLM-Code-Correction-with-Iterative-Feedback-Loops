"""
prob_list_gen.py
================

Generate problem lists for experiments by sampling from topic-based pools.

Supports stratified random sampling to maintain difficulty distribution
across Easy, Medium, and Hard problems.

Usage:
    python prob_list_gen.py --topics greedy,binary-search --count 32 --output data/problem_lists/sample.txt
"""

import argparse
import random
from collections import defaultdict
import leetcode_api as lc


def stratified_random_sample(data, k):
    """Perform stratified random sampling on a list of (name, difficulty) tuples.

    Maintains the proportion of each difficulty level in the sample.

    Args:
        data: List of tuples in format [(name, difficulty), ...]
        k: Number of samples to draw.

    Returns:
        List of sampled tuples with length k.
    """
    if k <= 0 or k > len(data):
        raise ValueError("k must be positive and not exceed list length")

    # Group by difficulty
    groups = defaultdict(list)
    for name, difficulty in data:
        groups[difficulty].append((name, difficulty))

    # Calculate proportional sample size for each group
    total_size = len(data)
    sample = []

    for difficulty, items in groups.items():
        group_size = len(items)
        # Round to nearest integer for proportional allocation
        group_sample_size = round(k * group_size / total_size)

        # Ensure at least 1 sample from non-empty groups
        if group_sample_size == 0 and group_size > 0:
            group_sample_size = 1

        # Sample from this group
        group_sample = random.sample(items, min(group_sample_size, len(items)))
        sample.extend(group_sample)

    # Adjust if total sample size doesn't match k due to rounding
    if len(sample) < k:
        remaining = [item for item in data if item not in sample]
        additional = random.sample(remaining, k - len(sample))
        sample.extend(additional)
    elif len(sample) > k:
        sample = random.sample(sample, k)

    return sample


def get_problem_list_with_difficulty(dir_path, topic_list):
    """Load and merge problem lists from multiple topic JSON files.

    Args:
        dir_path: Directory containing prob_topic_<topic>.json files.
        topic_list: List of topic names.

    Returns:
        List of (titleSlug, difficulty) tuples.
    """
    prob_set = set()
    for topic in topic_list:
        json_path = f"{dir_path}prob_topic_{topic}.json"
        json_data = lc.file_to_json(json_path)
        if json_data:
            prob_set.update(lc.json_metadata_to_prob_list_with_diff(json_data))
    return list(prob_set)


def generate_sample(topics, count, output_file, base_filter_file=None):
    """Generate a stratified random sample of problems and save to file.

    Args:
        topics: List of topic names to sample from.
        count: Number of problems to sample.
        output_file: Path to save the problem list (one titleSlug per line).
        base_filter_file: Optional path to a file with problems to restrict sampling to.
    """
    dir_path = "data/problem_lists/"

    # Load all problems with difficulty info
    all_problems = get_problem_list_with_difficulty(dir_path, topics)

    # Optionally filter to a subset
    if base_filter_file:
        base_list = lc.file_to_list(base_filter_file)
        all_problems = [p for p in all_problems if p[0] in base_list]

    # Perform stratified sampling
    sample = stratified_random_sample(all_problems, count)

    # Extract just the problem names
    output_list = [item[0] for item in sample]

    # Save to file
    lc.list_to_file(output_list, output_file)
    print(f"Generated {count} problems from topics {topics}")
    print(f"Saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate problem lists for experiments using stratified sampling."
    )
    parser.add_argument("--topics", default="greedy,binary-search,sorting,tree",
                        help="Comma-separated list of topics to sample from")
    parser.add_argument("--count", type=int, default=32,
                        help="Number of problems to sample")
    parser.add_argument("--output", default="data/problem_lists/sample_list.txt",
                        help="Output file path for the problem list")
    parser.add_argument("--filter", default=None,
                        help="Optional: filter problems to only those in this file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    topics = [t.strip() for t in args.topics.split(",")]

    generate_sample(
        topics=topics,
        count=args.count,
        output_file=args.output,
        base_filter_file=args.filter,
    )
