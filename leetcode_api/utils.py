"""
utils.py
========

Helper functions for the iterative feedback loop benchmark:
  - Text processing: shell escaping, HTML-to-text, code extraction
  - Data extraction: problem metadata parsing, code block extraction
  - File I/O: JSON, text, and list read/write utilities
"""

import re
import random
import os
import json
from bs4 import BeautifulSoup


# ======================================================================
# Text Processing
# ======================================================================

def to_shell_escape(s):
    """Escape special characters in a string for safe shell embedding."""
    escape_map = {
        "\n": r"\n",
        "\t": r"\t",
        "\r": r"\r",
        "\\": r"\\",
        "'": "'\\''",
        '"': r"\"",
    }
    return "".join(escape_map.get(c, c) for c in s)


def html_to_text(html_string):
    """Convert HTML to plain text, preserving superscript markers."""
    soup = BeautifulSoup(html_string, "html.parser")
    for sup in soup.find_all("sup"):
        sup.replace_with(f"^{sup.get_text()}")
    return soup.get_text()


def trim_str_after(string, target_substr):
    """Return the portion of string before the first occurrence of target_substr."""
    return string.split(target_substr)[0].strip()


def trim_str_before(string, target_substr):
    """Return the portion of string starting from the first occurrence of target_substr."""
    index = string.find(target_substr)
    return string[index:] if index != -1 else string


# ======================================================================
# Code Extraction
# ======================================================================

def extract_code_block_by_lang(text, language):
    """Extract the first fenced code block for the specified language.

    Handles common aliases: python3 -> python, golang -> go.
    Falls back to the raw text if no code block is found.
    """
    lang_alias = language
    if language == "python3" and f"```{language}" not in text:
        lang_alias = "python"
    elif language == "golang" and f"```{language}" not in text:
        lang_alias = "go"

    pattern = rf"```{lang_alias}\n(.*?)(?:```|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print(f"ERROR: code block not found for language {language}")
    return text


def extract_code_block(markdown_text):
    """Extract the first fenced code block from markdown text.

    Ignores anything after a ```json block (token info section).
    """
    pattern = r"```(\w*)\n(.*?)\n```"
    text = trim_str_after(markdown_text, "```json")
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0][1].strip()
    return text


def trim_text_at_closing_brace(text):
    """Trim text to end at the first line starting with '};'.

    Returns the original text if no such line exists.
    """
    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        result.append(line)
        if line.startswith("};"):
            return "".join(result)
    return text


def extract_clean_src_code(text, lang):
    """Extract the relevant solution code from an LLM response.

    Strips boilerplate like 'func main(', 'int main(', 'def main(',
    and test-case sections based on the target language.
    """
    if lang == "golang":
        ret = trim_str_after(trim_str_before(text, "func "), "func main(")
    elif lang == "cpp":
        ret = trim_str_after(trim_str_before(text, "class Solution"), "int main(")
    elif lang == "python3":
        ret = trim_str_after(trim_str_before(text, "class Solution:"), "def main(")
    elif lang == "java":
        ret = trim_str_before(text, "class Solution")
    else:
        ret = text

    return trim_str_after(ret, "# Test case")


# ======================================================================
# Problem Metadata Extraction
# ======================================================================

def json_metadata_to_prob_list(json_data):
    """Extract free problem titleSlug list from Leetcode JSON metadata."""
    try:
        prob_list = json_data["data"]["problemsetQuestionList"]["questions"]
    except (KeyError, TypeError):
        print("ERROR: invalid JSON metadata")
        return None
    return [p["titleSlug"] for p in prob_list if not p["paidOnly"]]


def json_metadata_to_prob_list_with_diff(json_data):
    """Extract (titleSlug, difficulty) tuples from Leetcode JSON metadata."""
    try:
        prob_list = json_data["data"]["problemsetQuestionList"]["questions"]
    except (KeyError, TypeError):
        print("ERROR: invalid JSON metadata")
        return None
    return [(p["titleSlug"], p["difficulty"]) for p in prob_list if not p["paidOnly"]]


def prob_metadata_to_url_list(file_path_in, data_dir):
    """Build a list of problem URLs from metadata, skipping already-fetched ones."""
    existing = get_file_names_in_dir(data_dir) if data_dir else []
    json_data = file_to_json(file_path_in)
    prob_list = json_metadata_to_prob_list(json_data) or []
    return [
        f"https://leetcode.com/problems/{prob}/description/"
        for prob in prob_list if prob not in existing
    ]


def merge_prob_list_by_topics(dir_path, topic_list, skip, count):
    """Merge and deduplicate problem lists from multiple topic JSON files.

    Args:
        dir_path: Directory containing prob_topic_<topic>.json files.
        topic_list: List of topic names.
        skip: Number of problems to skip from the start.
        count: Number of problems to return (0 = all).
    """
    prob_set = set()
    for topic in topic_list:
        path = f"{dir_path}prob_topic_{topic}.json"
        prob_set.update(json_metadata_to_prob_list(file_to_json(path)) or [])
    prob_list = sorted(prob_set)
    if skip == 0 and count == 0:
        return prob_list
    return prob_list[skip : skip + count]


def random_sample_from_list(entire_list, sample_count):
    """Return a random sample of sample_count items from the list."""
    return random.sample(entire_list, sample_count)


# ======================================================================
# File I/O
# ======================================================================

def file_exists(file_path):
    """Check if a file exists at the given path."""
    return os.path.exists(file_path)


def file_to_json(file_path):
    """Read a JSON file and return the parsed object."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading {file_path}: {e}")
        return None


def json_to_file(data, file_path):
    """Write a JSON-serializable object to a file with pretty formatting."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except TypeError as e:
        print(f"TypeError: {e}")
    except Exception as e:
        print(f"Error writing {file_path}: {e}")


def file_to_str(file_path):
    """Read a file and return its content as a string."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None


def get_file_names_in_dir(file_dir):
    """Get all JSON file names (without .json extension) in a directory."""
    file_list = []
    for entry in os.listdir(file_dir):
        if entry.endswith(".json"):
            full_path = os.path.join(file_dir, entry)
            if os.path.isfile(full_path):
                file_list.append(entry.split(".")[0])
    return file_list


def load_prob_info_file(file_path):
    """Load problem info from a JSON file in the problem pool.

    Returns a dict with titleSlug, questionId, content, difficulty,
    stats, codeSnippets, and exampleTestcaseList.
    """
    json_data = file_to_json(file_path)
    if not json_data or "question" not in json_data:
        return None
    q = json_data["question"]
    return {
        "titleSlug": q["titleSlug"],
        "questionId": q["questionId"],
        "content": q["content"],
        "difficulty": q["difficulty"],
        "stats": q["stats"],
        "codeSnippets": q["codeSnippets"],
        "exampleTestcaseList": q["exampleTestcaseList"],
    }


def load_ans_src_code_file(file_path, lang):
    """Load and extract clean source code from an answer file."""
    code = extract_code_block_by_lang(file_to_str(file_path), lang)
    return extract_clean_src_code(code, lang)


def list_to_file(str_list, file_path):
    """Write a list of strings to a file, one per line."""
    try:
        with open(file_path, "w") as f:
            for s in str_list:
                f.write(s + "\n")
    except Exception as e:
        print(f"Error writing list to {file_path}: {e}")


def file_to_list(file_path):
    """Read a file into a list of stripped lines."""
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
