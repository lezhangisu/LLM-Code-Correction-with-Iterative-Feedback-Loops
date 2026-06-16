"""
analyzer.py
===========

ResultAnalyzer class for post-experiment data processing:
  1. Load problem metadata (questionId, titleSlug, difficulty, category)
  2. Load submission results (status codes, percentiles, test case counts)
  3. Load token usage info (prompt/completion tokens)
  4. Merge all data into a unified DataFrame
  5. Export to CSV
"""

import os
import re
import json
import pandas as pd
from .utils import file_to_json, file_to_str


# Leetcode status code mapping
STATUS_CODE_MAP = {
    10: "Accepted",
    11: "Wrong Answer",
    12: "Memory Limit Exceeded",
    14: "Time Limit Exceeded",
    15: "Runtime Error",
    20: "Compile Error",
}


class ResultAnalyzer:
    """Collects experimental results and merges them into analysis-ready tables."""

    def __init__(self, prob_list, lang_list, model_list,
                 prob_info_dir="data/problems/",
                 submission_info_dir="data/res_baseline/",
                 token_info_dir="data/ai_answers/"):
        """
        Args:
            prob_list: List of problem titleSlugs.
            lang_list: List of language slugs (e.g., ['python3', 'java']).
            model_list: List of model identifier strings.
            prob_info_dir: Directory containing problem metadata JSON files.
            submission_info_dir: Directory containing submission result JSON files.
            token_info_dir: Directory containing token usage text files.
        """
        self.prob_list = prob_list
        self.lang_list = lang_list
        self.model_list = model_list
        self.prob_info_dir = prob_info_dir
        self.submission_info_dir = submission_info_dir
        self.token_info_dir = token_info_dir

        self.prob_info_df = self._load_prob_info(prob_info_dir)
        self.submission_info_df = self._load_submission_info(submission_info_dir)
        self.token_info_df = self._load_token_info(token_info_dir)
        self.merged_df = self._create_merged_df()

    # ------------------------------------------------------------------
    # Problem info loading
    # ------------------------------------------------------------------

    def _load_prob_info(self, prob_dir):
        """Load problem metadata for all problems in self.prob_list."""
        rows = []
        for prob_name in self.prob_list:
            file_path = os.path.join(prob_dir, f"{prob_name}.json")
            if not os.path.exists(file_path):
                print(f"ERROR: {file_path} does not exist")
                return None
            json_data = file_to_json(file_path)
            q = json_data["question"]
            rows.append({
                "questionId": q["questionId"],
                "titleSlug": q["titleSlug"],
                "difficulty": q["difficulty"],
                "categoryTitle": q["categoryTitle"],
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Submission info loading
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_submission_file(file_path):
        """Parse a single submission JSON file into a flat dict."""
        json_data = file_to_json(file_path)
        if not json_data or "submission_id" not in json_data:
            print(f"ERROR: invalid submission data in {file_path}")
            return None

        details = json_data["data"]["submissionDetails"]
        if not details:
            print(f"Error: No submission details in {file_path}")
            return None

        return {
            "submission_id": json_data["submission_id"],
            "username": details.get("user", {}).get("username") if details.get("user") else None,
            "statusCode": details.get("statusCode"),
            "titleSlug": details["question"]["titleSlug"],
            "questionId": details["question"]["questionId"],
            "lang": details["lang"]["name"],
            "runtimePercentile": details.get("runtimePercentile"),
            "memoryPercentile": details.get("memoryPercentile"),
            "runtimeError": details.get("runtimeError"),
            "compileError": details.get("compileError"),
            "totalCorrect": details.get("totalCorrect"),
            "totalTestcases": details.get("totalTestcases"),
        }

    def _load_submission_info(self, submission_dir):
        """Load submission results for all model/lang/problem combinations."""
        rows = []
        for model in self.model_list:
            for lang in self.lang_list:
                for prob_name in self.prob_list:
                    file_path = os.path.join(submission_dir, model, lang, f"{prob_name}.json")
                    if not os.path.exists(file_path):
                        print(f"ERROR: {file_path} does not exist")
                        return None
                    info = self._parse_submission_file(file_path)
                    if info is None:
                        return None
                    rows.append({
                        "submissionId": info["submission_id"],
                        "modelName": model,
                        "language": lang,
                        "questionId": info["questionId"],
                        "titleSlug": info["titleSlug"],
                        "runtimePercentile": info["runtimePercentile"] or -1,
                        "memoryPercentile": info["memoryPercentile"] or -1,
                        "runtimeError": bool(info["runtimeError"]),
                        "compileError": bool(info["compileError"]),
                        "totalCorrect": info["totalCorrect"],
                        "totalTestcases": info["totalTestcases"],
                        "status": info["statusCode"],
                    })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Token info loading
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_token_info(markdown_text):
        """Extract prompt_tokens and completion_tokens from a text file.

        Expects a ```json ... ``` block at the end of the file.
        """
        matches = re.split(r"```json\n|```", markdown_text)
        if len(matches) > 1:
            return json.loads(matches[-2].strip())
        print("WARNING: no token info block found in text")
        return None

    def _load_token_info(self, token_info_dir):
        """Load token usage for all model/lang/problem combinations."""
        rows = []
        for model in self.model_list:
            for lang in self.lang_list:
                for prob_name in self.prob_list:
                    file_path = os.path.join(token_info_dir, model, lang, f"{prob_name}.txt")
                    if not os.path.exists(file_path):
                        print(f"ERROR: {file_path} does not exist")
                        return None
                    content = file_to_str(file_path)
                    info = self._extract_token_info(content)
                    rows.append({
                        "modelName": model,
                        "language": lang,
                        "titleSlug": prob_name,
                        "promptTokens": info["prompt_tokens"],
                        "completionTokens": info["completion_tokens"],
                    })
        return pd.DataFrame(rows)

    def _load_token_info_5in1(self, token_info_dir):
        """Load token usage from a combined (5-in-1) file structure.

        Alternative layout where tokens are stored per model/problem
        without language separation.
        """
        rows = []
        for model in self.model_list:
            for prob_name in self.prob_list:
                file_path = os.path.join(token_info_dir, model, "5in1", f"{prob_name}.txt")
                if not os.path.exists(file_path):
                    print(f"ERROR: {file_path} does not exist")
                    return None
                content = file_to_str(file_path)
                info = self._extract_token_info(content)
                rows.append({
                    "modelName": model,
                    "titleSlug": prob_name,
                    "promptTokens": info["prompt_tokens"],
                    "completionTokens": info["completion_tokens"],
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _create_merged_df(self):
        """Merge problem info, submission results, and token usage."""
        if any(df is None for df in [self.submission_info_df, self.prob_info_df, self.token_info_df]):
            print("ERROR: One or more DataFrames is None, cannot merge")
            return None
        merged = pd.merge(self.submission_info_df, self.prob_info_df, on="titleSlug", how="left")
        merged = pd.merge(merged, self.token_info_df, on=["modelName", "titleSlug"], how="left")
        return merged

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_prob_info_df(self):
        return self.prob_info_df

    def get_submission_info_df(self):
        return self.submission_info_df

    def get_merged_df(self):
        return self.merged_df

    def merged_df_to_csv(self, file_path):
        """Export the merged DataFrame to a CSV file."""
        if self.merged_df is None:
            print("ERROR: merged_df is None")
            return
        self.merged_df.to_csv(file_path, index=False)
