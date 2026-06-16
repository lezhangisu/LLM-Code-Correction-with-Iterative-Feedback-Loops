"""
leetcode.py
===========

LeetcodeAPI class for interacting with leetcode.com:
  1. Fetch problem lists by category, topic, or curated list name
  2. Submit code solutions to leetcode.com
  3. Retrieve submission results (status, test cases, errors)

Uses curl via subprocess to communicate with the Leetcode GraphQL API.
Requires Leetcode session credentials in config/api_keys.yaml.
"""

import json
import subprocess
import time
from .utils import to_shell_escape
from .config_loader import get_config_section


# GraphQL query for fetching problem lists
_PROBLEM_LIST_QUERY = (
    '{"query":"query problemsetQuestionList('
    "$categorySlug: String, $limit: Int, $skip: Int, "
    "$filters: QuestionListFilterInput) {"
    "problemsetQuestionList: questionList("
    "categorySlug: $categorySlug limit: $limit skip: $skip filters: $filters) {"
    "total: totalNum questions: data {"
    "acRate difficulty freqBar frontendQuestionId: questionFrontendId "
    "isFavor paidOnly: isPaidOnly status title titleSlug "
    "topicTags {name id slug} hasSolution hasVideoSolution"
    '}}}",'
    '"variables":{"categorySlug":"%s","skip":0,"limit":%d,'
    '"filters":%s},"operationName":"problemsetQuestionList"}'
)

# Map of curated list names to their Leetcode list IDs
CURATED_LISTS = {
    "LeetCode Curated Algo 170": "552y65ke",
    "LeetCode Curated SQL 70": "5htp6xyg",
    "Top 100 Liked Questions": "79h8rn6",
    "Top Amazon Questions": "7p5x763",
    "Top Facebook Questions": "7p59281",
    "Top Google Questions": "7p55wqm",
    "Top Interview Questions": "wpwgkgt",
    "Top Microsoft Questions": "55vr69d7",
}

PROBLEM_CATEGORIES = ["algorithms", "javascript", "concurrency"]
PROBLEM_TOPICS = ["greedy", "sorting", "tree", "binary-search"]


class LeetcodeAPI:
    """Interface to leetcode.com via curl-based GraphQL requests."""

    def __init__(self):
        self.csrftoken = None
        self.leetcode_session = None
        self.user_agent = None
        self.load_cookies_from_config()

    def load_cookies_from_config(self):
        """Load Leetcode session credentials from config/api_keys.yaml."""
        try:
            leetcode_config = get_config_section("leetcode")
            self.csrftoken = leetcode_config.get("csrftoken")
            self.leetcode_session = leetcode_config.get("leetcode_session")
            self.user_agent = leetcode_config.get("user_agent", "Mozilla/5.0")
        except Exception as e:
            print(f"Error loading Leetcode credentials from config: {e}")
            raise

    def _curl(self, cmd):
        """Execute a curl command and return the response string."""
        if not cmd.startswith("curl"):
            return None
        process = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True
        )
        return process.stdout.decode("utf-8")

    def post_query(self, url, referer_url="", if_x_csrftoken=False, query=""):
        """Send an HTTPS POST request to the Leetcode GraphQL API.

        Args:
            url: Target URL.
            referer_url: Referer header value (defaults to problemset page).
            if_x_csrftoken: Whether to include the x-csrftoken header.
            query: JSON-encoded GraphQL query string.

        Returns:
            JSON response string, or None on failure.
        """
        if not referer_url:
            referer_url = "https://leetcode.com/problemset/"

        x_csrftoken_str = (
            f"-H 'x-csrftoken:{self.csrftoken}' " if if_x_csrftoken else ""
        )
        template = (
            "curl '{url}' "
            "-H 'content-type: application/json' "
            "-b 'csrftoken={csrf}; LEETCODE_SESSION={session}' "
            "-H 'referer: {referer}' "
            "-H 'user-agent: {ua}' "
            "{xcsrf}"
            "--data-raw '{query}'"
        )
        curl_cmd = template.format(
            url=url,
            csrf=self.csrftoken,
            session=self.leetcode_session,
            referer=referer_url,
            ua=self.user_agent,
            xcsrf=x_csrftoken_str,
            query=query,
        )

        max_retries = 3
        retry_delay = 30
        for attempt in range(max_retries):
            result = self._curl(curl_cmd)
            try:
                json.loads(result)
                return result
            except (ValueError, TypeError):
                print(f"CURL attempt {attempt + 1} returned invalid JSON")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached.")
        return None

    # ------------------------------------------------------------------
    # Problem list fetching
    # ------------------------------------------------------------------

    def _fetch_problem_list(self, category_slug, filters_json, use_csrftoken):
        """Internal helper: fetch full problem list in two steps.

        First fetches with limit=1 to get total count, then fetches all.
        """
        url = "https://leetcode.com/graphql/"

        # Step 1: get total count
        query = _PROBLEM_LIST_QUERY % (category_slug, 1, filters_json)
        result = self.post_query(url, "", use_csrftoken, query)
        try:
            total = json.loads(result)["data"]["problemsetQuestionList"]["total"]
        except (ValueError, TypeError, KeyError):
            print("ERROR: Could not parse problem list response")
            return None

        time.sleep(5)

        # Step 2: fetch all problems
        query = _PROBLEM_LIST_QUERY % (category_slug, total, filters_json)
        result = self.post_query(url, "", use_csrftoken, query)
        try:
            return json.loads(result)
        except (ValueError, TypeError):
            print("ERROR: Could not parse full problem list response")
            return None

    def get_prob_list_json_by_name(self, list_name):
        """Fetch a curated problem list by name.

        Args:
            list_name: One of the keys in CURATED_LISTS.

        Returns:
            JSON response with problem data, or None.
        """
        if list_name not in CURATED_LISTS:
            print(f"ERROR: Invalid list name. Accepted: {list(CURATED_LISTS.keys())}")
            return None
        list_id = CURATED_LISTS[list_name]
        filters = f'{{"listId":"{list_id}"}}'
        return self._fetch_problem_list("all-code-essentials", filters, False)

    def get_prob_list_json_by_category(self, category):
        """Fetch problems by category slug.

        Args:
            category: One of PROBLEM_CATEGORIES.
        """
        if category not in PROBLEM_CATEGORIES:
            print(f"ERROR: Invalid category. Accepted: {PROBLEM_CATEGORIES}")
            return None
        return self._fetch_problem_list(category, "{}", True)

    def get_prob_list_json_by_topic(self, topic):
        """Fetch problems by topic tag.

        Args:
            topic: One of PROBLEM_TOPICS.
        """
        if topic not in PROBLEM_TOPICS:
            print(f"ERROR: Invalid topic. Accepted: {PROBLEM_TOPICS}")
            return None
        filters = f'{{"orderBy":"FRONTEND_ID","sortOrder":"ASCENDING","tags":["{topic}"]}}'
        return self._fetch_problem_list("all-code-essentials", filters, True)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_answer(self, title_slug, question_id, language, code):
        """Submit a code solution to Leetcode.

        Args:
            title_slug: Problem titleSlug (e.g., 'two-sum').
            question_id: Numeric question ID as string.
            language: Programming language slug (e.g., 'python3', 'java', 'cpp').
            code: Source code string.

        Returns:
            Submission ID (int) on success, or None on failure.
        """
        url = f"https://leetcode.com/problems/{title_slug}/"
        referer_url = url + "description/"
        submit_url = url + "submit/"
        query = '{{"lang":"{}","question_id":"{}","typed_code":"{}"}}'.format(
            language, question_id, to_shell_escape(code)
        )
        result = self.post_query(submit_url, referer_url, True, query)
        try:
            return json.loads(result)["submission_id"]
        except (TypeError, KeyError):
            print(f"Submission error: {result}")
            return None

    def get_submission_info(self, submission_id):
        """Fetch basic submission details (status, percentiles, counts).

        Args:
            submission_id: Submission ID returned by submit_answer().

        Returns:
            JSON response string, or None on failure.
        """
        query = (
            '{"query":"query submissionDetails($submissionId: Int!){'
            "submissionDetails(submissionId: $submissionId){"
            "user {username} statusCode runtimePercentile memoryPercentile "
            "lang {name verboseName} question {questionId titleSlug} "
            "runtimeError compileError totalCorrect totalTestcases"
            '}}","variables":{"submissionId":%s},'
            '"operationName":"submissionDetails"}' % submission_id
        )
        result = self.post_query("https://leetcode.com/graphql/", "", False, query)
        try:
            json.loads(result)
            return result
        except (TypeError, ValueError):
            print(f"get_submission error: {result}")
            return None

    def get_submission_info_detailed(self, submission_id):
        """Fetch detailed submission info including failing test case data.

        Returns the full submission response with lastTestcase,
        codeOutput, and expectedOutput fields.
        """
        query = (
            '{"query":"query submissionDetails($submissionId: Int!){'
            "submissionDetails(submissionId: $submissionId){"
            "user {username} code statusCode runtimePercentile memoryPercentile "
            "lang {name verboseName} question {questionId titleSlug} "
            "runtimeError compileError lastTestcase codeOutput expectedOutput "
            "totalCorrect totalTestcases"
            '}}","variables":{"submissionId":%s},'
            '"operationName":"submissionDetails"}' % submission_id
        )
        result = self.post_query("https://leetcode.com/graphql/", "", False, query)
        try:
            json.loads(result)
            return result
        except (TypeError, ValueError):
            print(f"get_submission_detailed error: {result}")
            return None
