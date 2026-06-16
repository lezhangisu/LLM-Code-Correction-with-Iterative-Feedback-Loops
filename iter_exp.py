"""
iter_exp.py
===========

Iterative experiment runner for the feedback loop benchmark.

Workflow:
  1. Load a problem list from a text file
  2. For each problem, construct an initial prompt with the task description,
     code template, and example test cases
  3. Submit the LLM-generated code to leetcode.com and get evaluation feedback
  4. Feed the feedback back to the LLM for correction (up to max_iterations rounds)
  5. Save conversation logs and submission results as JSON files

Supports auto-resume: if an experiment is interrupted, it can pick up
from the last successful round.

Models evaluated in the paper:
  - deepseek_r1 (DeepSeek-R1)
  - deepseek_v3 (DeepSeek-V3)
  - o4_mini (GPT-o4-mini)
  - gpt_41_mini (GPT-4.1-mini)

Usage:
    python iter_exp.py --model <model_name> --lang <language> --prob_list <file>
"""

import json
import os
import time
import argparse
from collections import OrderedDict
import leetcode_api as lc


# ======================================================================
# Prompt Construction (matching the paper's experimental setup)
# ======================================================================

INITIAL_PROMPT_TEMPLATE = (
    "You are an experienced software developer. Implement a program in {lang} "
    "with the following information:\n"
    "## Task Descriptions: {description}\n"
    "## Code format:\n{snippet}\n"
    "## Testcases: {testcases}\n"
    "## Instructions: Ensure the code is well-formatted and adheres to best "
    "practices. Optimize your code with best time and space complexity, "
    "make sure the output is correct. Output the executable code only. "
    "Avoid unnecessary explanations or comment lines in your code."
)

FEEDBACK_TEMPLATES = {
    # Status 11: Wrong Answer
    "wrong_answer": (
        "Your code generated wrong outputs.\n"
        "Testcase: {testcase}\n"
        "Expected Output: {expected}\n"
        "Actual Output: {actual}\n"
        "Fix your code with above information. Output the executable code only. "
        "Avoid unnecessary explanations or comment lines in your code."
    ),
    "wrong_answer_long_testcase": (
        "Your code generated wrong outputs.\n"
        "Testcase is too long to show\n"
        "Expected Output: {expected}\n"
        "Actual Output: {actual}\n"
        "Fix your code with above information. Output the executable code only. "
        "Avoid unnecessary explanations or comment lines in your code."
    ),
    # Status 12: Memory Limit Exceeded
    "memory_limit": (
        "Your code exceeded the maximum memory allowance. Optimize the space "
        "complexity of your code to reduce the memory usage. Output the executable "
        "code only. Avoid unnecessary explanations or comment lines in your code."
    ),
    # Status 14: Time Limit Exceeded
    "time_limit": (
        "Your code exceeded the maximum runtime allowance. The input size can be "
        "huge. Optimize the time complexity of your code. Output the executable code "
        "only. Avoid unnecessary explanations or comment lines in your code."
    ),
    # Status 15: Runtime Error
    "runtime_error": (
        "Your code crashed at a Runtime Error:\n{error}\n"
        "Fix your code with above information. Output the executable code only. "
        "Avoid unnecessary explanations or comment lines in your code."
    ),
    # Status 20: Compile Error
    "compile_error": (
        "Your code crashed at a Compile Error:\n{error}\n"
        "Fix your code with above information. Output the executable code only. "
        "Avoid unnecessary explanations or comment lines in your code."
    ),
}

# Maximum length for a testcase before truncating the feedback prompt
MAX_TESTCASE_LEN = 20000


def problem_to_prompt(prob_info_dir, prob_name, lang):
    """Build the initial prompt message for a given problem and language.

    Reads the problem metadata, extracts the description, code snippet,
    and example test cases, then formats them into the prompt template.

    Returns:
        A list of message dicts (single user message).
    """
    prob_path = os.path.join(prob_info_dir, f"{prob_name}.json")
    if not lc.file_exists(prob_path):
        print(f"Error: Problem {prob_path} not found in pool!")
        return None

    prob_data = lc.load_prob_info_file(prob_path)
    description = lc.html_to_text(prob_data["content"])

    snippet = ""
    for item in prob_data["codeSnippets"]:
        if item["langSlug"] == lang:
            snippet = item["code"]

    testcases = "[" + " ".join(prob_data["exampleTestcaseList"]) + "]"

    prompt = INITIAL_PROMPT_TEMPLATE.format(
        lang=lang, description=description, snippet=snippet, testcases=testcases
    )
    return [{"role": "user", "content": prompt}]


# ======================================================================
# Submission and Feedback
# ======================================================================

def submit_and_check(leetcode, prob_name, prob_info_dir, lang, code, wait_seconds=60):
    """Submit code to Leetcode and retrieve the evaluation result.

    Args:
        leetcode: LeetcodeAPI instance.
        prob_name: Problem titleSlug.
        prob_info_dir: Directory containing problem metadata.
        lang: Programming language slug.
        code: Source code to submit.
        wait_seconds: Seconds to wait between submission and result check.

    Returns:
        JSON result dict with submission_id prepended, or None on failure.
    """
    prob_data = lc.load_prob_info_file(os.path.join(prob_info_dir, f"{prob_name}.json"))
    question_id = prob_data["questionId"]

    print(f"Submitting: {lang} | {prob_name}")
    submission_id = leetcode.submit_answer(prob_name, question_id, lang, code)
    if not submission_id:
        print("Submission FAILED")
        return None

    time.sleep(wait_seconds)

    print(f"Checking result: {submission_id}")
    result_data = leetcode.get_submission_info_detailed(submission_id)
    if not result_data:
        print("Result check FAILED")
        return None

    # Prepend submission_id to the result
    data = json.loads(result_data, object_pairs_hook=OrderedDict)
    new_data = OrderedDict([("submission_id", submission_id)])
    new_data.update(data)
    return new_data


def result_to_feedback(result_json):
    """Convert a submission result into a feedback prompt for the LLM.

    Args:
        result_json: OrderedDict from submit_and_check().

    Returns:
        tuple: (is_accepted: bool, feedback_messages: list or None)
               feedback_messages is a list of user-role message dicts.
    """
    try:
        status_code = result_json["data"]["submissionDetails"]["statusCode"]
    except (KeyError, TypeError) as e:
        print(f"Error reading status code: {e}")
        return False, None

    if status_code == 10:  # Accepted
        return True, None

    details = result_json["data"]["submissionDetails"]

    if status_code == 11:  # Wrong Answer
        last_testcase = details.get("lastTestcase", "")
        expected = details.get("expectedOutput", "")
        actual = details.get("codeOutput", "")
        if len(last_testcase) < MAX_TESTCASE_LEN:
            prompt = FEEDBACK_TEMPLATES["wrong_answer"].format(
                testcase=last_testcase, expected=expected, actual=actual
            )
        else:
            prompt = FEEDBACK_TEMPLATES["wrong_answer_long_testcase"].format(
                expected=expected, actual=actual
            )
        return False, [{"role": "user", "content": prompt}]

    elif status_code == 12:  # Memory Limit Exceeded
        return False, [{"role": "user", "content": FEEDBACK_TEMPLATES["memory_limit"]}]

    elif status_code == 14:  # Time Limit Exceeded
        return False, [{"role": "user", "content": FEEDBACK_TEMPLATES["time_limit"]}]

    elif status_code == 15:  # Runtime Error
        error = details.get("runtimeError", "Unknown error")
        prompt = FEEDBACK_TEMPLATES["runtime_error"].format(error=error)
        return False, [{"role": "user", "content": prompt}]

    elif status_code == 20:  # Compile Error
        error = details.get("compileError", "Unknown error")
        prompt = FEEDBACK_TEMPLATES["compile_error"].format(error=error)
        return False, [{"role": "user", "content": prompt}]

    else:
        print(f"WARNING: Non-standard status code: {status_code}")
        return False, None


def extract_code(answer_str, lang):
    """Extract clean source code from an LLM answer string."""
    return lc.extract_clean_src_code(
        lc.extract_code_block_by_lang(answer_str, lang), lang
    )


# ======================================================================
# Model Selection
# ======================================================================

def choose_model(model, model_name, messages, temperature=0.1, top_p=0.95):
    """Route to the correct model method based on model_name.

    Returns:
        tuple: (answer_message_list, prompt_tokens, completion_tokens)
    """
    dispatch = {
        "deepseek_r1": lambda: model.deepseek_r1_deepinfra_loop_stream(messages, temperature, top_p),
        "deepseek_v3": lambda: model.deepseek_v3_deepinfra_loop_stream(messages, temperature, top_p),
        "o4_mini": lambda: model.gpt_o4_mini_loop(messages),
        "gpt_41_mini": lambda: model.gpt_41_mini_loop(messages, temperature, top_p),
    }
    if model_name not in dispatch:
        print(f"ERROR: Unknown model '{model_name}'")
        return [], 0, 0
    return dispatch[model_name]()


# ======================================================================
# File I/O Helpers
# ======================================================================

def append_json_to_file(file_path, new_data):
    """Append a JSON object to a file (maintaining a JSON list)."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([], f)

    with open(file_path, "r") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    if isinstance(existing, list):
        existing.append(new_data)
    else:
        existing = [existing, new_data]

    with open(file_path, "w") as f:
        json.dump(existing, f, indent=4)


# ======================================================================
# Experiment Runner (with auto-resume support)
# ======================================================================

def _get_last_good_iter(results_file, max_iter):
    """Find the last successfully completed iteration from a results file.

    Returns:
        int: Last good iteration index (0 = start fresh, -1 = already complete/skip).
    """
    if not os.path.exists(results_file):
        return 0

    json_data = lc.file_to_json(results_file)
    for i in range(len(json_data)):
        if i >= max_iter * 2:
            break
        if "loop" in json_data[i]:
            if i + 1 >= len(json_data):
                # Unfinished iteration found
                return i // 2
    return -1  # All iterations are complete


def _load_existing_dialogue(prompts_file, resume_iter):
    """Load existing conversation history up to the resume point.

    Returns:
        tuple: (dialogue_messages, prompt_records)
    """
    json_data = lc.file_to_json(prompts_file)
    dialogue = []
    records = []
    for i in range(len(json_data)):
        if i > resume_iter * 3:
            break
        records.append(json_data[i])
        if "loop" not in json_data[i]:
            dialogue.append(json_data[i])
    return dialogue, records


def run_iterative_experiment(prob_info_dir, prob_name, lang, model_name, model,
                             output_dir, leetcode, max_iterations,
                             temperature=0.1, top_p=0.95):
    """Run the iterative feedback loop experiment for a single problem.

    Supports auto-resume: if results/prompts files already exist with
    partial data, the experiment continues from the last good iteration.

    Args:
        prob_info_dir: Directory with problem JSON metadata files.
        prob_name: Problem titleSlug.
        lang: Programming language slug.
        model_name: Model identifier string.
        model: Model instance.
        output_dir: Base output directory (model/lang files go underneath).
        leetcode: LeetcodeAPI instance.
        max_iterations: Maximum feedback loop rounds (default 10).
        temperature: LLM sampling temperature.
        top_p: LLM nucleus sampling top_p.

    Returns:
        True if the problem was solved (Accepted), False otherwise.
    """
    base = os.path.join(output_dir, model_name, lang, prob_name)
    prompt_file = f"{base}_prompts.json"
    results_file = f"{base}_results.json"

    # Check for resume point
    resume_iter = _get_last_good_iter(results_file, max_iterations)
    if resume_iter < 0:
        print(f"[INFO] Already complete, skipping: {prob_name}")
        return True

    messages = []

    # If resuming from a mid-point, reload existing state
    if resume_iter > 0:
        dialogue, records = _load_existing_dialogue(prompt_file, resume_iter)
        result_data = lc.file_to_json(results_file)
        # Remove trailing loop markers
        if records and "loop" in records[-1]:
            records.pop()
        if result_data and "loop" in result_data[-1]:
            result_data.pop()

        # Clear and restore files
        for path in [prompt_file, results_file]:
            with open(path, "w") as f:
                pass
        for r in records:
            append_json_to_file(prompt_file, r)
        for r in result_data:
            append_json_to_file(results_file, r)

        # Rebuild messages from dialogue for the LLM context
        messages = list(dialogue)
        print(f"[INFO] Resuming from iteration {resume_iter + 1}")

    # Clear files if starting fresh
    if resume_iter == 0:
        for path in [prompt_file, results_file]:
            if os.path.exists(path):
                os.remove(path)

    # Run the feedback loop
    for i in range(resume_iter, max_iterations):
        print(f"######\nLoop #{i + 1} / {max_iterations}\n######")
        append_json_to_file(prompt_file, {"loop": i + 1})
        append_json_to_file(results_file, {"loop": i + 1})

        # Build initial prompt on first iteration
        if not messages:
            initial = problem_to_prompt(prob_info_dir, prob_name, lang)
            if not initial:
                return False
            messages.extend(initial)

        print(f"[INFO] Prompt (last msg): {messages[-1]['content'][:200]}...")

        # Get LLM response
        answer_msg, pt, ct = choose_model(model, model_name, messages, temperature, top_p)
        if not answer_msg:
            print(f"[ERROR] Empty response for {model_name} | {lang} | {prob_name}")
            return False

        answer_str = answer_msg[0]["content"]
        append_json_to_file(prompt_file, messages[-1])
        append_json_to_file(prompt_file, answer_msg[0])
        messages.extend(answer_msg)

        if not answer_str:
            print(f"[ERROR] Empty answer text for {model_name} | {lang} | {prob_name}")
            return False

        # Submit and check
        code = extract_code(answer_str, lang)
        result_json = submit_and_check(leetcode, prob_name, prob_info_dir, lang, code)
        if not result_json:
            print(f"[ERROR] Submission failed for {model_name} | {lang} | {prob_name}")
            return False

        append_json_to_file(results_file, result_json)

        # Generate feedback
        accepted, feedback = result_to_feedback(result_json)
        if accepted:
            print(f"INFO: {prob_name} Accepted on iteration {i + 1}!")
            return True
        if feedback:
            messages.extend(feedback)

    return False


# ======================================================================
# Batch Runner
# ======================================================================

def run_batch(model_list, lang_list, prob_list,
              prob_info_dir, output_dir, model, max_iter,
              temperature=0.1, top_p=0.95):
    """Run iterative experiments for all model/language/problem combinations."""
    leetcode = lc.LeetcodeAPI()
    for model_name in model_list:
        for lang in lang_list:
            for i, prob_name in enumerate(prob_list):
                print(f"\n###### {i + 1} / {len(prob_list)} #######")
                print(f"{lang} | {model_name} | {prob_name}")
                success = run_iterative_experiment(
                    prob_info_dir, prob_name, lang, model_name, model,
                    output_dir, leetcode, max_iter, temperature, top_p
                )
                if not success:
                    print(f"[WARNING] NOT accepted: {lang} | {model_name} | {prob_name}")


# ======================================================================
# CLI Entry Point
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run iterative feedback loop experiments on Leetcode problems."
    )
    parser.add_argument("--model", nargs="+", default=["deepseek_v3"],
                        help="Model names to test (e.g., deepseek_r1 deepseek_v3 o4_mini gpt_41_mini)")
    parser.add_argument("--lang", nargs="+", default=["python3"],
                        help="Programming languages (e.g., python3 java cpp)")
    parser.add_argument("--prob_list", required=True,
                        help="Path to problem list file (one titleSlug per line)")
    parser.add_argument("--prob_info_dir", default="data/problems/",
                        help="Directory containing problem metadata JSON files")
    parser.add_argument("--output_dir", default="data/iter_exp/",
                        help="Base output directory for results")
    parser.add_argument("--max_iter", type=int, default=10,
                        help="Maximum feedback loop iterations per problem")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="LLM nucleus sampling top_p")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prob_list = lc.file_to_list(args.prob_list)
    if not prob_list:
        print(f"ERROR: Could not load problem list from {args.prob_list}")
        exit(1)

    model = lc.Model()

    run_batch(
        model_list=args.model,
        lang_list=args.lang,
        prob_list=prob_list,
        prob_info_dir=args.prob_info_dir,
        output_dir=args.output_dir,
        model=model,
        max_iter=args.max_iter,
        temperature=args.temperature,
        top_p=args.top_p,
    )
