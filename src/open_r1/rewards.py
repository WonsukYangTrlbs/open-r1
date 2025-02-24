"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict, Literal

import evaluate
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.if_functions import IF_FUNCTIONS_MAP
from .utils.ko_if_functions import KO_IF_FUNCTIONS_MAP


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()

CODE_EVAL = evaluate.load("code_eval")


def accuracy_reward(completions, **kwargs):
    ground_truth = kwargs.get("ground_truth")
    dataset = kwargs.get("dataset")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, dataset in zip(contents, ground_truth, dataset):
        if dataset == "ifeval":
            # sol is actually constraint
            reward = verify_ifeval_sample(content, sol)
        elif dataset == "ko-gsm8k":
            gold_parsed = sol.strip()
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = extract_gsm_response(content, pattern="####")
                
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
                print("[ko-GSM8k] Failed to parse gold solution: ", sol)
        elif dataset == "koifeval":
            reward = verify_koifeval_sample(content, sol)
        elif dataset == "MATH":
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
                print("[MATH] Failed to parse gold solution: ", sol)
        elif dataset == "gsm8k":
            gold_parsed = sol.strip()
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = extract_gsm_response(content, pattern="So the answer is")
                
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
                print("[GSM8k] Failed to parse gold solution: ", sol)
        elif dataset == "mbpp":
            reward = verify_mbpp_sample(content, sol)
        elif dataset == "kmmlu":
            extracted_answer = extract_kmmlu_response(content)
            reward = float(extracted_answer == sol)
        elif dataset == "mmlu":
            extracted_answer = extract_mmlu_response(content)
            reward = float(extracted_answer == sol)
        rewards.append(reward)
    return rewards


def parse_python_code(content: str) -> str:
    return content.split("""```python""")[1].split("""```""")[0].strip()


def check_failed_with_code_issue(text):
    return int("failed:" in text and len(text) > len("failed: "))


def verify_mbpp_sample(content, ground_truth, mode: Literal["strict", "loose"] = "strict"):
    content = parse_python_code(content)
    test_cases = json.loads(ground_truth)
    test_results = CODE_EVAL.compute(references=[test_cases], predictions=[[content] for _ in test_cases], k=[1])
    if mode == "strict":
        score = float(test_results[0]["pass@1"])
    else:  # mode == "loose"
        did_code_ran = [1.0 - check_failed_with_code_issue(result[0][1]["result"]) for result in test_results[1]]
        score = sum(did_code_ran) / len(did_code_ran)
    return score

THINK_PATTERN = r"<think>\n.*?\n</think>"

def remove_think(content):
    cleaned_content = re.sub(THINK_PATTERN, "", content, flags=re.DOTALL).strip()
    return cleaned_content

# NOTE: expected "The answer is (A)." => A
def extract_mmlu_response(content):
    answer = content.split()[-1].strip("().")
    return answer


# NOTE: expected "정답은 A입니다." => A
def extract_kmmlu_response(content):
    try:
        answer = content.split()[1][0]
    except IndexError:
        answer = ""
    return answer


def verify_koifeval_sample(content, constraint):
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    func_names = constraint.pop("func_name")
    func_constraints = constraint.pop("constraints")
    verified = True
    for func_name in func_names:
        func = KO_IF_FUNCTIONS_MAP[func_name]
        func_constraint = func_constraints[func_name]
        if func_constraint is None or len(func_constraint) == 0:
            verified = verified and func(content)
        else:
            non_none_args = {k: v for k, v in func_constraint.items() if v is not None}
            verified = verified and func(content, **non_none_args)
    return float(verified)

def verify_ifeval_sample(content, constraint):
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    func_name = constraint.pop("func_name")
    func = IF_FUNCTIONS_MAP[func_name]
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    if len(constraint) == 0:
        return 1.0 if func(content) else 0.0
    return 1.0 if func(content, **non_none_args) else 0.0

def instruction_following_reward(completions, **kwargs):
    constraints = kwargs.get("ground_truth")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, constraint in zip(contents, constraints):
        reward = verify_ifeval_sample(content, constraint)
        rewards.append(reward)
    return rewards

def extract_response(content):
    searched = re.search(r"So the answer is (.+)", content)
    return searched.group(1).strip() if searched else ""

def extract_gsm_response(text: str, pattern: str = "So the answer is") -> str | None:
    if pattern not in text:
        return None
    answer = text.split(pattern)[1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return answer

def gsm_accuracy_reward(completions, **kwargs):
    ground_truth = kwargs.get("ground_truth")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, ground_truth):
        gold_parsed = sol.strip()
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = extract_gsm_response(content)
            
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    return rewards


def math_accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    ground_truth = kwargs.get("ground_truth")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, ground_truth):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    dataset = kwargs.get("dataset")
    reverse = (dataset == "ifeval")
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    if reverse:
        return [0.0 if match else 1.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                    anchor_config=AnchorConfig(
                        text="So the answer is",
                        after_text=True,  # Extract the number that comes after this phrase
                        max_distance=10,  # Adjust based on expected distance from the phrase
                    ),
                )
            ],
            extraction_mode="first_match",
        )
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards

def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>.*?</think>\s*<answer>.*?```{language}\n.*?```.*?</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward
