"""
reward_engine.py
----------------
Deterministic, rule-based reward evaluator for DeepSeek-R1-Zero arithmetic tasks.
Evaluates both structural formatting (R_fmt) and mathematical correctness (R_acc).
"""

import re
import sympy
from typing import Dict, Any, Union


class R1ZeroRewardEngine:
    """
    Rule-based reward engine implementing R_rule = R_acc + R_fmt.
    Includes soft-format credit to prevent zero-variance gradient deadlocks 
    during early training of small models.
    """
    def __init__(self, format_weight: float = 0.1):
        self.format_weight = format_weight
        
        # Regex for strict end-to-end tag validation
        self.strict_format_regex = re.compile(
            r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$", 
            re.DOTALL
        )

    def evaluate_math_correctness(self, pred_str: str, target_str: str) -> bool:
        """
        Uses SymPy to check mathematical equivalence between prediction and target.
        Falls back to normalized string comparison if parsing fails.
        """
        clean_pred = pred_str.strip().replace(",", "")
        clean_target = target_str.strip().replace(",", "")

        if not clean_pred:
            return False

        try:
            pred_expr = sympy.parse_expr(clean_pred)
            target_expr = sympy.parse_expr(clean_target)
            return sympy.simplify(pred_expr - target_expr) == 0
        except Exception:
            # Fallback to direct string matching
            return clean_pred == clean_target

    def extract_answer_content(self, text: str) -> str:
        """
        Extracts answer content enclosed within <answer>...</answer> tags.
        If tags are missing, attempts to extract the last non-empty line.
        """
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback extraction if tags are partially formed
        if "<answer>" in text:
            return text.split("<answer>")[-1].replace("</answer>", "").strip()
            
        return text.strip()

    def compute_format_reward(self, text: str) -> float:
        """
        Calculates R_fmt.
        Strict match: +0.1
        Partial matches (<think> or <answer> tags present): +0.02 to +0.05
        """
        if self.strict_format_regex.match(text):
            return self.format_weight
        
        # Soft format credit for small models in early RL steps
        partial_score = 0.0
        if "<think>" in text:
            partial_score += 0.025
        if "</think>" in text:
            partial_score += 0.025
        if "<answer>" in text:
            partial_score += 0.025
        if "</answer>" in text:
            partial_score += 0.025
            
        return min(partial_score, self.format_weight)

    def compute_reward(self, completion_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Main evaluation entry point.
        Returns total_reward, r_acc, r_fmt, and metadata for logging.
        """
        r_fmt = self.compute_format_reward(completion_text)
        
        # Extract answer and check accuracy
        extracted_ans = self.extract_answer_content(completion_text)
        is_correct = self.evaluate_math_correctness(extracted_ans, ground_truth)
        
        # Accuracy reward is only awarded if answer is mathematically correct
        r_acc = 1.0 if is_correct else 0.0
        
        total_reward = r_acc + r_fmt
        
        return {
            "total_reward": total_reward,
            "r_acc": r_acc,
            "r_fmt": r_fmt,
            "is_correct": is_correct,
            "extracted_answer": extracted_ans
        }


# =====================================================================
# Standalone Unit Test Suite
# =====================================================================
if __name__ == "__main__":
    print("=== Running Unit Test Suite for reward_engine.py ===")
    verifier = R1ZeroRewardEngine(format_weight=0.1)

    test_cases = [
        {
            "name": "Perfect Output (Correct + Strict Format)",
            "input": "<think> 14 * 3 = 42, 42 - 12 = 30 </think> <answer> 30 </answer>",
            "truth": "30",
            "expected_reward": 1.1,
            "expected_correct": True
        },
        {
            "name": "SymPy Equivalence (Fraction vs Decimal)",
            "input": "<think> 1 / 2 </think> <answer> 0.5 </answer>",
            "truth": "1/2",
            "expected_reward": 1.1,
            "expected_correct": True
        },
        {
            "name": "Wrong Answer + Strict Format",
            "input": "<think> 14 * 3 = 42 </think> <answer> 99 </answer>",
            "truth": "30",
            "expected_reward": 0.1,
            "expected_correct": False
        },
        {
            "name": "Partial Tags (Soft Format Credit)",
            "input": "<think> 14 * 3 = 42, Answer is 30",
            "truth": "30",
            "expected_reward": 0.025,
            "expected_correct": False # No <answer> tag
        },
        {
            "name": "No Tags At All",
            "input": "The result is 30",
            "truth": "30",
            "expected_reward": 0.0,
            "expected_correct": False
        }
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        res = verifier.compute_reward(tc["input"], tc["truth"])
        passed = (abs(res["total_reward"] - tc["expected_reward"]) < 1e-5) and (res["is_correct"] == tc["expected_correct"])
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"Test {idx} [{tc['name']}]: {status}")
        print(f"  Result -> Reward: {res['total_reward']:.3f} | Correct: {res['is_correct']} | Extracted: '{res['extracted_answer']}'")

    print("\nUnit Test Suite Summary:", "ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED!")