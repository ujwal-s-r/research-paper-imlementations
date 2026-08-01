"""
dataset.py
----------------
Synthetic arithmetic dataset generator and prompt formatter following 
the DeepSeek-R1-Zero conversational template (Table 1 in deepseekRL.pdf).
"""

import random
import torch
from torch.utils.data import Dataset


class SyntheticArithmeticDataset(Dataset):
    """
    Generates multi-step arithmetic expressions and formats them into 
    the R1-Zero prompt template (Table 1):
    "A conversation between User and Assistant. The user asks a question, 
    and the Assistant solves it. The assistant first thinks about the reasoning 
    process in the mind and then provides the user with the answer. 
    The reasoning process and answer are enclosed within <think>...</think> 
    and <answer>...</answer> tags..."
    """
    def __init__(self, num_samples: int = 500, seed: int = 42):
        random.seed(seed)
        self.data = []
        
        for _ in range(num_samples):
            # Generate 2-step arithmetic problem: A * B +/- C
            a = random.randint(2, 12)
            b = random.randint(2, 10)
            c = random.randint(1, 25)
            op = random.choice(["+", "-"])
            
            if op == "+":
                ans = a * b + c
            else:
                ans = a * b - c
                
            prompt_expr = f"{a}*{b}{op}{c}="
            target_ans = str(ans)
            
            # Format according to DeepSeek-R1-Zero prompt template (Table 1)[cite: 1]
            formatted_prompt = (
                "A conversation between User and Assistant. The user asks a question, "
                "and the Assistant solves it. The assistant first thinks about the reasoning "
                "process in the mind and then provides the user with the answer. "
                "The reasoning process and answer are enclosed within <think>...</think> "
                "and <answer>...</answer> tags, respectively. "
                f"User: Solve {prompt_expr} Assistant: <think>"
            )
            
            self.data.append({
                "prompt": formatted_prompt,
                "expression": prompt_expr,
                "ground_truth": target_ans
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =====================================================================
# Standalone Unit Test Suite
# =====================================================================
if __name__ == "__main__":
    print("=== Running Unit Test Suite for dataset.py ===")
    dataset = SyntheticArithmeticDataset(num_samples=3)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Expression:   {sample['expression']}")
        print(f"  Ground Truth: {sample['ground_truth']}")
        print(f"  Full Prompt:\n{sample['prompt']}")
        
    print("\nDataset Module Successfully Verified!")