"""
grpo_loss.py
------------
Pure PyTorch implementation of Group Relative Policy Optimization (GRPO) loss functions,
Group Advantage normalization, and Schulman's unbiased token-level KL divergence.
(Equations 1, 2, and 3 in deepseekRL.pdf).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def compute_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes intra-group relative advantages (Equation 3).
    rewards shape: [group_size]
    returns shape: [group_size]
    """
    mean_r = torch.mean(rewards)
    std_r = torch.std(rewards, unbiased=False)
    
    # Handle zero variance edge case (e.g., all responses get 0 reward)
    if std_r < eps:
        return torch.zeros_like(rewards)
        
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages


def compute_unbiased_kl(log_p_curr: torch.Tensor, log_p_ref: torch.Tensor) -> torch.Tensor:
    """
    Schulman's unbiased token-level KL estimator (Equation 2):
    KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    
    log_p_curr shape: [batch_size, seq_len]
    log_p_ref shape:  [batch_size, seq_len]
    returns shape:     [batch_size, seq_len]
    """
    log_ratio = log_p_ref - log_p_curr
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return kl


def compute_grpo_loss(
    log_probs_curr: torch.Tensor,  # [G, T] - current policy log-probabilities
    log_probs_old: torch.Tensor,   # [G, T] - rollout policy log-probabilities (detached)
    log_probs_ref: torch.Tensor,   # [G, T] - reference policy log-probabilities (detached)
    advantages: torch.Tensor,      # [G]    - scalar advantages
    completion_mask: torch.Tensor, # [G, T] - boolean/float mask indicating non-padding completion tokens
    clip_eps: float = 0.2,         # PPO/GRPO clipping threshold
    beta_kl: float = 0.001         # KL penalty coefficient
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculates masked GRPO surrogate loss with unbiased token KL regularization (Equation 1).
    """
    group_size, seq_len = log_probs_curr.shape
    
    # Expand advantages across sequence dimension: [G] -> [G, 1]
    adv_expanded = advantages.unsqueeze(-1)
    
    # 1. Probability ratio r_t(theta) = exp(log_p_curr - log_p_old)
    log_ratios = log_probs_curr - log_probs_old
    ratios = torch.exp(log_ratios)
    
    # 2. PPO-style clipped surrogate objective
    surr1 = ratios * adv_expanded
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expanded
    policy_surrogate = torch.min(surr1, surr2) # [G, T]
    
    # 3. Unbiased token-level KL penalty
    kl_penalty = compute_unbiased_kl(log_probs_curr, log_probs_ref) # [G, T]
    
    # 4. Net per-token objective (we want to maximize policy_surrogate - beta * KL)
    # Loss is the negative of the objective
    token_loss = -(policy_surrogate - beta_kl * kl_penalty) # [G, T]
    
    # Apply completion mask to ignore padding tokens
    masked_loss = token_loss * completion_mask
    num_active_tokens = completion_mask.sum().clamp(min=1.0)
    
    total_loss = masked_loss.sum() / num_active_tokens
    
    # Metrics for monitoring
    with torch.no_grad():
        mean_kl = (kl_penalty * completion_mask).sum() / num_active_tokens
        mean_ratio = (ratios * completion_mask).sum() / num_active_tokens
        clip_fraction = (((ratios - 1.0).abs() > clip_eps).float() * completion_mask).sum() / num_active_tokens

    metrics = {
        "grpo_loss": total_loss.item(),
        "mean_kl": mean_kl.item(),
        "mean_ratio": mean_ratio.item(),
        "clip_fraction": clip_fraction.item()
    }

    return total_loss, metrics


# =====================================================================
# Standalone Unit Test Suite
# =====================================================================
if __name__ == "__main__":
    print("=== Running Unit Test Suite for grpo_loss.py ===")
    torch.manual_seed(42)

    G = 4  # Group size
    T = 6  # Sequence length

    # Test 1: Advantage Normalization
    rewards = torch.tensor([0.0, 1.1, 0.0, 0.1])
    advs = compute_group_advantages(rewards)
    print(f"Test 1 [Advantage Normalization]:")
    print(f"  Rewards:    {rewards.tolist()}")
    print(f"  Advantages: {[round(a, 3) for a in advs.tolist()]}")
    assert advs.shape == (G,), "Advantage shape mismatch"

    # Test 2: Advantage Zero-Variance Edge Case
    zero_rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
    zero_advs = compute_group_advantages(zero_rewards)
    print(f"Test 2 [Zero Variance Check]:")
    print(f"  Zero Advantages: {zero_advs.tolist()}")
    assert (zero_advs == 0.0).all(), "Zero variance advantages failed"

    # Test 3: Loss Backpropagation and Masking
    log_p_curr = torch.randn(G, T, requires_grad=True)
    log_p_old = log_p_curr.detach() - 0.02
    log_p_ref = log_p_curr.detach() - 0.05
    mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                          [1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 1, 0]], dtype=torch.float32)

    loss, metrics = compute_grpo_loss(
        log_probs_curr=log_p_curr,
        log_probs_old=log_p_old,
        log_probs_ref=log_p_ref,
        advantages=advs,
        completion_mask=mask
    )

    loss.backward()

    print(f"Test 3 [Loss Backpropagation]:")
    print(f"  Calculated GRPO Loss: {metrics['grpo_loss']:.6f}")
    print(f"  Mean KL Divergence:  {metrics['mean_kl']:.6f}")
    print(f"  Clipping Fraction:   {metrics['clip_fraction']:.2%}")
    print(f"  Gradients Computed:  {log_p_curr.grad is not None and not torch.isnan(log_p_curr.grad).any()}")
    
    assert log_p_curr.grad is not None, "Backward pass failed to compute gradients"
    print("\nGRPO Loss Module Successfully Verified!")