# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F


def mha_forward(Q, K, V, mask=None, dropout_p=0.0, training=False):
    """
    Q: [B, H, N, D]
    K: [B, H, M, D]
    V: [B, H, M, D]
    mask: [B, H, N, M] or None
    返回: output, attn_weights, dropout_mask
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5  # [B, H, N, M]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    dropout_mask = None
    if dropout_p > 0 and training:
        attn_weights, dropout_mask = (
            F.dropout(attn_weights, p=dropout_p, training=training, inplace=False),
            attn_weights > 0,
        )
    output = torch.matmul(attn_weights, V)  # [B, H, N, D]
    return output, attn_weights, dropout_mask


def mha_backward(
    Q, K, V, grad_O, attn_weights, mask=None, dropout_p=0.0, dropout_mask=None
):
    """
    Q, K, V: [B, H, N/M, D]
    grad_O: [B, H, N, D]
    attn_weights: [B, H, N, M]
    mask: [B, H, N, M] or None
    dropout_p: float
    dropout_mask: [B, H, N, M] or None
    """
    d_k = Q.size(-1)
    # grad w.r.t. V
    grad_V = torch.matmul(attn_weights.transpose(-2, -1), grad_O)  # [B, H, M, D]
    # grad w.r.t. attn_weights
    grad_attn = torch.matmul(grad_O, V.transpose(-2, -1))  # [B, H, N, M]
    # dropout backward
    if dropout_p > 0 and dropout_mask is not None:
        grad_attn = grad_attn * dropout_mask / (1 - dropout_p)
    # softmax backward
    # Let y = softmax(x), dy/dx = diag(y) - y y^T
    # grad_x = y * (grad_y - sum(y * grad_y))
    grad_scores = grad_attn * attn_weights
    grad_scores = grad_scores - attn_weights * grad_scores.sum(dim=-1, keepdim=True)
    # mask backward
    if mask is not None:
        grad_scores = grad_scores.masked_fill(mask == 0, 0.0)
    # grad w.r.t. Q
    grad_Q = torch.matmul(grad_scores, K) / d_k**0.5  # [B, H, N, D]
    # grad w.r.t. K
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q) / d_k**0.5  # [B, H, M, D]
    return grad_Q, grad_K, grad_V


# 用法示例
if __name__ == "__main__":
    B, H, N, M, D = 2, 4, 8, 8, 16
    Q = torch.randn(B, H, N, D, requires_grad=True)
    K = torch.randn(B, H, M, D, requires_grad=True)
    V = torch.randn(B, H, M, D, requires_grad=True)
    mask = torch.ones(B, H, N, M)
    grad_O = torch.randn(B, H, N, D)
    output, attn_weights, dropout_mask = mha_forward(
        Q, K, V, mask=mask, dropout_p=0.1, training=True
    )
    grad_Q, grad_K, grad_V = mha_backward(
        Q,
        K,
        V,
        grad_O,
        attn_weights,
        mask=mask,
        dropout_p=0.1,
        dropout_mask=dropout_mask,
    )
    print(output, grad_Q, grad_K, grad_V)
