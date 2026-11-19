from __future__ import annotations
import torch
from torch import nn

class EWC:
    """Diagonal Fisher Elastic Weight Consolidation."""
    def __init__(self, model: nn.Module, strength: float = 100.0):
        self.strength = strength
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    def estimate(self, model, loader, device, allowed_classes=None):
        model.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        count = 0
        criterion = nn.CrossEntropyLoss()
        for videos, labels, _ in loader:
            labels = labels.to(device)
            model.zero_grad(set_to_none=True)
            loss = criterion(model([v.to(device) for v in videos], allowed_classes), labels)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().pow(2)
            count += 1
        if count:
            for name in fisher:
                fisher[name] /= count
        self.fisher = fisher
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def penalty(self, model):
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                total += (self.fisher[name] * (param - self.params[name]).pow(2)).sum()
        return self.strength * total
