from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class VideoClassifier(nn.Module):
    """ResNet-18 frame encoder; video embedding is the mean of sampled frame embeddings."""
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, videos: list[torch.Tensor], allowed_classes: list[int] | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        embeddings = []
        for video in videos:
            feats = self.encoder(video.to(device))
            embeddings.append(feats.mean(dim=0))
        logits = self.classifier(torch.stack(embeddings))
        # In class-incremental evaluation, future/unseen classes must not compete.
        if allowed_classes is not None:
            mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
            mask[allowed_classes] = True
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)
        return logits
