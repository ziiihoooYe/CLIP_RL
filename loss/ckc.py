import torch
import torch.nn as nn
import torch.nn.functional as F

def ckc_loss(
    old_image_embeds: torch.Tensor,
    old_text_embeds: torch.Tensor,
    new_image_embeds: torch.Tensor,
    new_text_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Contrastive Knowledge Consolidation (CKC) Loss
    old_image_embeds, old_text_embeds: Frozen Model Feature (batch, dim)
    new_image_embeds, new_text_embeds: New Model Feature (batch, dim)
    projector: Projection Head
    temperature: temperature for contrastive loss
    """
    # 1) Normalize old_image, old_text embeddings
    with torch.no_grad():
        old_image_embeds = F.normalize(old_image_embeds, dim=1)
        old_text_embeds = F.normalize(old_text_embeds, dim=1)
    # 2) Normalize new_image, new_text embeddings
    new_image_embeds = F.normalize(new_image_embeds, dim=1)
    new_text_embeds = F.normalize(new_text_embeds, dim=1)

    # 3) concatenate old_image, old_text
    old_cat = torch.cat([old_image_embeds, old_text_embeds], dim=0)  # shape = [2B, dim]
    new_cat = torch.cat([new_image_embeds, new_text_embeds], dim=0) # shape = [2B, dim]

    # 4) Similarity matrix
    logits = (new_cat @ old_cat.t()) / temperature  # shape = [2B, 2B]

    labels = torch.arange(new_cat.size(0), device=new_cat.device)

    # 5) Calculate loss
    loss_i2j = F.cross_entropy(logits, labels)
    loss_j2i = F.cross_entropy(logits.t(), labels)
    loss = (loss_i2j + loss_j2i) / 2.0

    return loss

