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
    # 1) concatenate old_image, old_text
    old_cat = torch.cat([old_image_embeds, old_text_embeds], dim=0)  # shape = [2B, dim]
    new_cat = torch.cat([new_image_embeds, new_text_embeds], dim=0) # shape = [2B, dim]
    
    # 2) Normalize
    old_cat = F.normalize(old_cat, dim=-1)
    new_cat = F.normalize(new_cat, dim=-1)

    # 3) Similarity matrix
    logits = (new_cat @ old_cat.t()) / temperature  # shape = [2B, 2B]

    labels = torch.arange(new_cat.size(0), device=new_cat.device)

    # 4) Calculate loss
    loss_i2j = F.cross_entropy(logits, labels)
    loss_j2i = F.cross_entropy(logits.t(), labels)
    loss = (loss_i2j + loss_j2i) / 2.0

    return loss

def ckc_loss_test(
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
    from loss.infonce import infonce_loss
    img_loss = infonce_loss(new_image_embeds, old_image_embeds, temperature=temperature)
    txt_loss = infonce_loss(new_text_embeds, old_text_embeds, temperature=temperature)    

    return img_loss+txt_loss
