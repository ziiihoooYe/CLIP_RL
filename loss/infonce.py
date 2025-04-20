import torch
import torch.nn.functional as F

def infonce_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: [batch_size, embed_dim]
    text_embeds:  [batch_size, embed_dim]
    temperature:  float, default=0.07
    """
    # normalize embeddings
    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds  = F.normalize(text_embeds, p=2, dim=1)

    # similarity matrix
    logits = image_embeds @ text_embeds.t() / temperature

    # construct labels
    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)

    # calculate loss
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    loss = (loss_i + loss_t) / 2.0
    return loss


def infonce_multimodal_loss(image_embeds, text_embeds, temperature: float = 0.07):
    # L2‑normalize
    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds  = F.normalize(text_embeds,  p=2, dim=1)
 
    batch_size = image_embeds.size(0)
    assert text_embeds.size(0) == batch_size, "image/text batch sizes must match"
 
    # 1. pool embeddings: (2B, D)
    pooled = torch.cat([image_embeds, text_embeds], dim=0)
 
    # 2. similarity matrix: (2B, 2B)
    logits = pooled @ pooled.T / temperature
 
    # 3. mask self‑similarity
    logits.fill_diagonal_(float("-inf"))
 
    # 4. construct labels: imageᵢ → textᵢ (idx = B+ᵢ); textⱼ → imageⱼ (idx = j−B)
    labels = torch.arange(2 * batch_size, device=pooled.device)
    labels = torch.where(labels < batch_size,
                         labels + batch_size,     # image rows
                         labels - batch_size)     # text rows
 
    # 5. cross‑entropy over rows
    loss = F.cross_entropy(logits, labels)
    return loss