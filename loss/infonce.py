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