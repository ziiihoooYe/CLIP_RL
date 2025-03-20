import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loss.infonce import infonce_loss
from utils.utils import get_gpu_device

def ContrastiveLearning(model, dataset, preprocessor_list, config):
    """
    model: model with encode_image and encode_text method
    dataset: dataset with __getitem__ and __len__ method
    preprocessor_list: list of preprocessors for data augmentation
    config: configuration dictionary
    """
    ### --------------------
    ### Instantiation
    ### --------------------
    train_config = config["train"]
    device = get_gpu_device(config['default'].get("gpu", None))

    # hyperparameters
    epochs = train_config.get("epochs", 10)
    batch_size = train_config.get("batch_size", 32)
    lr = train_config.get("learning_rate", 1e-4)
    num_workers = train_config.get("num_workers", 4)
    temperature_init = train_config.get("temperature_init", 0.07)
    temperature_learnable = train_config.get("temperature_learnable", False)

    # prepare model
    model.to(device)
    model.train()
    

    # prepare optimizer
    if temperature_learnable:
        logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)), requires_grad=True)
        optimizer = optim.Adam([
            {"params": model.parameters()},
            {"params": [logit_scale]}
        ], lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (images, captions) in enumerate(train_loader):
            ctx = None

            for preprocessor in preprocessor_list: 
                images, captions, ctx = preprocessor.preprocess(images, captions, ctx)

            # images and captions are lists
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(captions)
            
            if temperature_learnable:
                temperature = 1.0 / logit_scale.exp()
            else:
                temperature = temperature_init

            loss = infonce_loss(image_embeds, text_embeds, temperature=temperature)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    print("Finish Contrastive Learning Training!")