import math
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from framework.experiments import Experiment
import torch.optim as optim
from torch.utils.data import DataLoader
from loss.infonce import infonce_loss
from utils.utils import get_gpu_device
from tqdm.contrib.logging import logging_redirect_tqdm

class ContrastiveLearningExperiment(Experiment):
    def __init__(self, config):
        super(ContrastiveLearningExperiment, self).__init__(config)
    
    def run(self, model, dataset, preprocessor_list, logger):
        ContrastiveLearning(model, dataset, preprocessor_list, self.config, logger)
        return model


def ContrastiveLearning(model, dataset, preprocessor_list, config, logger):
    """
    model: model with encode_image and encode_text method
    dataset: dataset with __getitem__ and __len__ method
    preprocessor_list: list of preprocessors for data augmentation
    config: configuration dictionary
    """
    ### --------------------
    ### Instantiation
    ### --------------------
    device = get_gpu_device(config.get("gpu", None))
    scaler = torch.amp.GradScaler()

    # hyperparameters
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 1e-4)
    num_workers = config.get("num_workers", 4)
    temperature_init = config.get("temperature_init", 0.07)
    temperature_learnable = config.get("temperature_learnable", False)
    amp_enabled = config.get("amp_enabled", False)
    max_iter = config.get("max_iter", None)

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
        logit_scale = None
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )
    
    logger.info(f"Contrastive Learning Training on dataset {dataset.name} Started----")
    with logging_redirect_tqdm(loggers=[logger]):
        global_iter = 0
        with tqdm(total=max_iter, desc=f"Training Progress") as pbar:
            for epoch in range(epochs):
                total_loss = 0.0
                for i, (images, captions) in enumerate(train_loader):
                    if max_iter is not None and global_iter >= max_iter:
                        break

                    ctx = None
                    if dataset.config.get("num_captions", 1) > 1:
                        captions = [caption[random.randint(0, len(caption)-1)] for caption in captions]

                    for preprocessor in preprocessor_list:
                        images, captions, ctx = preprocessor.preprocess(images, captions, ctx)

                    with torch.amp.autocast('cuda' if 'cuda' in device else 'cpu', enabled=amp_enabled):
                        image_embeds = model.encode_image(images)
                        text_embeds  = model.encode_text(captions)

                        if temperature_learnable:
                            temperature = (1.0 / logit_scale.exp()).clamp(min=0.01, max=1.0)
                        else:
                            temperature = temperature_init
                        loss = infonce_loss(image_embeds, text_embeds, temperature=temperature)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if temperature_learnable:
                        all_params = list(model.parameters()) + [logit_scale]
                    else:
                        all_params = list(model.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    if temperature_learnable:
                        with torch.no_grad():
                            logit_scale.clamp_(0.0, math.log(100.0))

                    total_loss += loss.item()
                    global_iter += 1
                    pbar.set_postfix(loss=loss.item(), epoch=epoch+1)
                    pbar.update(1)

    logger.info("Contrastive Learning Training Finished----")
    
    return model
