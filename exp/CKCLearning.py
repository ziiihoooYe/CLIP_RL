import math
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from framework.experiments import Experiment
from utils.utils import gpu_prep, get_main_device
import torch.optim as optim
from torch.utils.data import DataLoader
from loss.infonce import infonce_loss
from loss.ckc import ckc_loss, ckc_loss_test
from tqdm.contrib.logging import logging_redirect_tqdm
import copy

class CKCLearningExperiment(Experiment):
    def __init__(self, config):
        super(CKCLearningExperiment, self).__init__(config)
    
    def run(self, model, dataset, preprocessor_list, logger, device):
        model = CKCLearning(model, dataset, preprocessor_list, self.config, logger, device)
        return model

class MLPProjector(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim_out, dim_out)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class FrozenWithProjector(nn.Module):
    def __init__(self, base_model, img_projector, txt_projector):
        super().__init__()
        # base_model
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.img_projector = img_projector
        self.txt_projector = txt_projector
        

    def to(self, device):
        """
        Move the model to the specified device.
        :param device: The device to move the model to (e.g., 'cuda', 'cpu').
        :return: self
        """
        self.base_model = self.base_model.to(device)
        self.img_projector = self.img_projector.to(device)
        self.txt_projector = self.txt_projector.to(device)
        return super().to(device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Allow access to attributes of the base model
            # This allows us to access methods like `encode_image` and `encode_text`
            return getattr(self.base_model, name)

    def encode_image(self, images):
        with torch.no_grad():
            feats = self.base_model.encode_image(images)
        out = self.img_projector(feats)
        return out 

    def encode_text(self, texts):
        with torch.no_grad():
            feats = self.base_model.encode_text(texts)
        out = self.txt_projector(feats)
        return out
    
    def forward(self, images, texts):
        """
        Forward pass through the model.
        """
        # encode image and text
        image_feats = self.encode_image(images)
        text_feats = self.encode_text(texts)
        return image_feats, text_feats


def CKCLearning(model, dataset, preprocessor_list, config, logger, device):
    """
    model: model with encode_image and encode_text method
    dataset: dataset with __getitem__ and __len__ method
    preprocessor_list: list of preprocessors for data augmentation
    config: configuration dictionary
    """
    ### --------------------
    ### Instantiation
    ### --------------------
    scaler = torch.amp.GradScaler()

    # hyperparameters
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    lr = config.get("learning_rate", 1e-4)
    num_workers = config.get("num_workers", 4)
    ckc_temperature = config.get("ckc_temperature", 0.07)
    ckc_temperature_learnable = config.get("ckc_temperature_learnable", False)
    temperature_init = config.get("temperature_init", 0.07)
    temperature_learnable = config.get("temperature_learnable", False)
    amp_enabled = config.get("amp_enabled", False)
    update_iter = config.get("update_iter", 1)
    max_iter = config.get("max_iter", None)
    infonce_weight = config.get("infonce_weight", 0.5)

    # prepare model
    if not isinstance(model, FrozenWithProjector):
        # new model
        base = model.module if hasattr(model, 'module') else model
        base.eval()
        for p in base.parameters():
            p.requires_grad = False 
        projector_dim_in = config.get("projector_dim_in", 512)
        projector_dim_out = config.get("projector_dim_out", 512)
        img_projector = MLPProjector(projector_dim_in, projector_dim_out)
        txt_projector = MLPProjector(projector_dim_in, projector_dim_out)
        new_model = FrozenWithProjector(base, img_projector, txt_projector).to(get_main_device(config.get("gpu", None)))
        new_model, _ = gpu_prep(new_model, config.get("gpu", None))
        new_model.train()

        # old model
        old_model = copy.deepcopy(base)
        old_model.to(get_main_device(config.get("gpu", None)))
        old_model, _ = gpu_prep(old_model, config.get("gpu", None))
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False

    else:
        # new model
        old_model = model.old_model
        old_model.eval()

        # old model
        new_model = model
        new_model.train()
        

    # prepare optimizer
    if temperature_learnable:
        logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature_init)), requires_grad=True)
        optimizer = optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, new_model.parameters())},
            {"params": [logit_scale]}
        ], lr=lr)
    else: 
        logit_scale = None
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, new_model.parameters()), lr=lr)
    
    # DataLoader
    if isinstance(dataset, torch.utils.data.IterableDataset):
        dataset.shuffle()
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn 
        )
    
    # Start training
    logger.info(f"CKC Learning Training on dataset {dataset.name} Started----")
    break_flag = False
    with logging_redirect_tqdm(loggers=[logger]) and tqdm(total=max_iter, desc=f"CKC Training Progress") as pbar:
        global_iter = 0
        
        for epoch in range(epochs):
            if break_flag:
                break
            
            for i, (images, captions) in enumerate(train_loader):
                if max_iter is not None and global_iter >= max_iter:
                    break_flag = True
                    break

                # deal with the case when one image corresponds to multiple captions
                ctx = None
                if dataset.config.get("num_captions", 1) > 1:
                    captions = [caption[random.randint(0, len(caption)-1)] for caption in captions]

                # data preprocessing
                for preprocessor in preprocessor_list:
                    images, captions, ctx = preprocessor.preprocess(images, captions, ctx)

                ### Each iteration
                # 1) old_model features (no_grad)
                with torch.no_grad():
                    old_image_embeds, old_text_embeds = old_model(images, captions)

                # 2) new_model features
                image_embeds, text_embeds = new_model(images, captions)

                # 4) CKC Loss
                _ckc_loss = ckc_loss(old_image_embeds, old_text_embeds, image_embeds, text_embeds, temperature=ckc_temperature)
                # _ckc_loss = ckc_loss_test(old_image_embeds, old_text_embeds, image_embeds, text_embeds, temperature=ckc_temperature)
                _loss = _ckc_loss
                
                # 5) InfoNCE Loss
                if temperature_learnable:
                    temperature = (1.0 / logit_scale.exp()).clamp(min=0.01, max=1.0)
                else:
                    temperature = temperature_init
                _infonce_loss = infonce_loss(image_embeds, text_embeds, temperature=temperature)
                _loss += _infonce_loss * infonce_weight

                # 5) Optimization
                optimizer.zero_grad()
                scaler.scale(_loss).backward()
                scaler.unscale_(optimizer)
                all_params = list(model.parameters())
                if temperature_learnable:
                    all_params += [logit_scale]
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                
                # clamp logit scale if it's learnable
                if temperature_learnable:
                    with torch.no_grad():
                        logit_scale.clamp_(0.0, math.log(100.0))

                # update progress bar
                global_iter += 1
                pbar.set_postfix(loss=_loss.item(), epoch=epoch+1)
                pbar.update(1)

                # update the old_model every `update_iter` iterations
                if global_iter % update_iter == 0:
                    old_model = copy.deepcopy(new_model)
                    old_model.eval()
                    for param in old_model.parameters():
                        param.requires_grad = False

    logger.info("Contrastive Learning Training Finished----")
    model.old_model = new_model
    
    return new_model
