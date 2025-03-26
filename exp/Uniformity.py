import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from framework.experiments import Experiment
from utils.utils import get_gpu_device
from tqdm.contrib.logging import logging_redirect_tqdm

class UniformityExperiment(Experiment):
    def __init__(self, config):
        super(UniformityExperiment, self).__init__(config)
    
    def run(self, model, dataset, preprocessor_list, logger):
        # Data Preparation and Hyperparameters
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 512),
            num_workers=self.config.get("num_workers", 4),
            collate_fn=dataset.collate_fn
        )
        sample_size = self.config.get("sample_size", 5000)
        logger.info(f"Evaluating Uniformity & Alignment performance on {sample_size} samples")
        num_captions = dataset.config.get("num_captions", 1)
        loader_len = len(train_loader) if hasattr(train_loader, '__len__') else None
        if loader_len is not None:
            loader_len = min(loader_len, sample_size // self.config.get("batch_size", 512) + 1)
        model.eval()
        
        # Extract Features
        image_features = []
        text_features = []
        processed_samples = 0

        with logging_redirect_tqdm(loggers=[logger]) and torch.no_grad():
            for i, (images, captions) in enumerate(tqdm(train_loader, total=loader_len, desc="Extracting Features")):
                batch_size = len(images)

                if processed_samples >= sample_size:
                    break
                if processed_samples + batch_size > sample_size:
                    valid_count = sample_size - processed_samples
                    images = images[:valid_count]
                    captions = captions[:valid_count]
                    processed_samples = sample_size
                else:
                    processed_samples += batch_size

                ctx = None
                for preprocessor in preprocessor_list:
                    images, captions, ctx = preprocessor.preprocess(images, captions, ctx)

                image_feature = model.encode_image(images)
                text_feature  = model.encode_text(captions)
                
                image_features.append(image_feature)
                text_features.append(text_feature)
                

            image_features = torch.cat(image_features).squeeze()
            text_features = torch.cat(text_features).squeeze()

        # Evaluate Retrieval
        alignment_results = compute_alignment(text_features.cpu().numpy(), image_features.cpu().numpy(), captions_per_image=num_captions)
        img_uniformity_results = compute_uniformity_approx(image_features.cpu().numpy())
        txt_uniformity_results = compute_uniformity_approx(text_features.cpu().numpy())
        metrics = ["Alignment", "Image Uniformity", "Text Uniformity"]
        values = [alignment_results, img_uniformity_results, txt_uniformity_results]
        df = pd.DataFrame([values], columns=metrics)
        logger.info(f"\n{df.to_string()}")
    

# alignment metric
# features: (N, D)
def compute_alignment(text_features, image_features, alpha=2, captions_per_image=5):
    # image_features: shape [N, D]
    # text_features: shape [N * captions_per_image, D]
    num_images, D = image_features.shape

    # [N, captions_per_image, D]
    text_features = text_features.reshape(num_images, captions_per_image, D)

    # image_features 为 [N, 1, D] 以便广播
    image_features_expanded = image_features[:, None, :]
    
    # Difference between image and text features
    diff = text_features - image_features_expanded  # shape: [N, captions_per_image, D]
    dist = np.linalg.norm(diff, axis=2)  # shape: [N, captions_per_image]

    return np.mean(dist ** alpha)

# uniformity metric
def compute_uniformity_approx(features, t=2.0, num_samples=100000):
    """
    Approximate uniformity via random sampling of pairs.
    
    L = log( E_{x,y}[ exp(-t ||x-y||^2) ] )
    """
    N = features.shape[0]
    # Randomly sample pairs (i, j)
    idx1 = np.random.randint(0, N, size=num_samples)
    idx2 = np.random.randint(0, N, size=num_samples)
    
    # Compute squared distances for the sampled pairs
    diff = features[idx1] - features[idx2]
    dist2 = np.sum(diff * diff, axis=-1)
    
    # Compute exp(-t * dist^2) and take the average
    values = np.exp(-t * dist2)
    avg = np.mean(values)
    
    return np.log(avg)