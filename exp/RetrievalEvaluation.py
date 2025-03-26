import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from framework.experiments import Experiment
from utils.utils import get_gpu_device
from tqdm.contrib.logging import logging_redirect_tqdm

class RetrievalEvaluationExperiment(Experiment):
    def __init__(self, config):
        super(RetrievalEvaluationExperiment, self).__init__(config)
    
    def run(self, model, dataset, preprocessor_list, logger):
        # Data Preparation and Hyperparameters
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 512),
            num_workers=self.config.get("num_workers", 4),
            collate_fn=dataset.collate_fn
        )
        sample_size = self.config.get("sample_size", 5000)
        logger.info(f"Evaluating retrieval performance on {sample_size} samples")
        device = get_gpu_device(self.config.get("gpu", None))
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
        results = evaluate_retrieval(image_features, text_features, num_captions)
        
        metrics = ["I2T_top1", "I2T_top5", "I2T_top10", "T2I_top1", "T2I_top5", "T2I_top10"]
        values = [results[metric] for metric in metrics]
        df = pd.DataFrame([values], columns=metrics)
        logger.info(f"\n{df.to_string()}")
    

def evaluate_retrieval(image_features, text_features, num_captions=5):
    """
    Evaluate image-to-text (I2T) and text-to-image (T2I) retrieval performance.
    
    Args:
        image_features (torch.Tensor): shape -> (N, D)
        text_features (torch.Tensor): shape -> (N * num_captions, D)
        num_captions (int): number of captions per image
    
    Returns:
        dict: 
            {
                'I2T_top1': float,
                'I2T_top5': float,
                'I2T_top10': float,
                'T2I_top1': float,
                'T2I_top5': float,
                'T2I_top10': float
            }
    """
    num_images = image_features.size(0)
    assert text_features.size(0) == num_images * num_captions, "Number of captions should match number of images"

    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity matrix
    similarity = image_features @ text_features.t()
    
    # Evaluate image-to-text (I2T)
    I2T_top1, I2T_top5, I2T_top10 = 0, 0, 0
    for i in range(num_images):
        sim_i = similarity[i]  # similarity of the i-th image with all texts
        sorted_indices = torch.argsort(sim_i, descending=True)
        gt_indices = list(range(i * num_captions, i * num_captions + num_captions))
        if any(idx in sorted_indices[:1] for idx in gt_indices):
            I2T_top1 += 1
        if any(idx in sorted_indices[:5] for idx in gt_indices):
            I2T_top5 += 1
        if any(idx in sorted_indices[:10] for idx in gt_indices):
            I2T_top10 += 1

    I2T_top1_score = I2T_top1 / num_images
    I2T_top5_score = I2T_top5 / num_images
    I2T_top10_score = I2T_top10 / num_images

    # Evaluate text-to-image (T2I)
    similarity_t = similarity.t()
    T2I_top1, T2I_top5, T2I_top10 = 0, 0, 0
    for j in range(text_features.size(0)):
        sim_j = similarity_t[j]
        sorted_indices = torch.argsort(sim_j, descending=True)
        gt_image = j // num_captions
        if gt_image in sorted_indices[:1]:
            T2I_top1 += 1
        if gt_image in sorted_indices[:5]:
            T2I_top5 += 1
        if gt_image in sorted_indices[:10]:
            T2I_top10 += 1

    total_texts = text_features.size(0)
    T2I_top1_score = T2I_top1 / total_texts
    T2I_top5_score = T2I_top5 / total_texts
    T2I_top10_score = T2I_top10 / total_texts

    results = {
        'I2T_top1': I2T_top1_score,
        'I2T_top5': I2T_top5_score,
        'I2T_top10': I2T_top10_score,
        'T2I_top1': T2I_top1_score,
        'T2I_top5': T2I_top5_score,
        'T2I_top10': T2I_top10_score,
    }

    return results