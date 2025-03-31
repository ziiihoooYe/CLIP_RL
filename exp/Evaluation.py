import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from framework.experiments import Experiment
from tqdm.contrib.logging import logging_redirect_tqdm
try:
    import pymp
    if not hasattr(pymp, 'Parallel'):
         raise AttributeError("pymp does not have Parallel attribute")
    pymp_available = True
except (ImportError, AttributeError):
    pymp_available = False
    print("pymp library is not available or does not support Parallel. Falling back to sequential mode. For parallel speedup, please install a version of pymp that supports Parallel, e.g., `pip install pymp-pypi`")


class EvaluationExperiment(Experiment):
    def __init__(self, config):
        super(EvaluationExperiment, self).__init__(config)
        self.metrics = config.get('metrics', {})

    
    def run(self, model, dataset, preprocessor_list, logger, device):
        # iter of the model
        iter_now = 0 if getattr(model, 'iter_now') is None else getattr(model, 'iter_now')

        # Data Preparation and Hyperparameters
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 512),
            num_workers=self.config.get("num_workers", 4),
            collate_fn=dataset.collate_fn
        )
        sample_size = self.config.get("sample_size", 5000)
        num_captions = dataset.config.get("num_captions", 1)
        loader_len = len(train_loader) if hasattr(train_loader, '__len__') else None
        if loader_len is not None:
            loader_len = min(loader_len, sample_size // self.config.get("batch_size", 512) + 1) 
        model.eval()
        logger.info(f"Extracting Representations on dataset {dataset.name} with {sample_size} samples (iter={iter_now}).")
        
        # Extract Features
        image_features = []
        text_features = []
        processed_samples = 0

        with logging_redirect_tqdm(loggers=[logger]) and torch.no_grad():
            for i, (images, captions) in enumerate(tqdm(train_loader, total=loader_len, desc="Extracting Features")):

                # deal with the case when one image corresponds to multiple captions
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

                # data preprocessing
                ctx = None
                for preprocessor in preprocessor_list:
                    images, captions, ctx = preprocessor.preprocess(images, captions, ctx)

                # Extract features from the model
                image_feature, text_feature = model(images, captions) 
                image_features.append(image_feature)
                text_features.append(text_feature) 

            image_features = torch.cat(image_features).squeeze()
            text_features = torch.cat(text_features).squeeze()

        # store the features for the current iteration
        model.stored_img_feat[f'iter: {str(iter_now)}'] = image_features
        model.stored_txt_feat[f'iter: {str(iter_now)}'] = text_features

        # Evaluate Metric
        for metric in self.metrics:
            metric_name = list(metric.keys())[0] if isinstance(metric, dict) else metric

            if metric_name == "retrieval":
                evaluate_retrieval(logger, image_features, text_features, num_captions)
            elif metric_name == "uniformity":
                evaluate_uniformity(logger, image_features, text_features, num_captions)
            elif metric_name == "platonic":
                trials = metric.get("trials", 10) if isinstance(metric, dict) else 10
                evaluate_platonic_metrics(logger, image_features, text_features, trials, num_captions) 
            elif metric_name == "eigenfunction":
                save_path = self.config.get("save_path", "./eigenfunction_analysis")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                eigenfunction_evaluate(model.stored_img_feat, model.stored_txt_feat, save_path, logger)
            else:
                raise NotImplementedError(f"Metric {metric} is not implemented")

        return model
    

### -------------------- Retrieval Evaluation -------------------- ###
def evaluate_retrieval(logger, image_features, text_features, num_captions=5):
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

    metrics = ['I2T_top1', 'I2T_top5', 'I2T_top10', 'T2I_top1', 'T2I_top5', 'T2I_top10']
    values = [I2T_top1_score, I2T_top5_score, I2T_top10_score, T2I_top1_score, T2I_top5_score, T2I_top10_score]
    df = pd.DataFrame([values], columns=metrics)
    logger.info(f"\n{df.to_string()}")
    
    
### -------------------- Alignment and Uniformity -------------------- ###
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


def evaluate_uniformity(logger, image_features, text_features, num_captions=5):
    img_uniformity_results = compute_uniformity_approx(image_features.cpu().numpy())
    txt_uniformity_results = compute_uniformity_approx(text_features.cpu().numpy())
    alignment_results = compute_alignment(text_features.cpu().numpy(), image_features.cpu().numpy(), captions_per_image=num_captions)
    metrics = ["Image Uniformity", "Text Uniformity", "Alignment"]
    values = [img_uniformity_results, txt_uniformity_results, alignment_results]
    df = pd.DataFrame([values], columns=metrics)
    logger.info(f"\n{df.to_string()}")


### -------------------- Platonic Metrics -------------------- ###

def evaluate_platonic_metrics(logger, image_features, text_features, trials=10, num_captions=5):

    metric_dict = {}
    for metric in AlignmentMetrics.SUPPORTED_METRICS:
        scores = []
        for t in range(trials):
            kwargs = {}
            if 'nn' in metric:
                kwargs['topk'] = 10
            if 'cca' in metric:
                kwargs['cca_dim'] = 10
            if 'kernel' in metric:
                kwargs['dist'] = 'sample'
            
            if num_captions > 1:
                text_features_copy = text_features.clone().view(image_features.size(0), num_captions, -1)
                rand_idx = torch.randint(0, num_captions, (image_features.size(0),))
                text_features_copy = text_features_copy[torch.arange(image_features.size(0)), rand_idx, :]
                score = AlignmentMetrics.measure(metric, image_features, text_features_copy, **kwargs)
                scores.append(score)
            else:
                score = AlignmentMetrics.measure(metric, image_features, text_features, **kwargs)
                scores.append(score)
        metric_dict[metric] = np.mean(scores)
    
    metrics = AlignmentMetrics.SUPPORTED_METRICS
    values = [metric_dict[metric] for metric in metrics]
    df = pd.DataFrame([values], columns=metrics)
    logger.info(f"\n{df.to_string()}")


class AlignmentMetrics:

    SUPPORTED_METRICS = [
        "cycle_knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "unbiased_cka",
        "cknna",
        "svcca",
        "edit_distance_knn",
    ]

    @staticmethod
    def measure(metric, *args, **kwargs):
        """ metric is a string for the function """

        if metric not in AlignmentMetrics.SUPPORTED_METRICS:
            raise ValueError(f"Unrecognized metric: {metric}")

        return getattr(AlignmentMetrics, metric)(*args, **kwargs)


    @staticmethod
    def cycle_knn(feats_A, feats_B, topk):
        """
        LLM nearest neighbors -> Query Language Pair -> LVM nearest neighbors
        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            acc: a float representing the accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   
        return compute_knn_accuracy(knn_A[knn_B]).item()


    @staticmethod
    def mutual_knn(feats_A, feats_B, topk):
        """
        Computes the mutual KNN accuracy.

        Args:
            feats_A: A torch tensor of shape N x feat_dim
            feats_B: A torch tensor of shape N x feat_dim

        Returns:
            A float representing the mutual KNN accuracy
        """
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)   

        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        # Create a range tensor for indexing
        range_tensor = torch.arange(n, device=knn_A.device).unsqueeze(1)

        # Create binary masks for knn_A and knn_B
        lvm_mask = torch.zeros(n, n, device=knn_A.device)
        llm_mask = torch.zeros(n, n, device=knn_A.device)

        lvm_mask[range_tensor, knn_A] = 1.0
        llm_mask[range_tensor, knn_B] = 1.0
        
        acc = (lvm_mask * llm_mask).sum(dim=1) / topk
        
        return acc.mean().item()
    
    
    @staticmethod
    def lcs_knn(feats_A, feats_B, topk):
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)        
        score = longest_ordinal_sequence(knn_A, knn_B).float().mean()
        return score
    
    
    @staticmethod
    def cka(feats_A, feats_B, kernel_metric='ip', rbf_sigma=1.0, unbiased=False):
        """Computes the unbiased Centered Kernel Alignment (CKA) between features."""
        
        if kernel_metric == 'ip':
            # Compute kernel matrices for the linear case
            K = torch.mm(feats_A, feats_A.T)
            L = torch.mm(feats_B, feats_B.T)
        elif kernel_metric == 'rbf':
            # COMPUTES RBF KERNEL
            K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))
        else:
            raise ValueError(f"Invalid kernel metric {kernel_metric}")

        # Compute HSIC values
        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)

        # Compute CKA
        #print('hsic', hsic_kl)
        cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)        
        return cka_value.item()
    
    
    @staticmethod
    def unbiased_cka(*args, **kwargs):
        kwargs['unbiased'] = True
        return AlignmentMetrics.cka(*args, **kwargs)
    
    
    @staticmethod
    def svcca(feats_A, feats_B, cca_dim=10):
        
        from sklearn.cross_decomposition import CCA

        # Center and scale the activations
        def preprocess_activations(act):
            act = act - torch.mean(act, axis=0)
            act = act / (torch.std(act, axis=0) + 1e-8)
            return act

        feats_A = preprocess_activations(feats_A)
        feats_B = preprocess_activations(feats_B)

        # Compute SVD
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
        
        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Compute CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # sometimes it goes to nan, this is just to avoid that
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute SVCCA similarity
        svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
        )
        return svcca_similarity
    
    
    @staticmethod
    def edit_distance_knn(feats_A, feats_B, topk):
        """
        Computes the edit distance between the nearest neighbors of feats_A and feats_B.
        """
        import torchaudio.functional as TAF
        
        knn_A = compute_nearest_neighbors(feats_A, topk)
        knn_B = compute_nearest_neighbors(feats_B, topk)
        
        # given N x topk with integer entries, compute edit distance
        n = knn_A.shape[0]
        topk = knn_A.shape[1]

        edit_distance = compute_distance(knn_A, knn_B, TAF.edit_distance)
        return 1 - torch.mean(edit_distance) / topk
    
    
    @staticmethod
    def cknna(feats_A, feats_B, topk=None, distance_agnostic=False, unbiased=True):
        """ similarity only cka variant """
        n = feats_A.shape[0]
                
        if topk < 2:
            raise ValueError("CKNNA requires topk >= 2")
        
        if topk is None:
            topk = feats_A.shape[0] - 1
                            
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        device = feats_A.device

        def similarity(K, L, topk):                         
            if unbiased:            
                K_hat = K.clone().fill_diagonal_(float("-inf"))
                L_hat = L.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = K, L

            # get topk indices for each row
            # if unbiased we cannot attend to the diagonal unless full topk
            # else we can attend to the diagonal
            _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
            
            # create masks for nearest neighbors
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
            
            # intersection of nearest neighbors
            mask = mask_K * mask_L
                        
            if distance_agnostic:
                sim = mask * 1.0
            else:
                if unbiased:
                    sim = hsic_unbiased(mask * K, mask * L)
                else:
                    sim = hsic_biased(mask * K, mask * L)
            return sim

        sim_kl = similarity(K, L, topk)
        sim_kk = similarity(K, K, topk)
        sim_ll = similarity(L, L, topk)
                
        return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def hsic_unbiased(K, L):
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

    
def compute_knn_accuracy(knn):
    """
    Compute the accuracy of the nearest neighbors. Assumes index is the gt label.
    Args:
        knn: a torch tensor of shape N x topk
    Returns:
        acc: a float representing the accuracy
    """
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc
    

def compute_nearest_neighbors(feats, topk=1):
    """
    Compute the nearest neighbors of feats
    Args:
        feats: a torch tensor of shape N x D
        topk: the number of nearest neighbors to return
    Returns:
        knn: a torch tensor of shape N x topk
    """
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


def longest_ordinal_sequence(X, Y):
    """ For each pair in X and Y, compute the length of the longest sub-sequence (LCS) """
    
    def lcs_length(x, y):
        """
        Compute the length of the longest common subsequence between two sequences.
        This is a classic dynamic programming implementation.
        """
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    lcs = compute_distance(X, Y, lcs_length)
    return lcs


def compute_distance(X, Y, dist_fn):
    """ compute distance in parallel"""
    B, N = X.shape
    distances = np.zeros(B)
    X, Y = X.cpu().numpy(), Y.cpu().numpy()

    if pymp_available:
        with pymp.Parallel(4) as p:
            for i in p.range(B):
                distances[i] = dist_fn(X[i], Y[i])
    else:
        for i in range(B):
            distances[i] = dist_fn(X[i], Y[i])
    return torch.tensor(distances)


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        # sorts the whole tensor and gets the q-th percentile
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        # quantile for element in the tensor and take the average
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)


### --------------------- Eigenfunction Analysis --------------------- ###
def compare_topk_differences_sorted(model_repr_pairs, save_path, top_k=16, plot=True):

    data = {}
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(img_repr, "numpy"):
            img_repr = img_repr.cpu().numpy()
        if hasattr(txt_repr, "numpy"):
            txt_repr = txt_repr.cpu().numpy()
        
        mean_img = img_repr.mean(axis=0)
        mean_txt = txt_repr.mean(axis=0)
        diff = np.abs(mean_img - mean_txt)
        sorted_diff = np.sort(diff)[::-1]
        topk_values = sorted_diff[:top_k]
        data[model_name] = topk_values

    ranks = [f"Rank {i+1}" for i in range(top_k)]
    df_all = pd.DataFrame(data, index=ranks)

    if plot:
        plt.figure(figsize=(8, 5))
        for model_name in data.keys():
            plt.plot(df_all[model_name].values, marker='o', label=model_name)
        plt.title(f"Top-K Mean Differences (Descending) for Each Iter")
        plt.xlabel("Rank (1 = largest difference)")
        plt.ylabel("Absolute Mean Difference")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"topk_differences.png"))

    return df_all


def plot_multiple_models_sorted_stats(model_repr_pairs, save_path, top_k=16):
    """
    Plots four separate figures for each model:
     1) Image Mean (sorted descending)
     2) Image Variance (sorted descending)
     3) Text Mean (sorted descending)
     4) Text Variance (sorted descending)

    Args:
        model_repr_pairs (dict):
            {model_name: (img_repr, txt_repr)}, where img_repr/txt_repr is a torch.Tensor or numpy array of shape [N, D].
        save_path (str): directory to save the figures
        top_k (int): number of top elements to show in the sorted stats
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Figure 1: Image Mean
    plt.figure(figsize=(8, 5))
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(img_repr, 'cpu'):
            img_repr = img_repr.cpu().numpy()
        img_mean = img_repr.mean(axis=0)
        sorted_means = np.sort(img_mean)[::-1][:top_k]
        x = np.arange(1, top_k+1)
        plt.plot(x, sorted_means, marker='o', label=model_name)

    plt.title(f"Top {top_k} Image Mean (Sorted Descending)")
    plt.xlabel('Rank (1 = largest)')
    plt.ylabel('Mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"top_{top_k}_image_mean.png"))
    plt.close()

    # Figure 2: Image Variance
    plt.figure(figsize=(8, 5))
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(img_repr, 'cpu'):
            img_repr = img_repr.cpu().numpy()
        img_var = img_repr.var(axis=0)
        sorted_vars = np.sort(img_var)[::-1][:top_k]
        x = np.arange(1, top_k+1)
        plt.plot(x, sorted_vars, marker='o', label=model_name)

    plt.title(f"Top {top_k} Image Variance (Sorted Descending)")
    plt.xlabel('Rank (1 = largest)')
    plt.ylabel('Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"top_{top_k}_image_variance.png"))
    plt.close()

    # Figure 3: Text Mean
    plt.figure(figsize=(8, 5))
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(txt_repr, 'cpu'):
            txt_repr = txt_repr.cpu().numpy()
        txt_mean = txt_repr.mean(axis=0)
        sorted_means = np.sort(txt_mean)[::-1][:top_k]
        x = np.arange(1, top_k+1)
        plt.plot(x, sorted_means, marker='o', label=model_name)

    plt.title(f"Top {top_k} Text Mean (Sorted Descending)")
    plt.xlabel('Rank (1 = largest)')
    plt.ylabel('Mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"top_{top_k}_text_mean.png"))
    plt.close()

    # Figure 4: Text Variance
    plt.figure(figsize=(8, 5))
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(txt_repr, 'cpu'):
            txt_repr = txt_repr.cpu().numpy()
        txt_var = txt_repr.var(axis=0)
        sorted_vars = np.sort(txt_var)[::-1][:top_k]
        x = np.arange(1, top_k+1)
        plt.plot(x, sorted_vars, marker='o', label=model_name)

    plt.title(f"Top {top_k} Text Variance (Sorted Descending)")
    plt.xlabel('Rank (1 = largest)')
    plt.ylabel('Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"top_{top_k}_text_variance.png"))
    plt.close()


def plot_multiple_models_repr_stats_sorted(model_repr_pairs, save_path, top_k=16):

    stats_dict = {}
    for model_name, (img_repr, txt_repr) in model_repr_pairs.items():
        if hasattr(img_repr, 'cpu'):
            img_repr = img_repr.cpu().numpy()
        if hasattr(txt_repr, 'cpu'):
            txt_repr = txt_repr.cpu().numpy()
        img_mean = img_repr.mean(axis=0)
        img_var  = img_repr.var(axis=0)
        txt_mean = txt_repr.mean(axis=0)
        txt_var  = txt_repr.var(axis=0)

        diff = np.abs(img_mean - txt_mean)
        sorted_indices = np.argsort(diff)[::-1]
        topk_indices = sorted_indices[:top_k]

        sorted_img_mean = img_mean[topk_indices]
        sorted_img_var  = img_var[topk_indices]
        sorted_txt_mean = txt_mean[topk_indices]
        sorted_txt_var  = txt_var[topk_indices]

        stats_dict[model_name] = {
            'topk_indices': topk_indices,
            'img_mean': sorted_img_mean,
            'img_var': sorted_img_var,
            'txt_mean': sorted_txt_mean,
            'txt_var': sorted_txt_var
        }

    ranks = np.arange(1, top_k+1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    ax = axs[0, 0]
    for model_name, stats in stats_dict.items():
        ax.plot(ranks, stats['img_mean'], marker='o', label=model_name)
    ax.set_title("Image Mean (sorted by top difference)")
    ax.set_xlabel("Rank (1 = highest diff)")
    ax.set_ylabel("Mean")
    ax.legend()

    ax = axs[0, 1]
    for model_name, stats in stats_dict.items():
        ax.plot(ranks, stats['img_var'], marker='o', label=model_name)
    ax.set_title("Image Variance (sorted by top difference)")
    ax.set_xlabel("Rank (1 = highest diff)")
    ax.set_ylabel("Variance")
    ax.legend()

    ax = axs[1, 0]
    for model_name, stats in stats_dict.items():
        ax.plot(ranks, stats['txt_mean'], marker='o', label=model_name)
    ax.set_title("Text Mean (sorted by top difference)")
    ax.set_xlabel("Rank (1 = highest diff)")
    ax.set_ylabel("Mean")
    ax.legend()

    ax = axs[1, 1]
    for model_name, stats in stats_dict.items():
        ax.plot(ranks, stats['txt_var'], marker='o', label=model_name)
    ax.set_title("Text Variance (sorted by top difference)")
    ax.set_xlabel("Rank (1 = highest diff)")
    ax.set_ylabel("Variance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"repr_stats_sorted.png"))

    return stats_dict


def eigenfunction_evaluate(stored_img_results, stored_txt_results, save_path, logger, plot=True):

    model_repr_pairs = {
        iter_name: (stored_img_results[iter_name], stored_txt_results[iter_name])
        for iter_name in stored_img_results.keys()
    }
    
    # compare_topk_diff
    df_diff = compare_topk_differences_sorted(
        model_repr_pairs,
        save_path=save_path,
        plot=plot
    )
    logger.info(f"[EigenEvaluation] Top-K differences:\n{df_diff}")

    # plot_sorted_stats using the new signature
    plot_multiple_models_sorted_stats(model_repr_pairs, save_path=save_path)

    stats_dict = plot_multiple_models_repr_stats_sorted(model_repr_pairs, save_path=save_path)
    logger.info(f"[EigenEvaluation] Stats dict from repr analysis: \n{stats_dict}")
