from framework.preprocessor import Preprocessor
import clip
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import litellm

class ImagePreprocessor(Preprocessor):
    """
    Standard CLIP preprocessing:
    1. Resize (maintaining aspect ratio)
    2. Center crop to square
    3. Convert to RGB if needed
    4. Convert to tensor and normalize with CLIP-specific values
    """
    def __init__(self, config):
        self.config = config
        
        # Default image size for CLIP (224x224)
        self.image_size = config.get('image_size', 224)
        
        # CLIP standard preprocessing from OpenAI
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        
        # For text, we'll rely on the tokenizer at the model level
        # This preprocessor is mainly for image preprocessing
    
    def preprocess(self, img_data, txt_data, context=None):
        # Process images if they are PIL Images or numpy arrays
        processed_images = []
        for img in img_data:
            if img is not None:
                # Convert numpy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                # Apply transformation if it's a PIL Image
                if isinstance(img, Image.Image):
                    processed_img = self.transform(img)
                    processed_images.append(processed_img)
                else:
                    # If already a tensor or other format, keep as is
                    processed_images.append(img)
            else:
                processed_images.append(None)
        
        for text in txt_data:
            if text is not None:
                prompt = "You are a helpful assistant, please rewrite the following caption "
        img_data = processed_images
        return img_data, txt_data, context


class MaskedPreprocessor(Preprocessor):
    """
    Masked Autoencoder (MAE) style preprocessing:
    - Divides the image into patches
    - Randomly masks a large portion (e.g., 75%) of the patches
    - Keeps the identities of the masked patches for later reconstruction
    """
    def __init__(self, config):
        self.config = config
        
        # Image size, patch size, and masking ratio
        self.image_size = config.get('image_size', 224)
        self.patch_size = config.get('patch_size', 16)  # Default patch size in ViT/MAE
        self.mask_ratio = config.get('mask_ratio', 0.5)  # Default 50% masking as in MAE
        
        # Base CLIP transform (without normalization, which comes after masking)
        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
        ])
        
        # Normalization to apply after masking
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    
    def apply_mask(self, img_tensor):
        """Apply random masking to patches of the image"""
        # Number of patches in each dimension
        n_patches = self.image_size // self.patch_size
        total_patches = n_patches * n_patches
        
        # Calculate how many patches to keep (not mask)
        num_keep = int(total_patches * (1 - self.mask_ratio))
        
        # Create a list of patch indices and shuffle
        patch_indices = list(range(total_patches))
        random.shuffle(patch_indices)
        
        # Select which patches to keep
        keep_indices = patch_indices[:num_keep]
        keep_indices.sort()  # Sort to make sequential access more efficient
        
        # Create masked image tensor
        # We'll reshape to patches, mask, and reshape back
        _, c, h, w = 1, 3, self.image_size, self.image_size
        
        # Reshape to patches
        patches = img_tensor.reshape(
            c, 
            n_patches, self.patch_size, 
            n_patches, self.patch_size
        ).permute(1, 3, 0, 2, 4).reshape(total_patches, c, self.patch_size, self.patch_size)
        
        # Create a new tensor of zeroed patches
        masked_patches = torch.zeros_like(patches)
        
        # Fill in the patches to keep
        for idx in keep_indices:
            masked_patches[idx] = patches[idx]
        
        # Reshape back to image
        masked_img = masked_patches.reshape(
            n_patches, n_patches, c, self.patch_size, self.patch_size
        ).permute(2, 0, 3, 1, 4).reshape(c, h, w)
        
        # Store masking information in a dict
        mask_info = {
            'kept_indices': keep_indices,
            'mask_ratio': self.mask_ratio,
            'n_patches': n_patches,
            'patch_size': self.patch_size,
            'original_shape': (c, h, w)
        }
        
        return masked_img, mask_info
    
    def preprocess(self, img_data, txt_data, context=None):
        processed_images = []
        mask_info_list = []
        
        for img in img_data:
            if img is not None:
                # Convert numpy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                # Apply base transformation if it's a PIL Image
                if isinstance(img, Image.Image):
                    # Apply base transform to get tensor
                    img_tensor = self.base_transform(img)
                    
                    # Apply masking
                    masked_img, mask_info = self.apply_mask(img_tensor)
                    
                    # Apply normalization
                    normalized_masked_img = self.normalize(masked_img)
                    
                    processed_images.append(normalized_masked_img)
                    mask_info_list.append(mask_info)
                else:
                    # If it's not a PIL image, keep as is and append None for mask info
                    processed_images.append(img)
                    mask_info_list.append(None)
            else:
                processed_images.append(None)
                mask_info_list.append(None)
        
        img_data = processed_images
        # Store mask information in the dataset for later use in reconstruction
        if context is None:
            context = {}
        context['mask_info'] = mask_info_list
        
        return img_data, txt_data, context


class PatchCutPreprocessor(Preprocessor):
    """
    Preprocessor that cuts an image into 4 patches, selects one, and resizes it.
    """
    def __init__(self, config):
        self.config = config
        
        # Image size for final output
        self.image_size = config.get('image_size', 224)
        
        # Patch selection strategy
        self.patch_selection = config.get('patch_selection', 'random')  # 'random' or 'external'
        
        # CLIP normalization parameters
        self.clip_normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    
    def select_patch(self, image, idx=None):
        """
        Cut image into 4 equal patches and select one.
        
        Args:
            image: PIL Image or tensor
            idx: Optional patch index (0-3) to select. If None, uses the selection strategy.
        
        Returns:
            The selected patch as a PIL Image
        """
        # Convert to PIL image if tensor
        if isinstance(image, torch.Tensor):
            # Convert tensor [C,H,W] to PIL
            image = transforms.ToPILImage()(image)
        
        width, height = image.size
        half_width, half_height = width // 2, height // 2
        
        # Define the 4 patches - (left, top, right, bottom)
        patches = [
            (0, 0, half_width, half_height),              # Top-left
            (half_width, 0, width, half_height),          # Top-right
            (0, half_height, half_width, height),         # Bottom-left
            (half_width, half_height, width, height)      # Bottom-right
        ]
        
        # Select patch based on strategy
        if idx is not None:
            # Use provided index
            patch_idx = idx % 4
        elif self.patch_selection == 'random':
            # Random selection
            patch_idx = random.randint(0, 3)
        elif self.patch_selection == 'external':
            # External modules should have set a patch index somehow
            # Default to random if not specified
            patch_idx = getattr(self, 'external_patch_idx', random.randint(0, 3))
        else:
            # Fallback to random
            patch_idx = random.randint(0, 3)
        
        # Crop the selected patch
        patch = image.crop(patches[patch_idx])
        
        return patch, patch_idx
    
    def preprocess(self, img_data, txt_data, context=None):
        processed_images = []
        patch_indices = []
        
        for img in img_data:
            if img is not None:
                # Convert numpy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                # Apply transformation if it's a PIL Image
                if isinstance(img, Image.Image):
                    # Select a patch
                    patch, patch_idx = self.select_patch(img)
                    
                    # Resize to target size
                    resized_patch = transforms.Resize(
                        self.image_size, 
                        interpolation=transforms.InterpolationMode.BICUBIC
                    )(patch)
                    
                    # Center crop to ensure square
                    cropped_patch = transforms.CenterCrop(self.image_size)(resized_patch)
                    
                    # Convert to tensor and normalize
                    tensor_patch = transforms.ToTensor()(cropped_patch)
                    normalized_patch = self.clip_normalize(tensor_patch)
                    
                    processed_images.append(normalized_patch)
                    patch_indices.append(patch_idx)
                else:
                    # Keep as is
                    processed_images.append(img)
                    patch_indices.append(None)
            else:
                processed_images.append(None)
                patch_indices.append(None)
        
        img_data = processed_images
        # Store which patch was selected for each image
        if context is None:
            context = {}
        context['patch_indices'] = patch_indices
        
        return img_data, txt_data, context


class GPTCaptionPreprocessor(Preprocessor):
    """
    Preprocessor that uses GPT to rewrite captions
    """
    def __init__(self, config):
        self.config = config
        self.prompt = config.get('prompt', "You are a helpful assistant, please rewrite the following caption to be more descriptive and detailed: ")
        self.model = config.get('model', 'gpt-4o-mini')
        
    def preprocess(self, img_data, txt_data, context=None):
        processed_captions = []
        for text in txt_data:
            if text is not None:
                try:
                    # Call litellm to rewrite the caption
                    response = litellm.completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.prompt},
                            {"role": "user", "content": text}
                        ]
                    )
                    # Extract the rewritten caption from the response
                    rewritten_caption = response.choices[0].message.content
                    processed_captions.append(rewritten_caption)
                except Exception as e:
                    print(f"Error processing caption: {e}")
                    processed_captions.append(text)  # Keep original caption if there's an error
            else:
                processed_captions.append(None)
        
        txt_data = processed_captions
        return img_data, txt_data, context


class TextPreprocessor(Preprocessor):
    def __init__(self, config):
        self.config = config
        self.cutoff_length = config.get('cutoff_length', 77)
        self.device = config.get('device', 'cpu')

    def preprocess(self, img_data, txt_data, context=None):
        processed_captions = []
        for text in txt_data:
            if text is not None:
                tokens = clip.tokenize([text], truncate=True)
                
                if tokens.size(1) > self.cutoff_length:
                    tokens = tokens[:, :self.cutoff_length]
                tokens = tokens.to(self.device)
                processed_captions.append(tokens)
            else:
                processed_captions.append(None)
        
        txt_data = processed_captions
        return img_data, txt_data, context