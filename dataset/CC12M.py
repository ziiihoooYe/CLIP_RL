from PIL import Image
from framework.dataset import IterDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset


def cc_collate_fn(batch):
    images, captions = zip(*batch)
    images = list(images)
    captions = list(captions)
    return images, captions


class CCDataset(IterDataset):
    def __init__(self, config): 
        super(CCDataset, self).__init__(config)
        self.collate_fn = cc_collate_fn

    def __getitem__(self, idx):
        """
        Loads an image and caption dynamically from disk.
        """
        for i, data_item in enumerate(self.dataset):
            if i == idx:  # Workaround since indexing is not possible in streaming mode
                try:
                    # Load image
                    image = data_item['jpg']
                except Exception as e:
                    print(f"Error loading image at index {idx}: {e}")
                    return None
                try:
                    caption = data_item['json']['caption']
                except Exception as e:
                    print(f"Error loading caption at index {idx}: {e}")
                    return None
                return image, caption
        raise IndexError(f"Index {idx} out of bounds")
    
    def __iter__(self):
        """ Allow iteration over dataset """
        for data_item in self.dataset:
            try:
                image = data_item['jpg']
                caption = data_item['json']['caption']
                yield image, caption
            except (SyntaxError, OSError, ValueError) as e:
                continue
            except Exception as e:
                continue
            

class CC3MTrainDataset(CCDataset):
    def __init__(self, config): 
        super(CC3MTrainDataset, self).__init__(config)
        self.dataset = load_dataset("pixparse/cc3m-wds", split="train", streaming=True)


class CC3MValDataset(CCDataset):
    def __init__(self, config): 
        super(CC3MValDataset, self).__init__(config)
        self.dataset = load_dataset("pixparse/cc3m-wds", split="validation", streaming=True)


class CC12MTrainDataset(CCDataset):
    def __init__(self, config): 
        super(CC12MTrainDataset, self).__init__(config)
        self.dataset = load_dataset("pixparse/cc12m-wds", split="train", streaming=True)
