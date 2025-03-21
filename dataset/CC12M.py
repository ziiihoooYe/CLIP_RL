from PIL import Image
from framework.dataset import IterDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

def cc12m_collate_fn(batch):
    images, captions = zip(*batch)
    images = list(images)
    captions = list(captions)
    return images, captions

class CC12MDataset(IterDataset):
    def __init__(self, config): 
        self.dataset = load_dataset("pixparse/cc12m-wds", split="train", streaming=True)
        self.config = config
        self.collate_fn = cc12m_collate_fn

    def __getitem__(self, idx):
        """
        Loads an image and caption dynamically from disk.
        """
        for i, data_item in enumerate(self.dataset):
            if i == idx:  # Workaround since indexing is not possible in streaming mode
                try:
                    # Load image
                    image = data_item['jpg']
                    
                    # image = self.transform(image)

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
                # print(data_item)
                image = data_item['jpg']
                
                # image = self.transform(image)

                caption = data_item['json']['caption']

                # print(image, caption)

                yield image, caption
            except Exception as e:
                print(f"Error loading data: {e}")
                continue




# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# # Fetch a batch of data
# for batch in dataloader:
#     images, captions = batch
#     print(f"Batch Images Shape: {images.shape}")
#     print(f"Batch Captions: {captions}")
#     break  # Process only one batch




# transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# dataset = CC12MDataset(transform=transform)

# for i, (image, caption) in enumerate(dataset):
    
#     print(f"Image {i}: {image}, Caption: {caption}")