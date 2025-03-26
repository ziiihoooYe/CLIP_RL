from framework.dataset import Dataset
from torchvision.datasets import CocoCaptions

def coco_collate_fn(batch):
    images, captions = zip(*batch)
    images = list(images)
    captions = list(captions)
    return images, captions


class MSCOCOVal2017Dataset(Dataset):
    def __init__(self, config):
        super(MSCOCOVal2017Dataset, self).__init__(config)
        image_path = self.config['image_path']
        annotations_path = self.config['annotations_path']
        self.data = CocoCaptions(root=image_path, annFile=annotations_path, transform=None)
        self.collate_fn = coco_collate_fn
        self.num_captions = self.config.get("num_captions", None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, captions = self.data[idx]
        if self.num_captions is not None:
            captions = captions[:self.num_captions]
        return img, captions