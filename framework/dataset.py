from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset

class Dataset(TorchDataset):
    def __init__(self, config):
        super(Dataset, self).__init__()
        self.config = config
        self.name = config.get("name", "Dataset")
        
        # Dataset class should have the following attributes:
        # - img_data: list of image data samples
        # - text_data: list of text data samples
        # - collate_fn: function to collate data samples into batches
        self.img_data = None 
        self.text_data = None
        self.collate_fn = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample
        Returns:
            Tuple: ()
        """
        raise NotImplementedError
    
class IterDataset(IterableDataset):
    def __init__(self, config):
        super(IterDataset, self).__init__()
        self.config = config
        self.name = config.get("name", "Dataset")
        
        # Dataset class should have the following attributes:
        # - img_data: list of image data samples
        # - text_data: list of text data samples
        self.img_data = None 
        self.text_data = None

    def __iter__(self):
        """
        Args:
            idx: Index of the sample
        Returns:
            Tuple: ()
        """
        raise NotImplementedError