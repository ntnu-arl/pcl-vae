import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob

class RangeImageDataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self._get_all_paths(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _get_all_paths(self, root_dir):
        # Recursively get all file paths from the root directory
        return glob.glob(os.path.join(root_dir, '**', '*.npy'), recursive=True)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        range_path = self.file_paths[idx]
        range_np = np.load(range_path)
        
        range = self.transform(range_np)
        
        return range