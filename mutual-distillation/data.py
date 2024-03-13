import torch
from torch.utils.data import Dataset


class ExtractedFeaturesDataset(Dataset):
    def __init__(self, original_dataset, feature_extractor):
        super().__init__()
        self.original_dataset = original_dataset
        self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        data = self.feature_extractor(data)
        return data, target
    
    def __len__(self):
        return len(self.original_dataset)