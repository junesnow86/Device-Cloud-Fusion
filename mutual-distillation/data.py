import torch
from torch.utils.data import Dataset


class ExtractedFeaturesDataset(Dataset):
    def __init__(self, original_dataset, feature_extractor, device="cuda"):
        super().__init__()
        self.original_dataset = original_dataset
        self.feature_extractor = feature_extractor
        self.device = device

    def __getitem__(self, index):
        data, target = self.original_dataset[index]
        data_original_device = data.device
        data = data.to(self.device)
        fe_original_device = next(self.feature_extractor.parameters()).device
        self.feature_extractor.to(self.device)
        data = data.unsqueeze(0)
        with torch.no_grad():
            data = self.feature_extractor(data)
        data = data.squeeze(0)
        data = data.to(data_original_device)
        self.feature_extractor.to(fe_original_device)
        return data, target
    
    def __len__(self):
        return len(self.original_dataset)
