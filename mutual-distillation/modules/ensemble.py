import torch
from scipy import stats


class Ensemble(torch.nn.Module):
    def __init__(self, models, mode="mean"):
        super(Ensemble, self).__init__()
        self.models = models
        assert mode in ["mean", "voting"]
        self.mode = mode

    def forward(self, x):
        if self.mode == "mean":
            outputs = torch.stack([model(x) for model in self.models])
            return torch.mean(outputs, dim=0)
        elif self.mode == "voting":
            outputs = torch.stack([model(x).argmax(dim=1) for model in self.models])
            outputs = outputs.detach().cpu().numpy()
            voted_outputs = stats.mode(outputs, axis=0)[0]
            return torch.from_numpy(voted_outputs).to(x.device)
