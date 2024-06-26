import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitModuleForImageClassification(LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        val_loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LitModuleForFusion(LightningModule):
    def __init__(self, cloud_model, control_model, shared_encoder, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["cloud_model", "control_model"])
        self.cloud_model = cloud_model
        self.control_model = control_model
        self.shared_encoder = shared_encoder
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        features = self.shared_encoder(inputs)
        cloud_outputs = self.cloud_model(features)
        control_outputs = self.control_model(features)
        outputs = cloud_outputs + control_outputs
        loss = F.cross_entropy(outputs, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
