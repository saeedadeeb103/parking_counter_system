import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import timm
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from omegaconf import DictConfig

# Define the paths to the "empty" and "not_empty" folders

# Define the target size for your images

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'image': self.X[idx], 'label': self.y[idx]}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

class ResNet18(pl.LightningModule):
    def __init__(self, num_classes=2, optimizer_cfg=None):
        super().__init__()

        # Load a pretrained ResNet-18 model from timm
        self.resnet = timm.create_model('resnet18', pretrained=True)

        # Modify the final classification layer to match the number of classes in your dataset
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, num_classes)
        if optimizer_cfg is not None:
            optimizer_name = optimizer_cfg.name
            optimizer_lr = optimizer_cfg.lr
            optimizer_weight_decay = optimizer_cfg.weight_decay

            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            elif optimizer_name == 'SGD':
                # You can add more optimizers as needed
                self.optimizer = optim.SGD(self.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        else:
            self.optimizer = None
        print("Optimizer Used:", self.optimizer)

    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y = y.long()

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y = y.long()

        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(y.cpu(), preds.cpu())

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.logged_metrics['val_loss_epoch']
        accuracy = self.trainer.logged_metrics['val_acc_epoch']

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True)

        return {'Average Loss:': avg_loss, 'Accuracy:': accuracy}

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y = y.long()
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'test_loss': loss, 'test_preds': preds, 'test_targets': y}

def load_and_preprocess_data(target_size, empty_folder, not_empty_folder):
    X = []
    y = []

    # Load images from the "empty" folder
    for filename in os.listdir(empty_folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(empty_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(0)  # Label 0 for empty

    # Load images from the "not_empty" folder
    for filename in os.listdir(not_empty_folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(not_empty_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(1)  # Label 1 for not empty

    # Convert lists to NumPy arrays for easier handling
    X = np.array(X)
    y = np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

@hydra.main(config_path="configs",config_name="train")
def main(cfg: DictConfig):
    target_size = cfg.data.target_size
    empty = cfg.data.empty_folder
    not_empty = cfg.data.not_empty_folder
    X_train, X_test, y_train, y_test = load_and_preprocess_data(target_size, empty, not_empty)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast()
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)

    # Initialize the Lightning model
    model = ResNet18(num_classes=2, optimizer_cfg=cfg.model.optimizer)
    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='checkpoints/',  # Directory where checkpoints will be saved

        
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='max'
    )

    # Define a TensorBoard logger to log metrics
    logger = TensorBoardLogger('logs', name='simple_cnn')

    # Initialize a PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=300,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger, 
        check_val_every_n_epoch=10,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback,early_stop_callback],

        
    )
    # Train the model
    trainer.fit(model, train_dataloaders=DataLoader(train_dataset, batch_size=8, shuffle=True), val_dataloaders=DataLoader(test_dataset, batch_size=8))

    # Evaluate the model
    trainer.test(model, dataloaders=DataLoader(test_dataset, batch_size=8))
    state_dict = model.state_dict()
    model_name = 'model.pth'
    torch.save(state_dict, model_name)

if __name__ == "__main__":
    main()