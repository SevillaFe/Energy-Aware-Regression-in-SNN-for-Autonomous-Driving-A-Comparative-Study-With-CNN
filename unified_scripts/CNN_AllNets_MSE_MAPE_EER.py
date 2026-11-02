import argparse
import os
import random

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image
from torchmetrics import MeanAbsolutePercentageError
from torch.utils.data import DataLoader, Dataset


class CNNDrivingNet(pl.LightningModule):
    def __init__(self, model_name="pilotnet", learning_rate=1e-4):
        super().__init__()
        self.model_name = model_name.lower()
        self.learning_rate = learning_rate
        self.mape_metric = MeanAbsolutePercentageError()
        self.tracker_eta = EmissionsTracker()
        self.loss_fn = nn.MSELoss()
        self.val_energy_cost = 0

        self.model = self._build_model()

    def _build_model(self):
        if self.model_name == "pilotnet":
            return nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1152, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

        if self.model_name == "laksnet":
            return nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3840, 120),
                nn.ReLU(),
                nn.Linear(120, 20),
                nn.ReLU(),
                nn.Linear(20, 1),
            )

        if self.model_name == "mininet":
            return nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        raise ValueError("Unsupported model_name. Use one of: pilotnet, laksnet, mininet")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images).squeeze(-1)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images).squeeze(-1)
        val_loss = self.loss_fn(predictions, labels)
        val_mape = self.mape_metric(predictions, labels)
        self.log("val_loss", val_loss)
        self.log("val_mape", val_mape, on_epoch=True)

        if self.val_energy_cost > 0:
            eer = (1 / val_loss.item()) / self.val_energy_cost
            self.log("eer", eer, on_epoch=True)

        return val_loss

    def on_train_epoch_start(self):
        self.tracker_eta.start()

    def on_train_epoch_end(self, outputs=None):
        epoch_energy = torch.tensor(self.tracker_eta.stop(), dtype=torch.float32)
        self.val_energy_cost += epoch_energy
        self.log("energy_cost_per_epoch", epoch_energy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class SteeringAngleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, steering_correction=0.2):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.steering_correction = steering_correction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_entry = self.data.iloc[idx]

        camera_choice = random.choice(["left", "center", "right"])
        if camera_choice == "left":
            img_path = os.path.join(self.img_dir, data_entry.iloc[0].split("/")[-1])
            steering_angle = data_entry.iloc[3] + self.steering_correction
        elif camera_choice == "right":
            img_path = os.path.join(self.img_dir, data_entry.iloc[2].split("/")[-1])
            steering_angle = data_entry.iloc[3] - self.steering_correction
        else:
            img_path = os.path.join(self.img_dir, data_entry.iloc[1].split("/")[-1])
            steering_angle = data_entry.iloc[3]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(steering_angle, dtype=torch.float32)
        return image, label


class DrivingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, csv_file, batch_size=64, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((66, 200), scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((66, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def setup(self, stage=None):
        full_dataset = SteeringAngleDataset(
            csv_file=self.csv_file,
            img_dir=os.path.join(self.data_dir, "IMG"),
            transform=self.train_transform,
            steering_correction=0.2,
        )

        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN steering model (PilotNet/LaksNet/MiniNet)")
    parser.add_argument("--model", default="pilotnet", choices=["pilotnet", "laksnet", "mininet"])
    parser.add_argument("--data-dir", default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images/")
    parser.add_argument("--csv-file", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--tb-root", default="tb_logs")
    parser.add_argument(
        "--emissions-root",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/No_encoded_cnn/Emissions/Emissions_model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_file = args.csv_file or os.path.join(args.data_dir, "driving_log.csv")

    model_dir_name = {
        "pilotnet": "PilotNet",
        "laksnet": "LaksNet",
        "mininet": "MiniNet",
    }[args.model]

    output_dir_cnn = os.path.join(args.emissions_root, model_dir_name)
    os.makedirs(output_dir_cnn, exist_ok=True)

    data_module = DrivingDataModule(
        data_dir=args.data_dir,
        csv_file=csv_file,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
    )
    model = CNNDrivingNet(model_name=args.model, learning_rate=args.learning_rate)

    logger_name = f"CNN_{model_dir_name}"
    logger = TensorBoardLogger(args.tb_root, name=logger_name)

    tracker_training = EmissionsTracker(
        project_name="encoding_" + logger_name,
        output_dir=output_dir_cnn,
        output_file="training_emissions_" + logger_name + ".csv",
    )
    tracker_training.start()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="mps",
        devices="auto",
        precision="16-mixed",
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, data_module)

    tracker_training.stop()


if __name__ == "__main__":
    main()

# Example run:
# python 00_CNN_AllNets_MSE_MAPE_EER.py --model pilotnet --epochs 20 --batch-size 64 --learning-rate 1e-4 --data-dir /Users/fernando/Documents/Doctorado/Udacity_Datset/Original_Images/ --split-ratio 0.8 --tb-root tb_logs --emissions-root /Users/fernando/Documents/Doctorado/Udacity_Datset/No_encoded_cnn/Emissions/Emissions_model
