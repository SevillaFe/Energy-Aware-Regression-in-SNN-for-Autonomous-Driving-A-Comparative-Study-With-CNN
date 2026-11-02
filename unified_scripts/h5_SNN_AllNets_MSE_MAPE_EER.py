import argparse
import os

import h5py
import pytorch_lightning as pl
import snntorch as snn
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from lightning.pytorch.loggers import TensorBoardLogger
from snntorch import surrogate, utils
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsolutePercentageError


def normalize_model_name(model_name: str) -> str:
    name = model_name.lower()
    alias = {
        "pilonet": "pilotnet",
        "pilotnet": "pilotnet",
        "laksnet": "lasknet",
        "lasknet": "lasknet",
        "mininet": "mininet",
    }
    if name not in alias:
        raise ValueError("model must be one of: pilotnet/pilonet, lasknet/laksnet, mininet")
    return alias[name]


class SNNDrivingNet(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "pilotnet",
        learning_rate: float = 1e-4,
        beta: float = 0.9,
        threshold: float = 0.5,
        momentum: float = 0.5492389157972004,
    ):
        super().__init__()
        self.model_name = normalize_model_name(model_name)
        self.learning_rate = learning_rate
        self.beta = beta
        self.threshold = threshold
        self.momentum = momentum

        self.mape_metric = MeanAbsolutePercentageError()
        self.tracker_eta = EmissionsTracker()
        self.val_energy_cost = 0
        self.loss_fn = nn.MSELoss()

        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.net = self._build_net(spike_grad)

    def _build_net(self, spike_grad):
        if self.model_name == "pilotnet":
            return nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(24, 36, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(36, 48, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(48, 64, kernel_size=3),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(64, 64, kernel_size=3),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Flatten(),
                nn.Linear(1152, 100),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(100, 50),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(50, 10),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(10, 1),
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    learn_beta=True,
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                    reset_mechanism="none",
                ),
            )

        if self.model_name == "lasknet":
            return nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(24, 36, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(36, 48, kernel_size=5, stride=2),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Conv2d(48, 64, kernel_size=3),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Flatten(),
                nn.Linear(3840, 120),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(120, 20),
                snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
                nn.Linear(20, 1),
                snn.Leaky(
                    beta=self.beta,
                    threshold=self.threshold,
                    learn_beta=True,
                    spike_grad=spike_grad,
                    init_hidden=True,
                    output=True,
                    reset_mechanism="none",
                ),
            )

        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=0),
            snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            snn.Leaky(beta=self.beta, threshold=self.threshold, learn_beta=True, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(32, 1),
            snn.Leaky(
                beta=self.beta,
                threshold=self.threshold,
                learn_beta=True,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
                reset_mechanism="none",
            ),
        )

    def forward(self, x):
        utils.reset(self.net)
        if x.dim() < 5:
            x = x.unsqueeze(1)

        mem_out = None
        for step in range(x.size(1)):
            step_data = x[:, step, :, :, :]
            _, mem_out = self.net(step_data)
        return mem_out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        labels = labels.unsqueeze(1)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        labels = labels.unsqueeze(1)
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
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)


class H5SpikingDataset(Dataset):
    def __init__(self, h5_file_path: str, split: str = "train"):
        self.h5_file_path = h5_file_path
        self.split = split

        self.h5_file = h5py.File(self.h5_file_path, "r")
        if self.split == "train":
            self.images = self.h5_file["train_images"]
            self.labels = self.h5_file["train_labels"]
        else:
            self.images = self.h5_file["val_images"]
            self.labels = self.h5_file["val_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def close(self):
        self.h5_file.close()


class H5DataModule(pl.LightningDataModule):
    def __init__(self, h5_file_path: str, batch_size: int = 64):
        super().__init__()
        self.h5_file_path = h5_file_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = H5SpikingDataset(self.h5_file_path, split="train")
        self.val_dataset = H5SpikingDataset(self.h5_file_path, split="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage=None):
        self.train_dataset.close()
        self.val_dataset.close()


def resolve_h5_path(args) -> tuple[str, str]:
    if args.h5_file_path:
        file_name = os.path.basename(args.h5_file_path)
        return args.h5_file_path, file_name

    if args.encoding == "rate":
        if args.num_steps is None or args.gain is None:
            raise ValueError("For rate encoding, provide --num-steps and --gain")
        file_name = f"encoded_dataset_rate_numsteps_{args.num_steps}_gain_{args.gain}.h5"
        return os.path.join(args.h5_dir_rate, file_name), file_name

    if args.encoding == "latency":
        if args.num_steps is None or args.tau is None:
            raise ValueError("For latency encoding, provide --num-steps and --tau")
        file_name = f"encoded_dataset_latency_numsteps_{args.num_steps}_tau_{args.tau}.h5"
        return os.path.join(args.h5_dir_latency, file_name), file_name

    if args.threshold is None:
        raise ValueError("For delta encoding, provide --threshold")
    file_name = f"encoded_dataset_delta_threshold_{args.threshold}.h5"
    return os.path.join(args.h5_dir_delta, file_name), file_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train H5 SNN model (PilotNet/LaskNet/MiniNet) with one unified script")

    parser.add_argument("--model", default="pilotnet", choices=["pilotnet", "pilonet", "lasknet", "laksnet", "mininet"])
    parser.add_argument("--encoding", default="latency", choices=["rate", "latency", "delta"])

    parser.add_argument("--h5-file-path", default=None, help="Direct path to H5 file. If set, encoding-derived paths are ignored.")
    parser.add_argument("--h5-dir-rate", default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Encoded_Datasets")
    parser.add_argument("--h5-dir-latency", default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Encoded_Datasets")
    parser.add_argument("--h5-dir-delta", default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Encoded_Datasets")

    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.25)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.5492389157972004)

    parser.add_argument("--tb-root", default="tb_logs")
    parser.add_argument(
        "--emissions-root",
        default="/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images",
        help="Root folder that contains Rate/Latency/Delta emissions folders",
    )
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--accelerator", default="mps")

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.model)

    h5_file_path, h5_file_name = resolve_h5_path(args)

    encoding_cap = args.encoding.capitalize()
    model_cap = model_name.capitalize()

    emissions_dir = os.path.join(args.emissions_root, encoding_cap, "Emissions", "Emissions_model", model_cap)
    os.makedirs(emissions_dir, exist_ok=True)

    data_module = H5DataModule(h5_file_path=h5_file_path, batch_size=args.batch_size)
    model = SNNDrivingNet(
        model_name=model_name,
        learning_rate=args.learning_rate,
        beta=args.beta,
        threshold=0.5,
        momentum=args.momentum,
    )

    logger_name = f"SNN_{model_cap}_{h5_file_name.replace('.h5', '')}"
    logger = TensorBoardLogger(args.tb_root, name=logger_name)

    tracker_training = EmissionsTracker(
        project_name="encoding_" + logger_name,
        output_dir=emissions_dir,
        output_file="training_emissions_" + logger_name + ".csv",
    )
    tracker_training.start()

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices="auto",
        precision=args.precision,
        logger=logger,
        log_every_n_steps=50,
    )
    trainer.fit(model, data_module)

    tracker_training.stop()


if __name__ == "__main__":
    main()

# Example:
# python 00_h5_SNN_AllNets_MSE_MAPE_EER.py --model laksnet --encoding latency --num-steps 25 --tau 5.0 --max-epochs 20
