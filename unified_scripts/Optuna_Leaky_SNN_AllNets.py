import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch.multiprocessing as mp
import gc
from torch import mps
import h5py
from torch.utils.data import DataLoader, Dataset
import os
from codecarbon import EmissionsTracker
from torchmetrics import MeanAbsolutePercentageError
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



class SNNDrivingNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25), threshold=0.5,
                 optimizer_name='Adam', momentum=0.9, model_name='pilotnet'):
        
        
        super().__init__()
        self.learning_rate = learning_rate
        self.beta = beta
        self.threshold = threshold
        self.optimizer_name = optimizer_name
        self.momentum = momentum
        self.model_name = model_name.lower()
        self.mape_metric = MeanAbsolutePercentageError()
        self.tracker_eta = EmissionsTracker()
        self.val_energy_cost = 0
        
        # Define the loss function and evaluation metric
        self.loss_fn = nn.MSELoss()
        
        self.net = self._build_network(spike_grad)
        
        #self.init_weights()

    def _build_network(self, spike_grad):
        if self.model_name == 'pilotnet':
            return self._build_pilotnet(spike_grad)
        if self.model_name == 'lasknet':
            return self._build_lasknet(spike_grad)
        if self.model_name == 'mininet':
            return self._build_mininet(spike_grad)
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def _build_pilotnet(self, spike_grad):
        return nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            snn.Leaky(
                beta=torch.ones(98) * self.beta,
                threshold=torch.ones(98) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            snn.Leaky(
                beta=torch.ones(47) * self.beta,
                threshold=torch.ones(47) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            snn.Leaky(
                beta=torch.ones(22) * self.beta,
                threshold=torch.ones(22) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Conv2d(48, 64, kernel_size=3),
            snn.Leaky(
                beta=torch.ones(20) * self.beta,
                threshold=torch.ones(20) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Conv2d(64, 64, kernel_size=3),
            snn.Leaky(
                beta=torch.ones(18) * self.beta,
                threshold=torch.ones(18) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Flatten(),
            nn.Linear(1152, 100),
            snn.Leaky(
                beta=torch.ones(100) * self.beta,
                threshold=torch.ones(100) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Linear(100, 50),
            snn.Leaky(
                beta=torch.ones(50) * self.beta,
                threshold=torch.ones(50) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Linear(50, 10),
            snn.Leaky(
                beta=torch.ones(10) * self.beta,
                threshold=torch.ones(10) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                init_hidden=True,
            ),
            nn.Linear(10, 1),
            snn.Leaky(
                beta=torch.ones(1) * self.beta,
                threshold=torch.ones(1) * self.threshold,
                learn_beta=True,
                learn_threshold=True,
                spike_grad=spike_grad,
                reset_mechanism="none",
                init_hidden=True,
                output=True,
            ),
        )

    def _build_lasknet(self, spike_grad):
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

    def _build_mininet(self, spike_grad):
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

    #def init_weights(self):
    #    for m in self.modules():
    #        if isinstance(m, nn.Conv2d):
    #            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
    #            if m.bias is not None:
    #                nn.init.constant_(m.bias, 0.01)
    #        elif isinstance(m, nn.Linear):
    #            nn.init.xavier_normal_(m.weight)  # Xavier initialization
    #            if m.bias is not None:
    #                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        utils.reset(self.net)
        mem_rec = []  # Record membrane potentials for analysis if needed
        if x.dim() < 5: # Add a dummy time dimension to the input if it's missing (case delta encoding)
            x = x.unsqueeze(1)
        for step in range(x.size(1)):
            step_data = x[:, step, :, :, :]
            _, mem_out = self.net(step_data)
            #print("mem_out dtype:", mem_out.dtype)
            mem_rec.append(mem_out)
        return mem_out  # Return final time step output

    #def _get_conv_output_size(self):
    #    """Helper function to calculate the size of the output from the conv layers."""
    #    # Create a version of conv_layers with batch normalization layers
    #    conv_only_layers = nn.Sequential(
    #            nn.Conv2d(1, 24, kernel_size=5, stride=2),
    #            nn.Conv2d(24, 36, kernel_size=5, stride=2),
    #            nn.Conv2d(36, 48, kernel_size=5, stride=2),
    #            nn.Conv2d(48, 64, kernel_size=3),
    #            nn.Conv2d(64, 64, kernel_size=3)
    #        )
     #   
     #   # Pass a dummy input through the conv_only_layers to calculate the output size
     #   dummy_input = torch.zeros(1, 1, 66, 200)  # Adjust the size according to your input dimensions
     #   output = conv_only_layers(dummy_input)
     #   return output.view(1, -1).size(1)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        #print("Predictions dtype:", predictions.dtype)
        #print("Labels dtype:", labels.dtype)
        labels = labels.unsqueeze(1)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        labels = labels.unsqueeze(1)
        val_loss = self.loss_fn(predictions, labels)
        val_mape = self.mape_metric(predictions, labels) # new
        self.log("val_loss", val_loss)
        self.log("val_mape", val_mape, on_epoch=True) # new
        
        # New
        # Logging EER using val_loss and cumulative energy cost at epoch end
        if self.val_energy_cost > 0:
            eer = (1 / val_loss.item()) / self.val_energy_cost
            self.log("eer", eer, on_epoch=True)
        
        return val_loss

    
    # New
    def on_train_epoch_start(self):
        # Start tracking emissions for this epoch
        self.tracker_eta.start()
    
    #def on_fit_start(self):
    #    self.init_weights()
    #    # Check if all parameters have requires_grad=True
    #    for name, param in self.named_parameters():
    #        if not param.requires_grad:
    #            print(f"{name} does not require gradients!")
                

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999))
        elif self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=self.momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    #def on_fit_start(self):
    #    self.init_weights()

    #def print_grads(self):
    #    found_gradients = False  # To track if we find any gradients
    #    for name, param in self.named_parameters():
    #        if param.grad is not None:
    #            found_gradients = True
    #            print(f"{name}.grad: {param.grad.norm().item()}")
    #    if not found_gradients:
    #        print("No gradients found")

class H5SpikingDataset(Dataset):
    def __init__(self, h5_file_path, split="train"):
        self.h5_file_path = h5_file_path
        self.split = split

        # Open the H5 file in read mode
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        if self.split == "train":
            self.images = self.h5_file['train_images']
            self.labels = self.h5_file['train_labels']
        elif self.split == "val":
            self.images = self.h5_file['val_images']
            self.labels = self.h5_file['val_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve the encoded image and label
        image = torch.tensor(self.images[idx])  # Convert to PyTorch tensor
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

    def close(self):
        self.h5_file.close()
        

# DataModule with loading logic based on the custom dataset
class H5PilotNetDataModule(pl.LightningDataModule):
    def __init__(self, h5_file_path, batch_size=64):
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
        # Close the H5 file after training/validation
        self.train_dataset.close()
        self.val_dataset.close()


def objective(trial, fixed_model_name=None, fixed_encoding=None):
    # Define the hyperparameters to tune
    valid_model_names = ['pilotnet', 'lasknet', 'mininet']
    if fixed_model_name is None:
        model_name = trial.suggest_categorical('model_name', valid_model_names)
    else:
        model_name = fixed_model_name.lower()
        if model_name not in valid_model_names:
            raise ValueError(
                f"Invalid fixed_model_name '{fixed_model_name}'. Use one of {valid_model_names} or None."
            )
    beta = trial.suggest_float('beta', 0.01, 0.5)
    threshold = trial.suggest_float('threshold', 0.01, 1.0)
    lr = trial.suggest_float('lr', 1e-5, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD'])
    valid_encodings = ['rate', 'latency', 'delta']
    if fixed_encoding is None:
        encoding = trial.suggest_categorical('encoding', valid_encodings)
    else:
        encoding = fixed_encoding.lower()
        if encoding not in valid_encodings:
            raise ValueError(
                f"Invalid fixed_encoding '{fixed_encoding}'. Use one of {valid_encodings} or None."
            )

    # Tune momentum only for SGD.
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
    else:
        momentum = 0.0

    
    # Define the path to the encoded dataset 
    h5_dir_rate = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Encoded_Datasets'
    h5_file_rate = 'encoded_dataset_rate_numsteps_25_gain_1.0.h5'
    h5_file_path_rate = os.path.join(h5_dir_rate, h5_file_rate)
    
    h5_dir_latency = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Encoded_Datasets'
    h5_file_latency = 'encoded_dataset_latency_numsteps_25_tau_1.0.h5'
    h5_file_path_latency = os.path.join(h5_dir_latency, h5_file_latency)
    
    h5_dir_delta = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Encoded_Datasets'
    h5_file_delta = 'encoded_dataset_delta_threshold_1.0.h5'
    h5_file_path_delta = os.path.join(h5_dir_delta, h5_file_delta)

    if encoding == 'rate':
        selected_h5_file_path = h5_file_path_rate
    elif encoding == 'latency':
        selected_h5_file_path = h5_file_path_latency
    else:
        selected_h5_file_path = h5_file_path_delta
    
    # Create the model with the suggested hyperparameters
    model = SNNDrivingNet(
        learning_rate=lr,
        beta=beta,
        threshold=threshold,
        optimizer_name=optimizer_name,
        momentum=momentum,
        model_name=model_name,
    )
    
    # Initialize the DataModule
    data_module = H5PilotNetDataModule(h5_file_path=selected_h5_file_path, batch_size=64)
    
    # Add a pruning callback
    #pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_mse')
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    
    # Train the model
    trainer = pl.Trainer(max_epochs=10, callbacks=[early_stopping], accelerator= 'mps', devices='auto', precision='16-mixed')
    trainer.fit(model, data_module)
        
    # Validate the model
    val_result = trainer.validate(model, data_module)
    
    # Extract the validation loss
    val_loss = val_result[0]['val_loss']
    
    # Clean up
    del model
    del data_module
    gc.collect()
    mps.empty_cache()
    
    # Return the validation loss
    #return trainer.callback_metrics['val_mse'].item()
    return val_loss

if __name__ == '__main__':
    # Set the seed for reproducibility
    base_seed = 42
    pl.seed_everything(base_seed)

    # Set this to 'pilotnet', 'lasknet', or 'mininet' to lock a single architecture per run.
    # Keep as None to let Optuna choose the architecture.
    fixed_model_name = None

    # Set this to 'rate', 'latency', or 'delta' to lock a single encoding per run.
    # Keep as None to let Optuna choose the encoding.
    fixed_encoding = None
    
    mp.set_start_method('spawn', force=True)  # Use 'spawn' method for multiprocessing
    
    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler())
    study.optimize(
        lambda trial: objective(
            trial,
            fixed_model_name=fixed_model_name,
            fixed_encoding=fixed_encoding,
        ),
        n_trials=25,
        n_jobs=1,
    )

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save the study results
    if fixed_model_name is None and fixed_encoding is None:
        study_name = "optuna_study_mixed_models_encodings"
    elif fixed_model_name is not None and fixed_encoding is None:
        study_name = f"optuna_study_{fixed_model_name.lower()}_encodings"
    elif fixed_model_name is None and fixed_encoding is not None:
        study_name = f"optuna_study_models_{fixed_encoding.lower()}"
    else:
        study_name = f"optuna_study_{fixed_model_name.lower()}_{fixed_encoding.lower()}"
    
    study_dir = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Optuna"
    os.makedirs(study_dir, exist_ok=True)

    study_path = os.path.join(study_dir, study_name + ".pkl")
    study.trials_dataframe().to_pickle(study_path)
    
    # Plot the optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig_2 = optuna.visualization.plot_slice(study, params = ['model_name', 'beta', 'threshold', 'lr', 'optimizer_name', 'momentum', 'encoding'])
    fig_3 = optuna.visualization.plot_param_importances(study)
    
    # Save the plot to a file and display it
    plot_path = os.path.join(study_dir, study_name + "_history.png")
    plot_path_2 = os.path.join(study_dir, study_name + "_slice.png")
    plot_path_3 = os.path.join(study_dir, study_name + "_importances.png")
    fig.write_image(plot_path)
    fig_2.write_image(plot_path_2)
    fig_3.write_image(plot_path_3)
    
        
        
        