import scipy as sp
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchsummary import summary
import torch.optim as optim
import pytorch_lightning as pl
import optuna
from snntorch import utils

class RLeakyModel(nn.Module):
    def __init__(self):
        super(RLeakyModel, self).__init__()
        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.9
        threshold=0.5 
        learn_beta = True
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(48, 64, kernel_size=3),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(64, 64, kernel_size=3),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(1152, 100),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Linear(100, 50),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Linear(50, 10),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Linear(10, 1),
            snn.Leaky(beta=beta, threshold=threshold, learn_beta=learn_beta, spike_grad=self.spike_grad,
              init_hidden=True, output=True, reset_mechanism="none")
)
        

    def forward(self, x):
        print(x.size())
        utils.reset(self.net)
        mem_rec = []  # Record membrane potentials for analysis if needed
        if x.dim() < 5: # Add a dummy time dimension to the input if it's missing (case delta encoding)
            x = x.unsqueeze(1)
        for step in range(x.size(1)):
            step_data = x[:, step, :, :, :]
            print(x.size())
            spk_out, mem_out = self.net(step_data)
            #print("mem_out dtype:", mem_out.dtype)
            mem_rec.append(mem_out)
        return mem_out  # Return final time step output
    

# Instantiate the model
rlif = RLeakyModel()


# Print the model summary
summary(rlif, input_size=( 1, 1, 66, 200))

# Example input tensor with batch size of 2
dummy_input = torch.zeros(2, 1, 1, 66, 200)
output = rlif(dummy_input)
print(output.shape)