import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import HTML
import snntorch.spikeplot as splt


data_path='/Users/fernando/Documents/TFM_DS_Datos_Pytorch/TrainImagesTFM1_2/'
save_dir="/Users/fernando/Documents/TFM_DS_Datos_Pytorch/Transformed"
spike_output_file_path = '/Users/fernando/Documents/TFM_DS_Datos_Pytorch/Spike_rate_Transformed/spike_train_data'
train_data_path = '/Users/fernando/Documents/TFM_DS_Datos_Pytorch/Spike_rate_Transformed/train_data'
labels_path = '/Users/fernando/Documents/TFM_DS_Datos_Pytorch/Spike_rate_Transformed/labels'
abs_path = data_path + 'IMG/'


# Function to display an image
def show_image(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    # Normalize the image for displaying if necessary
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

output_file_path = os.path.join(spike_output_file_path,f"spike_data_num_steps_10_gain_0.5.h5")

# Open the HDF5 file
def plot_images(batch, image):
    with h5py.File(output_file_path, 'r') as f:
        # Access the specific dataset
        dataset_name = f'original_data_batch_{batch}' 
        dataset_spike = f'spike_data_batch_{batch}'
        
        
        if dataset_name in f:
            data = f[dataset_name][:]

            # Convert the dataset to a PyTorch tensor
            tensor_data = torch.tensor(data)
            # Select the first image in the dataset
            image_tensor = tensor_data[image]
            # Display the image
            show_image(image_tensor)
            
        
        if dataset_spike in f:
            spike_data_batch = torch.from_numpy(f[dataset_spike][:])
            image_spike_data = spike_data_batch[:, 0, 0]
            print(image_spike_data.size())
            
            ## Display the animation
            fig, ax = plt.subplots()
            anim = splt.animator(image_spike_data, fig, ax)
            plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
            HTML(anim.to_html5_video())
            #anim.save("Rate_encoding/Animaciones/spike_test_rate.mp4") 
            plt.show()        
        

plot_images(12, 0)     
    