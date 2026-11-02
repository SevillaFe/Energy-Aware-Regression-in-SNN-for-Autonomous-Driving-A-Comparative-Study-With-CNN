import h5py
from sympy import N
import torch
import numpy as np
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML


ENCODED_PATH = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/'

RATE_ENCODED_PATH_TRAIN =  ENCODED_PATH + 'Rate/Train/train_encoded_data.h5'
LATENCY_ENCODED_PATH_TRAIN =  ENCODED_PATH + 'Latency/Train/train_encoded_data.h5'
DELTA_ENCODED_PATH_TRAIN =  ENCODED_PATH + 'Delta/Train/train_encoded_data.h5'
NO_ENCODED_PATH_TRAIN =  ENCODED_PATH + 'No_Encoded/Train/train_encoded_data.h5'

print("all imported")

def list_hdf5_groups(h5_file):
    with h5py.File(h5_file, 'r') as f:
        def print_attrs(name, obj):
            print(name)
        f.visititems(print_attrs)

def plot_images(h5_file_path, num_images=5, delta = False):
    with h5py.File(h5_file_path, 'r') as f:
        # Check available datasets
        print("Datasets in file:", list(f.keys()))

        # Access training images and labels
        train_images = f['train_images']
        train_labels = f['train_labels']
        print("Train images shape:", train_images.shape)
        print("Train labels shape:", train_labels.shape)

        # Access validation images and labels
        val_images = f['val_images']
        val_labels = f['val_labels']
        print("Validation images shape:", val_images.shape)
        print("Validation labels shape:", val_labels.shape)
        
        # Display the specified number of images and their steering angles
        
        if delta:
            for i in range(num_images):
                img = train_images[i]
                angle = train_labels[i]
                print(f"Steering angle: {angle}")
                print(f"Image shape: {img.shape}")
                
                img_tensor = torch.from_numpy(img) #.unsqueeze(1) # Add a single-channel dimension
                print(f"img_tensor shape: {img_tensor.shape}")
                #img_tensor[0]
                first_image = img_tensor.squeeze(0) # Remove the single-channel dimension
                #first_image = img_tensor[0]#.squeeze(0) # Remove the single-channel dimension

                #print(first_image.shape)
                plt.figure(figsize=(10, 5))
                plt.imshow(first_image, cmap='gray') 
                plt.colorbar()
                plt.show()
                
        else:
            for i in range(num_images):
                img = train_images[i]
                angle = train_labels[i]
                print(f"Steering angle: {angle} and image shape: {img.shape}")
                
                img_tensor = torch.from_numpy(img).squeeze()  # Remove the single-channel dimension if present
                fig, ax = plt.subplots()
                print(f"img_tensor shape: {img_tensor.shape}")
                anim = splt.animator(img_tensor, fig, ax)
                plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
                HTML(anim.to_html5_video())
                anim.save(f"/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Animations/spike_image_{i}.mp4")
                plt.show()


#plt.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
# Example usage
h5_file_path = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Encoded_Datasets/encoded_dataset_rate_numsteps30_gain1.5.h5'

print("Available groups and datasets in the HDF5 file:")
list_hdf5_groups(h5_file_path)

#exit()
group_name = 'num_steps_30_tau_1.0' #'num_steps_20_gain_0.5' #'threshold_0.5'  # Example group name, change as needed
#group_name = 'num_steps_30_tau_1.0' #'num_steps_15_gain_1.0'

plot_images(h5_file_path, num_images=5, delta = False)