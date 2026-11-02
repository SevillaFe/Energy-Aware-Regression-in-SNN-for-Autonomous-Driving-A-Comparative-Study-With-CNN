from math import e
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from imgaug import augmenters as iaa
from PIL import Image
import torch
import h5py
import snntorch.spikegen as spikegen
from torch.utils.data import DataLoader
from torch import mps
import re



from codecarbon import EmissionsTracker

# declaration of image parameters
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

"""
* @brief Function to load the images from the path 
* @param data direcrtory 
* @param File path of the images
* @return The image file
"""
def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

"""
* @brief Function to crop the imeages to the required shape
* @param The image to crop
* @return The cropped image
"""
def crop(image):
   
   # Crop the image (removing the sky at the top and the car front at the bottom)
    
    return image[60:-25, :, :] # remove the sky and the car front

"""
* @brief Resize the image to the input shape used by the network model
* @param Image file to Resize
* @return The Resized image for the network
""" 
def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

"""
* @brief Fuinction to convert the color space of the image from RGB to YUV.
* @param The image to change the colorspace.
* @return The converted image. 
"""
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

"""
* @brief Fuinction to convert the image from RGB to grayscale.
* @param The image to change the colorspace.
* @return The converted image. 
"""
def rgb2grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

"""
* @brief Function to run the preprocess of the images.
* @param The images from the dataset
* @return The preprocessed images.
"""
def preprocess(image):
    image = crop(image)
    #image = resize(image)
    #image = rgb2yuv(image)
    return image

"""
* @brief Function to randomly choose the center, left and right images 
* to adjust the steering angles.
* @param Steering angles that corresponds to the images.
* @return The new adjusted steering angles.
"""
def choose_image(steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return "img_left_pth", float(steering_angle) + 0.2
    elif choice == 1:
        return "img_right_pth", float(steering_angle) - 0.2
    return "img_center_pth", float(steering_angle)

"""
* @brief Function to randomly flip the left and right images and adjust
* the steering angles.
* @param The images from the left or right dataset.
* @param The steering angle of the corresponding images.
* @return the flipped images and their corresponding steering angles.
"""
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def convert_to_grayscale(image):
    if image is None or image.size == 0:
        raise ValueError("Empty image provided for grayscale conversion.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

"""
* @brief Fuinction to randomly translate the image vertically and horizontally.
* @param The image to translate.
* @param The steering angle of the selected image.
* @param Range of x translation.
* @param Range of y translation.
* @return The image and its corresponding steering angle
"""
def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

"""
* @brief Randomly generates shadows in the image.
* @param The image to add shadow on.
* @return The image with added shadows.
""" 
def random_shadow(image):
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para trabajar en el espacio HLS
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Ensure the image is a NumPy array
    #if isinstance(image, Image.Image):
    #    image = np.array(image)
    
    IMAGE_HEIGHT, IMAGE_WIDTH = image.shape[:2]
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1)) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    shadowed_image = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    #shadowed_image = cv2.cvtColor(shadowed_image, cv2.COLOR_BGR2GRAY)  # Convertir de vuelta a escala de grises
    # Convert back to PIL Image if needed
    #shadowed_image = Image.fromarray(shadowed_image)

    return shadowed_image

"""
* @brief Function to adjust the brightness of the images.
* @param The image to process
* @return The brightness equalized image.
"""
def random_brightness(image):
    #if isinstance(image, Image.Image):
    #    image = np.array(image)
    brightness_augmenter = iaa.Multiply((0.2, 1.6))
    brightened_image = brightness_augmenter.augment_image(image)
    # Convert back to PIL Image if needed
    #brightened_image = Image.fromarray(brightened_image)
    return brightened_image

def apply_pan(image):
    pan_transformer = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    panned_image = pan_transformer.augment_image(image)
    return panned_image

def apply_zoom(image):
    zoom_transformer = iaa.Affine(scale=(1, 1.5))
    zoomed_image = zoom_transformer.augment_image(image)
    return zoomed_image

def apply_image_brighten(image):
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para trabajar en el espacio HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = 0.25 + np.random.uniform()
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
    brightened_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    brightened_image = cv2.cvtColor(brightened_image, cv2.COLOR_BGR2GRAY)  # Convertir de vuelta a escala de grises
    return brightened_image

"""
* @brief Function to augment the image and trhe steering angle of the center image
* @param The data directory containing thje images.
* @param The center, left and right images and their corresponding steering angles.
* @param The x and y range of the image augmentation allowd(set to a default value
* unless specified on execution)
* @return Returns the output from all image augmentation process.
"""
def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle

def random_augment(image):
    # Convert the image to a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC format   
    
    # Perform random augmentations
    if np.random.rand() < 0.5:
        #print("Pan")
        image = apply_pan(image)
    if np.random.rand() < 0.5:
        #print("Zoom")
        image = apply_zoom(image)
    if np.random.rand() < 0.5:
        #print("Brightness")
        image = random_brightness(image)
    if np.random.rand() < 0.5:
        #print("Brighten")
        image = apply_image_brighten(image)
    if np.random.rand() < 0.5:
        #print("Shadow")
        image = random_shadow(image)
    
    # Convert the image back to a PIL image
    image = Image.fromarray(image)
    
    return image

"""
* @brief Function to Generate training image give image paths and associated steering angles
* @param The directory path where the data is stored.
* @param The image paths for each image.
* @param The steering angles for the images.
* @param The variable containing the size of the required batch size.
* @param Binary value of the status of the training process.
* @return The images and steering angles.
"""
def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


#########
# Methods to encode the data with snnTorch 
#########

def rate_encoding(data_loader, num_steps_range, gain_range, path_to_save, device='cpu'):
    with h5py.File(path_to_save, 'w') as h5f:
        for num_steps in num_steps_range:
            for gain in gain_range:
                print(f"Encoding with num_steps={num_steps} and gain={gain}")
                group = h5f.create_group(f'num_steps_{num_steps}_gain_{gain}')
                spike_images = []
                steering_angles_list = []
                
                tracker = EmissionsTracker(
                    project_name="Experiment_" + str(num_steps)+ '_' + str(gain),
                    output_dir = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Emissions',
                    output_file= 'emissions_' + str(num_steps)+ '_' + str(gain) + '.csv')
                
                tracker.start()

                for batch_idx, (images, steering_angles) in enumerate(data_loader):
                    images = images.to(torch.float32).to(device)
                    steering_angles = steering_angles.to(torch.float32).to(device)
                    

                    for img, angle in zip(images, steering_angles):  # Use zip to iterate over both images and angles
                        spike_data = spikegen.rate(img, gain=gain, num_steps=num_steps)
                        spike_images.append(spike_data.cpu().numpy())  # Store the spike-encoded image
                        steering_angles_list.append(angle.cpu().numpy())  # Store the corresponding steering angle

                    #spike_data = spikegen.rate(images, gain=gain, num_steps=num_steps)
                    
                    # Convert the batch to CPU and NumPy
                    #spike_images.append(spike_data.cpu().numpy())
                    #steering_angles_list.append(steering_angles.cpu().numpy())
                
                #spike_images = np.concatenate(spike_images, axis=0)  # Concatenate all batches
                #steering_angles_list = np.concatenate(steering_angles_list, axis=0)

                group.create_dataset('spike_images', data=spike_images, compression="gzip", compression_opts=5) #compression="gzip", compression_opts=5
                group.create_dataset('steering_angles', data=steering_angles_list, compression="gzip", compression_opts=5)
                
                emissions: float = tracker.stop()
                
                # Clear lists to free memory
                #spike_images.clear()
                #steering_angles_list.clear()
                
                # Clear GPU memory if needed
                mps.empty_cache()

 
def latency_encoding(data_loader, num_steps_range, tau_range, threshold, path_to_save, device='cpu'):
    with h5py.File(path_to_save, 'w') as h5f:
        for num_steps in num_steps_range:
            for tau in tau_range:
                print(f"Encoding with num_steps={num_steps} and tau={tau}")
                group = h5f.create_group(f'num_steps_{num_steps}_tau_{tau}')
                spike_images = []
                steering_angles_list = []
                
                tracker = EmissionsTracker(
                    project_name="Experiment_" + str(num_steps)+ '_' + str(tau),
                    output_dir = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions',
                    output_file= 'emissions_' + str(num_steps)+ '_' + str(tau) + '.csv')
                tracker.start()

                for batch_idx, (images, steering_angles) in enumerate(data_loader):
                    images = images.to(torch.float32).to(device)
                    steering_angles = steering_angles.to(torch.float32).to(device)

                    #for img in images:
                    for img, angle in zip(images, steering_angles):
                        #img_normalized = (img + 1) / 2
                        #img_normalized = img * 255
                        #img_normalized = torch.clamp(img_normalized, 0, 1)
                        spike_data = spikegen.latency(img, num_steps=num_steps, tau=tau, threshold=threshold,
                              clip=True, normalize=True, linear=True)
                        spike_images.append(spike_data.cpu().numpy())
                        steering_angles_list.append(angle.cpu().numpy())
                    
                    # Apply latency encoding for the entire batch if possible
                    #spike_data_batch = spikegen.latency(images, num_steps=num_steps, tau=tau, threshold=threshold,
                    #                    clip=True, normalize=True, linear=True)

                     # Convert the batch to CPU and NumPy
                    #spike_images.append(spike_data_batch.cpu().numpy())

                    #steering_angles_list.append(steering_angles.cpu().numpy())
                
                #spike_images = np.array(spike_images)
                #steering_angles_list = np.concatenate(steering_angles_list, axis=0)

                group.create_dataset('spike_images', data=spike_images, compression="gzip", compression_opts=5) #compression="gzip", compression_opts=5
                group.create_dataset('steering_angles', data=steering_angles_list, compression="gzip", compression_opts=5)
                
                emissions: float = tracker.stop()
                
                # Clear lists to free memory
                #spike_images.clear()
                #steering_angles_list.clear()
                
                # Clear GPU memory if needed
                mps.empty_cache()       

# Reinfolge der Bilder sich anschauen!!! es kann nach zur Problemen führen!!
def delta_encoding(data_loader, threshold_range, path_to_save, device='cpu'):
    with h5py.File(path_to_save, 'w') as h5f:
        for threshold in threshold_range:
            print(f"Encoding_delta with threshold={threshold}")
            group = h5f.create_group(f'threshold_{threshold}')
            spike_images = []
            steering_angles_list = []
                    
            tracker = EmissionsTracker(
                project_name="Experiment_" + str(threshold),
                output_dir = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Emissions',
                output_file= 'emissions_' + str(threshold) + '.csv')
            tracker.start()

            for batch_idx, (images, steering_angles) in enumerate(data_loader):
                images = images.to(torch.float32).to(device)
                steering_angles = steering_angles.to(torch.float32).to(device)
                prev_image = torch.zeros_like(images[0])  # Assuming the first image in a batch as the initial previous image

                for img in images:
                    delta = img - prev_image

                    spike_data = spikegen.delta(delta , threshold, off_spike=True)
                    spike_images.append(spike_data.cpu().numpy())
                    #spike_images.append(spike_data.squeeze(0).cpu().numpy())
                    prev_image = img

                steering_angles_list.append(steering_angles.cpu().numpy())
            
            spike_images = np.array(spike_images)
            steering_angles_list = np.concatenate(steering_angles_list, axis=0)

            group.create_dataset('spike_images', data=spike_images, compression="gzip", compression_opts=5) #compression="gzip", compression_opts=5
            group.create_dataset('steering_angles', data=steering_angles_list, compression="gzip", compression_opts=5)
            
            emissions: float = tracker.stop()
            mps.empty_cache()        


########
# Methods to create a hd5 file with no encoding
########

def no_encoding(data_loader, path_to_save, device='cpu'):
    with h5py.File(path_to_save, 'w') as h5f:
        
        print(f"No encoding")
        group = h5f.create_group(f'No_encoding')
        images_list = []
        steering_angles_list = []
                
        tracker = EmissionsTracker(
            project_name="Experiment_" + 'No_encoding',
            output_dir = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/No_Encoded/Emissions',
            output_file= 'no_encoding_emissions_' + '.csv')
                
        tracker.start()

        for batch_idx, (images, steering_angles) in enumerate(data_loader):
            images = images.to(torch.float32).to(device)
            steering_angles = steering_angles.to(torch.float32).to(device)
                    

            for img, angle in zip(images, steering_angles):  # Use zip to iterate over both images and angles
                images_list.append(img.cpu().numpy())  # Store the spike-encoded image
                steering_angles_list.append(angle.cpu().numpy())  # Store the corresponding steering angle

        group.create_dataset('No_encode_images', data=images_list, compression="gzip", compression_opts=5) #compression="gzip", compression_opts=5
        group.create_dataset('steering_angles', data=steering_angles_list, compression="gzip", compression_opts=5)
            
        emissions: float = tracker.stop()
                
        # Clear lists to free memory
        #spike_images.clear()
        #steering_angles_list.clear()
                
        # Clear GPU memory if needed
        mps.empty_cache()



#########
# Methods to load hdf5 files and plot images
#########

def list_hdf5_groups(h5_file):
    with h5py.File(h5_file, 'r') as f:
        def print_attrs(name, obj):
            print(name)
        f.visititems(print_attrs)

def get_dataset_names(h5_file):
    with h5py.File(h5_file, 'r') as f:
        dataset_names = list(f.keys())
    return dataset_names

def load_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        delta_images = torch.tensor(f['spike_images'][:])
        angles = torch.tensor(f['steering_angles'][:])
    return delta_images, angles

def create_logger_name(h5_file, datagroup_name):
    dataset_names = get_dataset_names(h5_file)
    # Filter for dataset pairs, assuming naming convention like "spike_images_train"
    relevant_datasets = [name for name in dataset_names if datagroup_name in name]
    # Create a unique name by joining the relevant dataset names
    logger_name = "_".join(relevant_datasets)
    return logger_name

#########
# Methods independent of image transformation or image augmentation 
#########

"""
* @brief Function to load a CSV file and return a specified column as a list
* @param The desired column name.
* @return A list based on a column name
"""
def load_csv_as_list(file_path, column_name):
    # Function to load a CSV file and return a specified column as a list
    df = pd.read_csv(file_path)
    return df[column_name].tolist()

"""
* @brief It takes six different lists, creates a DataFrame with column names as the list names, 
+ and saves it as a CSV file in the same path as the script
* @param The desired variable number of lists.
* @return DataFrame with columns named 'list1', 'list2', etc., and save it as a CSV file in the same directory as the script
"""
def create_and_save_dataframe(df_name,save_path,**kwargs):
    
    # Create a dictionary using function argument names as keys and corresponding lists as values
    data = {key: value for key, value in kwargs.items()}
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Get the current script's directory
    #script_directory = os.path.dirname(os.path.realpath(__file__))

    # Create a CSV file path in the same directory as the script
    csv_file_path = os.path.join(save_path, df_name + '_dataframe_output.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"DataFrame saved successfully at: {csv_file_path}")
    
"""
* @brief It assumes that all CSV files in the specified folder have the same column names. 
    It concatenates them into a single DataFrame, extracts the information from the file names using the provided pattern, 
    and adds new columns ('num_steps' and 'gain'). Finally, it saves the concatenated DataFrame with the new columns 
    to a new CSV file
* @param The path to the .csv files.
* @param The path to store the output file.
* @return Concatenated DataFrame with the new columns to a new CSV file
"""
def concatenate_emissions_csv_files(folder_path, output_file_path, name_file, encode_type):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.startswith(name_file)]

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    for file in csv_files:
        # Read each CSV file
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Extract values from the file name
        parts = file.split('_')
        
        if encode_type == 'Rate':
            num_steps = int(parts[5]) if len(parts) > 1 else None
            gain_part = str(parts[7].split('.csv')[0]) if len(parts) >= 1 else None
            gain = float(gain_part.replace(',', '.'))
            # Add new columns to the DataFrame
            df['num_steps'] = num_steps
            df['gain'] = gain
        elif encode_type == 'Latency':
            num_steps = int(parts[5]) if len(parts) > 1 else None
            tau_part = str(parts[7].split('.csv')[0]) if len(parts) >= 1 else None
            tau = float(tau_part)
            # Add new columns to the DataFrame
            df['num_steps'] = num_steps
            df['tau'] = tau
        elif encode_type == 'Delta':
            #threshold_part = str(parts[1].split('.csv')[0]) if len(parts) >= 1 else None
            threshold_part = str(parts[5].split('.csv')[0]) if len(parts) >= 1 else None
            print(f"Threshold part: {threshold_part}")
            threshold = float(threshold_part)
            # Add new columns to the DataFrame
            df['threshold'] = threshold


        # Concatenate the DataFrame to the overall DataFrame
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        
        # Sort the DataFrame by the timestamp column (replace 'timestamp_column' with the actual column name)
        concatenated_df.sort_values(by='timestamp', inplace=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file_path, index=False)
    

def concatenate_csv_batch_files(folder_path, output_file_path, name_file):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.startswith(name_file)]  #'emissions_'

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    for file in csv_files:
        # Read each CSV file
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Extract values from the file name
        parts = file.split('_')
        num_steps = int(parts[2]) if len(parts) > 1 else None
        gain_part = str(parts[3]) if len(parts) >= 1 else None
        gain = float(gain_part.replace(',', '.'))
        batch = str(parts[4].split('.csv')[0]) if len(parts) >= 1 else None

        # Add new columns to the DataFrame
        df['num_steps'] = num_steps
        df['gain'] = gain
        df['batch'] = batch

        # Concatenate the DataFrame to the overall DataFrame
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        
        # Sort the DataFrame by the timestamp column (replace 'timestamp_column' with the actual column name)
        concatenated_df.sort_values(by='timestamp', inplace=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file_path, index=False)
    
def concatenate_csv_load_batch_files(folder_path, output_file_path, name_file):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.startswith(name_file)]  #'emissions_'

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    for file in csv_files:
        # Read each CSV file
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Extract values from the file name
        parts = file.split('_')
        num_steps = int(parts[3]) if len(parts) > 1 else None
        gain_part = str(parts[4].split('.h5')[0]) if len(parts) >= 1 else None
        gain = float(gain_part)
        batch = str(parts[8].split('.csv')[0]) if len(parts) >= 1 else None

        # Add new columns to the DataFrame
        df['num_steps'] = num_steps
        df['gain'] = gain
        df['batch'] = batch

        # Concatenate the DataFrame to the overall DataFrame
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        
        # Sort the DataFrame by the timestamp column (replace 'timestamp_column' with the actual column name)
        concatenated_df.sort_values(by='timestamp', inplace=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file_path, index=False)
    
def delete_emission_csv_files(folder_path, file_name):
    try:
        # Get the list of files in the specified folder
        files_to_delete = [file for file in os.listdir(folder_path) if file.startswith(file_name)]

        # Iterate through the files and delete each one
        for file_name in files_to_delete:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            #print(f"Deleted: {file_path}")

        print("Deletion complete.")
    except OSError as e:
        print(f"Error: {e}")


def concatenate_training_csv_files(folder_path, output_file_path, name_file):
    # Get a list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.startswith(name_file)]

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_df = pd.DataFrame()

    for file in csv_files:
        # Read each CSV file
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Extract values from the file name
        parts = file.split('_')
        num_steps = int(parts[2]) if len(parts) > 1 else None
        gain = str(parts[3]) if len(parts) >= 1 else None
        

        # Add new columns to the DataFrame
        df['num_steps'] = num_steps
        df['gain'] = gain

        # Concatenate the DataFrame to the overall DataFrame
        concatenated_df = pd.concat([concatenated_df, df], ignore_index=True)
        
        # Sort the DataFrame by the timestamp column (replace 'timestamp_column' with the actual column name)
        concatenated_df.sort_values(by='timestamp', inplace=True)

    # Save the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file_path, index=False)
    
    
    #####
    # Methods to concatenate the training csv files
    #####
    

def concatenate_tensorboard_csv_files(folder_path, output_file_path, join_column, encode_type, model_type, tensorboard_type):
    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    # Initialize the first DataFrame
    concatenated_df = pd.read_csv(os.path.join(folder_path, csv_files[0]))

    # Iterate through the remaining files and merge them
    for file in csv_files[1:]:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        concatenated_df = pd.merge(concatenated_df, df, on=join_column, how='inner')  # Adjust 'how' as needed

    # Drop all columns that start with "Wall time"
    wall_time_columns = [col for col in concatenated_df.columns if col.startswith("Wall time")]
    concatenated_df.drop(columns=wall_time_columns, inplace=True)

    # Create a new column with the average of the three merged columns
    value_columns = [col for col in concatenated_df.columns if col.startswith("Value")]
    concatenated_df['Value_average'] = concatenated_df[value_columns].mean(axis=1)
    concatenated_df['Encode_Method'] = encode_type
    concatenated_df['Model'] = model_type
    concatenated_df['Measure'] = tensorboard_type.split('_')[1]

    # Save the final concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file_path, index=False)
    print(f"Concatenated CSV saved to: {output_file_path}")




def parse_params(filename):
    """
    Parse a filename to extract Model and any encoding-specific parameters.
    Returns a dict: { 'Model': str, 'threshold': float or None, 'num_step': int or None,
                       'gain': float or None, 'tau': float or None }
    """
    
    MODEL_NAMES = ['LaskNet', 'MiniNet', 'PilotNet']
    params = {
        'Model': None,
        'threshold': None,
        'num_step': None,
        'gain': None,
        'tau': None
    }
    name, _ = os.path.splitext(filename)
    # Extract model name (first token before underscore or full name)
    # Identify model by checking known model names in filename
    # Identify model by checking known model names in filename
    for model in MODEL_NAMES:
        if model.lower() in name.lower():
            params['Model'] = model
            break

    # Delta: threshold
    m = re.search(r'threshold[_]?(\d+\.?\d*)', name, re.IGNORECASE)
    if m:
        params['threshold'] = float(m.group(1))

    # Rate: numstep and gain
    m1 = re.search(r'numsteps[_]?(\d+)', name, re.IGNORECASE)
    m2 = re.search(r'gain[_]?(\d+\.?\d*)', name, re.IGNORECASE)
    if m1:
        params['num_step'] = int(m1.group(1))
    if m2:
        params['gain'] = float(m2.group(1))

    # Latency: numstep and tau
    m3 = re.search(r'tau[_]?(\d+\.?\d*)', name, re.IGNORECASE)
    if m3:
        params['tau'] = float(m3.group(1))

    return params


def load_and_parse_results(root_dir):
    """
    Walks through the specified root directory with the structure:
        <Encoded_Method>/Tensorboard_results/Tensorboard_<Measure>/*.csv
    Loads all CSVs and returns a tidy DataFrame with columns:
        Encoded_Method, Measure, Model, threshold, num_step, gain, tau, Step, Value
    """
    all_records = []

    # Iterate over encoding methods
    for enc in os.listdir(root_dir):
        enc_path = os.path.join(root_dir, enc, 'Tensorboard_results') 
        if not os.path.isdir(enc_path):
            continue

        # Iterate over measures
        for tb_folder in os.listdir(enc_path):
            if not tb_folder.startswith('Tensorbaord_'): # folder name should 'Tensorboard_results' but is 'Tensorbaord_results'
                continue
            measure = tb_folder.replace('Tensorbaord_', '')
            measure_path = os.path.join(enc_path, tb_folder)
            
            # Iterate over model subfolders
            for model_folder in os.listdir(measure_path):
                model_path = os.path.join(measure_path, model_folder)
                if not os.path.isdir(model_path):
                    continue

                # Iterate over CSV files
                for fname in os.listdir(model_path):
                    if not fname.endswith('.csv'):
                        continue
                    file_path = os.path.join(model_path, fname)
                    df = pd.read_csv(file_path)

                
                    # Parse parameters
                    params = parse_params(fname)
                    df['Encoded_Method'] = enc
                    df['Measure'] = measure
                    df['Model'] = params.pop('Model')
                    df['threshold'] = params['threshold']
                    df['num_step'] = params['num_step']
                    df['gain'] = params['gain']
                    df['tau'] = params['tau']

                    # Ensure Step and Value columns exist
                    if 'Step' not in df.columns or 'Value' not in df.columns:
                        raise ValueError(f"Missing required columns in {file_path}")

                    all_records.append(df)

    # Combine into one DataFrame
    tidy_df = pd.concat(all_records, ignore_index=True)
    # Drop unnecessary column if present
    tidy_df = tidy_df.drop(columns=['Wall time'], errors='ignore')
    # Optional: convert types
    tidy_df['Step'] = tidy_df['Step'].astype(int)
    tidy_df['Value'] = tidy_df['Value'].astype(float)

    return tidy_df
