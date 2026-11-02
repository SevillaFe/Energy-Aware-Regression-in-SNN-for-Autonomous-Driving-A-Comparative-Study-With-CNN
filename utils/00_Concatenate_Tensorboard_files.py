import utils.utils as myUtils
import os

if __name__ == "__main__":
    
    
    encode_type = 'Rate'  # Options: 'Rate', 'Latency', 'Delta', 'No_encoded'
    model_type = 'MiniNet' # Options: 'MiniNet', 'LaskNet', 'PilotNet'
    tensorboard_type = 'Tensorbaord_EER_Train'  # Options: 'Tensorbaord_EER_Train', 'Tensorbaord_EnergyCost', 'Tensorbaord_MAPE_Val', 'Tensorbaord_MSE_Train','Tensorbaord_MSE_Val'
    
    # Define the directory containing the CSV files
    folder_path_no_encoded = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/No_encoded_cnn/Tensorboard_Results/{model_type}/{tensorboard_type}'
    folder_path_encoded = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/{encode_type}/Tensorboard_Results/{tensorboard_type}/{model_type}'

    
    # Define the output file path for the concatenated CSV
    output_file_path_no_encoded = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/No_encoded_cnn/Tensorboard_Results/{model_type}/{tensorboard_type}/average_{tensorboard_type}.csv'
    output_file_path_encoded = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/{encode_type}/Tensorboard_Results/{tensorboard_type}/{model_type}/average_{tensorboard_type}_{model_type}.csv'

    
    #output_file_path = "/path/to/your/output_file.csv"
    join_column = "Step"  # Replace with the column name to join on

    #myUtils.concatenate_tensorboard_csv_files(folder_path_no_encoded, output_file_path_no_encoded, 
    #                                          join_column, 
    #                                          encode_type='No_encoded',
    #                                          model_type=model_type,
    #                                          tensorboard_type=tensorboard_type)
    #
    #print(f"Concatenated CSV files saved to {output_file_path_no_encoded}")
    
    myUtils.concatenate_tensorboard_csv_files(folder_path_encoded, output_file_path_encoded, join_column,
                                             encode_type=encode_type,
                                             model_type=model_type,
                                             tensorboard_type=tensorboard_type)
    print(f"Concatenated CSV files saved to {output_file_path_encoded}")
