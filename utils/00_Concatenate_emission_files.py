from pyexpat import model
import utils.utils as myUtils
import os

if __name__ == "__main__":
    
    encode_type = 'Latency'  # Options: 'Rate', 'Latency', 'Delta'
    model_type = 'LaskNet' # Options: 'MiniNet', 'LaskNet', 'PilotNet'
    
    # Define the directory containing the CSV files
    folder_path = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/{encode_type}/Emissions/Emissions_model/{model_type}'
    name_file = f'training_emissions_'
    
    name_file_output = f'concatenated_emissions_{encode_type}.csv'
    # Define the output file path for the concatenated CSV
    output_file_path = f'/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/{encode_type}/Emissions/Emissions_model/{model_type}/{name_file_output}'
    

    
    # Call the function to concatenate CSV files
    myUtils.concatenate_emissions_csv_files(folder_path, output_file_path, name_file, encode_type)
    
    print(f"Concatenated CSV files saved to {output_file_path}")