import os
import pandas as pd
import pickle


def load_pkl_file(file_path):
    """
    Load and display a DataFrame stored in a .pkl file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
            if isinstance(data, pd.DataFrame):
                print("File contains a pandas DataFrame.")
                
                # Adjust pandas display settings
                pd.set_option('display.max_rows', None)  # Show all rows
                pd.set_option('display.max_columns', None)  # Show all columns
                pd.set_option('display.width', 1000)  # Set the width for wide DataFrames
                pd.set_option('display.colheader_justify', 'left')  # Align columns for readability
                
                print(data)
            else:
                print("Contents of the file:")
                print(data)  # For non-DataFrame objects
            return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Replace 'your_file.pkl' with the path to your .pkl file
    
    study_dir_rate = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Optuna"
    study_dir_latency = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Optuna"
    study_dir_delta = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Optuna"
    pkl_file = "optuna_study.pkl"
    
    pkl_file_path_rate = os.path.join(study_dir_rate, pkl_file)
    pkl_file_path_latency = os.path.join(study_dir_latency, pkl_file)
    pkl_file_path_delta = os.path.join(study_dir_delta, pkl_file)
    
    contents = load_pkl_file(pkl_file_path_rate)
    
    # If the contents are loaded successfully, print them
    #if contents is not None:
    #    print("Contents of the .pkl file:")
    #    print(contents)