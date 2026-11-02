import utils.utils as myUtils


experiment_csv_file_path = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions/Emissions_model'
experiment_csv_output_file_path = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions/Emissions_model/output_emissions_file.csv'
myUtils.concatenate_csv_files(experiment_csv_file_path, experiment_csv_output_file_path, 'emissions_', 'rate')
#myUtils.delete_emission_csv_files(experiment_csv_file_path, 'emissions_')
#myUtils.delete_emission_csv_files(experiment_csv_file_path, 'powermetrics_')