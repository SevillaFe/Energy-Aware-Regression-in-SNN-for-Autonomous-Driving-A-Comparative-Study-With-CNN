from math import tau
import os
from tkinter import font
from turtle import color, pos
import data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import numpy as np
from sympy import fu
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker


def plot_mse_results(folder_path):
    """
    Function to visualize MSE results from CSV files.
    It filters files ending with 'tag-val_loss.csv' and plots Step vs. Value.
    """
    # Filter and read relevant files
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith("tag-val_loss.csv")
    ]

    if not csv_files:
        print("No files found matching the criteria.")
        return

    # Prepare the plot
    plt.figure(figsize=(12, 6))

    for file in csv_files:
        # Extract parameters from the filename
        base_name = os.path.splitext(file)[0]
        num_steps = base_name.split("_")[2]
        gain = base_name.split("_")[4]

        # Read the CSV file
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep=",")
        #data["Value"] = data["Value"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Plot the data
        plt.plot(
            data["Step"],
            data["Value"],
            label=f"Num_steps {num_steps} | Gain {gain}"
        )

    # Customize the plot
    plt.title("MSE Results by Num_steps and Gain", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("MSE Value", fontsize=12)
    #plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend(title="Datasets", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    

def plot_mse_results_seaborn(folder_path):
    """
    Function to visualize MSE results from CSV files using seaborn.
    It filters files ending with 'tag-val_loss.csv' and plots Step vs. Value.
    """
    # Filter and read relevant files
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith("tag-val_loss.csv")
    ]

    if not csv_files:
        print("No files found matching the criteria.")
        return

    # Prepare a DataFrame to consolidate data for seaborn
    consolidated_data = pd.DataFrame(columns=["Step", "Value", "Configuration"])

    for file in csv_files:
        # Extract parameters from the filename
        base_name = os.path.splitext(file)[0]
        num_steps = base_name.split("_")[2]
        gain = base_name.split("_")[4]
        configuration = f"Num_steps {num_steps} | Gain {gain}"

        # Read the CSV file
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep=",")
        #data["Value"] = data["Value"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Add a configuration column
        data["Configuration"] = configuration

        # Append to consolidated DataFrame
        consolidated_data = pd.concat([consolidated_data, data[["Step", "Value", "Configuration"]]])

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=consolidated_data, x="Step", y="Value", hue="Configuration")

    # Customize the plot
    plt.title("MSE Results by Num_steps and Gain", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("MSE Value", fontsize=12)
    #plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend(title="Datasets", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    plt.tight_layout()
    plt.show()




def plot_mse_results_with_table(folder_path):
    """
    Function to visualize MSE results from CSV files using seaborn with an embedded table
    showing the last value of each line in the plot.
    """
    # Filter and read relevant files
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith("tag-val_loss.csv")
    ]

    if not csv_files:
        print("No files found matching the criteria.")
        return

    # Prepare a DataFrame to consolidate data for seaborn
    consolidated_data = pd.DataFrame(columns=["Step", "Value", "Configuration"])

    summary_data = []

    for file in csv_files:
        # Extract parameters from the filename
        base_name = os.path.splitext(file)[0]
        num_steps = base_name.split("_")[2]
        gain = base_name.split("_")[4]
        configuration = f"Num_steps {num_steps} | Gain {gain}"

        # Read the CSV file
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep=",")
        #data["Value"] = data["Value"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Add a configuration column
        data["Configuration"] = configuration

        # Append to consolidated DataFrame
        consolidated_data = pd.concat([consolidated_data, data[["Step", "Value", "Configuration"]]])

        # Get the last value
        last_step = data["Step"].iloc[-1]
        last_value = data["Value"].iloc[-1]
        summary_data.append({"Configuration": configuration, "Last Step": last_step, "Last Value": last_value})

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=consolidated_data, x="Step", y="Value", hue="Configuration")

    # Customize the plot
    plt.title("MSE Results by Num_steps and Gain", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("MSE Value", fontsize=12)
    plt.legend(title="Datasets", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    #plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add a table with the last values
    cell_text = summary_df[["Configuration", "Last Value"]].values
    table = plt.table(
        cellText=cell_text,
        colLabels=["Data set", "Last Value"],
        cellLoc='center',
        loc='bottom',
        bbox=[0.2, -0.4, 0.5, 0.25]
    )
    table.auto_set_font_size(True)
    table.set_fontsize(10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Adjust to fit the table below the plot
    plt.show()



def plot_mse_results_with_colored_table(folder_path):
    """
    Function to visualize MSE results from CSV files using seaborn with an embedded table.
    Highlights the lowest two values in the table with color.
    """
    # Filter and read relevant files
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith("tag-val_loss.csv")
    ]

    if not csv_files:
        print("No files found matching the criteria.")
        return

    # Prepare a DataFrame to consolidate data for seaborn
    consolidated_data = pd.DataFrame(columns=["Step", "Value", "Configuration"])

    summary_data = []

    for file in csv_files:
        # Extract parameters from the filename
        base_name = os.path.splitext(file)[0]
        num_steps = base_name.split("_")[2]
        gain = base_name.split("_")[4]
        configuration = f"Num_steps {num_steps} | Gain {gain}"

        # Read the CSV file
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep=",")
        #data["Value"] = data["Value"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Add a configuration column
        data["Configuration"] = configuration

        # Append to consolidated DataFrame
        consolidated_data = pd.concat([consolidated_data, data[["Step", "Value", "Configuration"]]])

        # Get the last value
        last_step = data["Step"].iloc[-1]
        last_value = round(data["Value"].iloc[-1], 5)
        summary_data.append({"Configuration": configuration, "Last Step": last_step, "Last Value": last_value})

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Highlight the two lowest values
    sorted_indices = summary_df["Last Value"].argsort()[:2]

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=consolidated_data, x="Step", y="Value", hue="Configuration")

    # Customize the plot
    plt.title("MSE Results by Num_steps and Gain", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("MSE Value", fontsize=12)
    plt.legend(title="Datasets", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    #plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add a table with the last values
    cell_text = summary_df[["Configuration", "Last Value"]].values
    table = plt.table(
        cellText=cell_text,
        colLabels=["Data set", "Last Value"],
        cellLoc='center',
        loc='bottom',
        bbox=[0.2, -0.6, 0.5, 0.25],
        colWidths=[0.3, 0.3]  # Adjust column widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Highlight the lowest two values
    for idx in sorted_indices:
        table[(idx + 1, 1)].set_facecolor("#d4edda")  # Light green background
        table[(idx + 1, 0)].set_facecolor("#d4edda")  # Light green background
        table[(idx + 1, 1)].set_edgecolor("black")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Adjust to fit the table below the plot
    plt.show()

# Custom formatter for scientific notation
def scientific_notation_formatter(val, pos):
    """Format tick labels as 1eX instead of 1×10^X."""
    return f"{val:.0e}"

def smooth_data(values, smoothing_factor=0.6):
    """
    Smooth the data using an exponential moving average.
    :param values: Array-like values to be smoothed.
    :param smoothing_factor: Factor for exponential smoothing (0 < smoothing_factor < 1).
    :return: Smoothed values as a list.
    """
    smoothed_values = []
    last = values[0]  # Start with the first value
    for value in values:
        smoothed_value = last * smoothing_factor + value * (1 - smoothing_factor)
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return smoothed_values

def plot_smoothed_results(folder_path, loss, encode, smoothing_factor=0.6):
    """
    Function to visualize smoothed MSE results from CSV files using seaborn.
    Includes an embedded table highlighting the lowest two smoothed final values.
    """
    # Filter and read relevant files
    if loss == 'mse':
        full_path = os.path.join(folder_path, "Val")
        csv_files = [
            f for f in os.listdir(full_path)
            if f.endswith("tag-val_loss.csv")
        ]
    elif loss == 'mape':
        full_path = os.path.join(folder_path, "Val")
        csv_files = [
            f for f in os.listdir(full_path)
            if f.endswith("tag-val_mape.csv")
        ]
    elif loss == 'energy_cost':
        full_path = os.path.join(folder_path, "Energy_cost_epoch")
        csv_files = [
            f for f in os.listdir(full_path)
            if f.endswith("tag-energy_cost_per_epoch.csv")
        ]
    elif loss == 'eer':
        full_path = os.path.join(folder_path, "EER")
        csv_files = [
            f for f in os.listdir(full_path)
            if f.endswith("tag-eer.csv")
        ]

    if not csv_files:
        print("No files found matching the criteria.")
        return

    # Prepare a DataFrame to consolidate data for seaborn
    consolidated_data = pd.DataFrame(columns=["Step", "Value", "Configuration"])

    summary_data = []

    for file in csv_files:
        # Extract parameters from the filename
        base_name = os.path.splitext(file)[0]
        if encode == 'rate':
            num_steps = base_name.split("_")[2]
            gain = base_name.split("_")[4]
            configuration = f"Num_steps {num_steps} | Gain {gain}"
        elif encode == 'latency':
            num_steps = base_name.split("_")[2]
            tau = base_name.split("_")[4]
            configuration = f"Num_steps {num_steps} | Tau {tau}"
        elif encode == 'delta':
            threshold = base_name.split("_")[2]
            configuration = f"Threshold {threshold}"
        else:
            configuration = 'CNN PilotNet'

        # Read the CSV file
        file_path = os.path.join(full_path, file)
        data = pd.read_csv(file_path, sep=",")
        
        #data["Value"] = data["Value"].str.replace(".", "").str.replace(",", ".").astype(float)

        # Smooth the data
        data["Smoothed Value"] = smooth_data(data["Value"], smoothing_factor=smoothing_factor)

        # Add a configuration column
        data["Configuration"] = configuration

        # Append to consolidated DataFrame
        consolidated_data = pd.concat([consolidated_data, data[["Step", "Smoothed Value", "Configuration"]]])

        # Get the last smoothed value and round to 4 decimals
        last_step = data["Step"].iloc[-1]
        last_value = round(data["Smoothed Value"].iloc[-1], 4)
        summary_data.append({"Configuration": configuration, "Last Step": last_step, "Last Smoothed Value": last_value})

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Highlight the two lowest smoothed values
    sorted_indices = summary_df["Last Smoothed Value"].argsort()[:2]

    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    palette = ['#ffa07a', '#20b2aa', '#778899', '#dda0dd']
    
    # Ensure consistent mapping between Configuration and palette
    configurations = summary_df["Configuration"].values
    color_mapping = {config: palette[i % len(palette)] for i, config in enumerate(configurations)}

    if encode == 'cnn':
        sns.lineplot(data=consolidated_data, x="Step", y="Smoothed Value", palette=color_mapping)
    else:
        sns.lineplot(data=consolidated_data, x="Step", y="Smoothed Value", hue="Configuration", palette=color_mapping)

    # Customize the plot
   
    #plt.title("Smoothed MSE Results by Num_steps and Gain", fontsize=14)
    plt.xlabel("Step", fontsize=16)
    if loss == 'mse':
        plt.ylabel("Smoothed MSE Value", fontsize=16)
    elif loss == 'mape':
        plt.ylabel("Smoothed MAPE Value", fontsize=16)
    elif loss == 'energy_cost':
        plt.ylabel("Smoothed Energy Cost Value", fontsize=16)
    elif loss == 'eer':
        plt.ylabel("Smoothed EER Value", fontsize=16)
    
    
    # Apply scientific notation to the Y-axis
    #ax = plt.gca()  # Get the current axes
    #ax.yaxis.set_major_formatter(mticker.FuncFormatter(scientific_notation_formatter))

    # Apply scientific notation to the Y-axis and X-axis
    ax = plt.gca()  # Get the current axes
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation

    # Increase font size for X and Y tick labels
    plt.tick_params(axis='x', labelsize=16)  # X-axis tick labels
    plt.tick_params(axis='y', labelsize=16)  # Y-axis tick labels
    
    # Define the last value
    #last_x = summary_df[["Last Step"]].values[-1]
    #last_y = summary_df[["Last Smoothed Value"]].values[-1]  # Replace `your_data` with your Y-axis data
    #last_x = last_x[0] if isinstance(last_x, list) or isinstance(last_x, np.ndarray) else last_x
    #last_y = last_y[0] if isinstance(last_y, list) or isinstance(last_y, np.ndarray) else last_y
    #
    ##print(last_x, last_y)
#
    ## Draw a line from the last point to the Y-axis
    #ax.plot([last_x, 0], [last_y, last_y], color='red', linestyle='--', linewidth=1.5)
    #ax.set_xlim(left=0)
    #
    ## Annotate the value on the Y-axis
    #ax.text(
    #    x=last_x, y=last_y,  # Position at the Y-axis and last_y
    #    s=f"{last_y:.2f}",  # Format the value (adjust precision as needed)
    #    color='red',  # Color of the text
    #    ha='left',  # Align the text to the right
    #    position=(last_x - 150, last_y+4000),  # Position relative to the text
    #    #va='top',  # Center the text vertically
    #    fontsize=12  # Font size of the annotation
    #)

    plt.legend(title=None, bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)

    # Add a table with the last smoothed values
    cell_text = summary_df[["Configuration", "Last Smoothed Value"]].values
    column_headers = ["Last Smoothed Value"]
    row_headers = [i for i in configurations]
    
    
    #palette = (palette * ((len(row_headers) // len(palette)) + 1))[:len(row_headers)]
    #rcolors = palette
    ccolors = ['#d3d3d3']
    
    if encode == 'cnn':
        bbox=[0.8, 0.2, 0.15, 0.15]
    elif encode == 'delta' or encode == 'latency':
        bbox=[0.8, 0.2, 0.15, 0.25]
    else:
        bbox=[0.8, 0.4, 0.15, 0.25]

    table = plt.table(cellText = summary_df[["Last Smoothed Value"]].values,
                      cellLoc='center',
                      rowLabels=row_headers,
                      #rowColours=rcolors,
                      rowLoc='right',
                      colColours=ccolors,
                      colLabels=column_headers,
                      bbox=bbox,
                      loc='center',
                      colWidths=[0.5, 0.5]  # Adjust column widths
                      )
    
        # Apply text colors to row labels using the consistent mapping
    for i, config in enumerate(row_headers):
        cell = table[(i + 1, -1)]  # Row labels are in column -1
        cell.get_text().set_color(color_mapping[config])  # Match the line colors
        cell.get_text().set_fontweight('bold')  # Optional: Make the text bold for emphasis

        

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Highlight the lowest two smoothed values
    for idx in sorted_indices:
        table[(idx + 1, 0)].set_facecolor("#d4edda")  # Light green background
        #table[(idx + 1, 0)].set_facecolor("#d4edda")
        table[(idx + 1, 0)].set_edgecolor("black")

    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.4)  # Adjust to fit the table below the plot
    plt.show()


def plot_EER_mean(file_path, smoothing_factor = 0.6):
    """
    Function to visualize EER mean results from Excel file.
    """
    # Read the CSV file
    #file_path = os.listdir(folder_path)
    df = pd.read_excel(file_path)
    # in case of smmothed data uncomment the following line
    #df["Smoothed Value"] = smooth_data(df["Value"], smoothing_factor=smoothing_factor)

    # Prepare a DataFrame to consolidate data for seaborn
    #df = pd.DataFrame(columns=["Wall time",	"Step",	"Value",	"Metric",	
    #                                          "Encoding",	"threshold",	"Num_steps",	"Gain",	"Tau"])

    #print(df)
    # Analyze and visualize EER for Delta, Rate, and Latency encoding strategies
    plt.figure(figsize=(12, 8))
    palette = ['#ffa07a', '#20b2aa', '#778899', '#dda0dd']

    # Plot the average EER across all configurations for each encoding strategy
    for encoding in ['Delta', 'Rate', 'Latency']:
        filtered_data = df[df['Metric'] == 'EER']
        filtered_data = filtered_data[filtered_data['Encoding'] == encoding]
        # Smooth the data
        # In case of smoothed data uncomment the following line
        #avg_eer = filtered_data.groupby('Step')['Smoothed Value'].mean()
        avg_eer = filtered_data.groupby('Step')['Value'].mean()
        plt.plot(avg_eer.index, avg_eer.values, label=f"{encoding} (Average EER)")

    # Plot settings for comparative EER
    #plt.title("Comparison of EER Across Delta, Rate, and Latency Encoding Strategies")
    plt.xlabel("Training Step")
    plt.ylabel("EER (Energy-to-Error Ratio)")
    plt.legend(title=None, bbox_to_anchor=(0.5, 1), loc="upper center", ncols=3)
    #plt.legend()
    #plt.grid()
    plt.show()


def plot_compare_EER_snn_cnn(file_path, loss, smoothing_factor=0.6):
    """
    Function to visualize EER mean results from SNN and EER from CNN.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    sheets = pd.read_excel(file_path, sheet_name=None)
    snn_df = sheets['SNN']
    cnn_df = sheets['CNN']
    
    filtered_cnn_data = cnn_df[cnn_df['Metric'] == 'EER']
    
    plt.figure(figsize=(12, 8))

    for encoding in ['Delta', 'Rate', 'Latency']: 
        # Filter anew for each encoding
        encoding_data = snn_df[(snn_df['Metric'] == 'EER') & (snn_df['Encoding'] == encoding)]
        avg_snn_eer = encoding_data.groupby('Step')['Value'].mean()
        plt.plot(avg_snn_eer.index, avg_snn_eer.values, label=f"SNN {encoding} (Average EER)")
    
    avg_cnn_eer = filtered_cnn_data.groupby('Step')['Value'].mean()
    plt.plot(avg_cnn_eer.index, avg_cnn_eer.values, label=f"CNN (Average EER)")
    
    plt.title("Mean EER Comparison: CNN vs. SNN Encoding Strategies")
    plt.xlabel("Step")
    plt.ylabel("EER Value")
    plt.ylim(0, 40000)  # Set the Y-axis range
    plt.legend(title=None, bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    plt.grid()
    plt.show()



def plot_compare_EER_snn_cnn_zoom(file_path, loss, smoothing_factor=0.6, zoom_xlim=None, zoom_ylim=None,
                                  inset_position="upper right", inset_bbox=None):
    """
    Function to visualize EER mean results from SNN and EER from CNN, with a zoomed-in inset plot.
    """

    sheets = pd.read_excel(file_path, sheet_name=None)
    snn_df = sheets['SNN']
    cnn_df = sheets['CNN']
    
    filtered_cnn_data = cnn_df[cnn_df['Metric'] == 'EER']
    
    plt.figure(figsize=(12, 6))

    # Main plot
    ax_main = plt.gca()  # Main plot axis handle
    for encoding in ['Delta', 'Rate', 'Latency']: 
        encoding_data = snn_df[(snn_df['Metric'] == 'EER') & (snn_df['Encoding'] == encoding)]
        avg_snn_eer = encoding_data.groupby('Step')['Value'].mean()
        ax_main.plot(avg_snn_eer.index, avg_snn_eer.values, label=f"SNN {encoding} (Average EER)")
        ax_main.grid(None)
    
    avg_cnn_eer = filtered_cnn_data.groupby('Step')['Value'].mean()
    ax_main.plot(avg_cnn_eer.index, avg_cnn_eer.values, label=f"CNN (Average EER)")
    
    #ax_main.set_title("Mean EER Comparison: CNN vs. SNN Encoding Strategies")
    ax_main.set_xlabel("Step", fontsize=16)
    ax_main.set_ylabel("EER Value", fontsize=16)
    ax_main.legend(title=None, bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    ax_main.grid(None)

    # Ensure consistent scientific notation on the main plot
    #ax_main.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
     # Apply scientific notation to the Y-axis and X-axis
    ax = plt.gca()  # Get the current axes
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation
    
    # Increase font size for X and Y tick labels
    ax_main.tick_params(axis='x', labelsize=16)  # X-axis tick labels
    ax_main.tick_params(axis='y', labelsize=16)  # Y-axis tick labels

   # Add inset plot (zoomed-in section)
    if inset_bbox:
        # Use bbox_to_anchor for precise positioning
        ax_inset = inset_axes(plt.gca(), width="40%", height="30%", bbox_to_anchor=inset_bbox, bbox_transform=plt.gcf().transFigure)
    else:
        # Use default location if bbox_to_anchor is not specified
        ax_inset = inset_axes(plt.gca(), width="40%", height="30%", loc=inset_position)
    
    for encoding in ['Delta', 'Rate', 'Latency']: 
        encoding_data = snn_df[(snn_df['Metric'] == 'EER') & (snn_df['Encoding'] == encoding)]
        avg_snn_eer = encoding_data.groupby('Step')['Value'].mean()
        ax_inset.plot(avg_snn_eer.index, avg_snn_eer.values, label=f"SNN {encoding} (Average EER)")

    avg_cnn_eer = filtered_cnn_data.groupby('Step')['Value'].mean()
    ax_inset.plot(avg_cnn_eer.index, avg_cnn_eer.values, label=f"CNN (Average EER)")

    # Set zoomed-in range
    if zoom_xlim:
        ax_inset.set_xlim(zoom_xlim)
    if zoom_ylim:
        ax_inset.set_ylim(zoom_ylim)
    
    # Format Y-axis in "4e4" scientific notation style
    #formatter = ScalarFormatter(useMathText=True)
    #formatter.set_powerlimits((0, 0))  # Force scientific notation
    #ax_inset.yaxis.set_major_formatter(formatter)
    #ax_inset.xaxis.set_major_formatter(formatter)
    
    ax2 = plt.gca()  # Get the current axes
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))  # Adjust limits for when to use scientific notation
    ax2.set_xlabel("Step", fontsize=16)
    ax2.set_ylabel("EER Value", fontsize=16)
    # Increase font size for X and Y tick labels
    ax2.tick_params(axis='x', labelsize=16)  # X-axis tick labels
    ax2.tick_params(axis='y', labelsize=16)  # Y-axis tick labels
    
    # Add title for the inset plot
    ax_inset.set_title("Zoomed In View", fontsize=16)
    ax_inset.grid()
    plt.grid(None)
    plt.show()



# Specify the folder containing the CSV files and call the function
folder_path_rate = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate"
folder_path_latency = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency"
folder_path_delta = "/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta"
folder_path_cnn = "/Users/fernando/Documents/Doctorado/Udacity_Datset/No_encoded_cnn"

file_path_all_results = "/Users/fernando/Downloads/Todos los resultados.xlsx"
# Uncomment this line and set the correct folder path to visualize results.
#plot_mse_results(folder_path_rate_val) 
#plot_mse_results_seaborn(folder_path_rate_val)
#plot_mse_results_with_table(folder_path_rate_val)
#plot_mse_results_with_colored_table(folder_path_rate_val)
#plot_smoothed_results(folder_path_latency, 'eer', 'latency')
#plot_EER_mean(file_path_all_results)
#plot_compare_EER_snn_cnn(file_path_all_results, 'eer')
plot_compare_EER_snn_cnn_zoom(file_path=file_path_all_results,loss='EER',
                              zoom_xlim=(10, 2820), zoom_ylim=(0, 40000),
                              inset_position="center") #inset_bbox=(0.1, 0.7, 0.2, 0.2) 
# 
# mse mape eer energy_cost

