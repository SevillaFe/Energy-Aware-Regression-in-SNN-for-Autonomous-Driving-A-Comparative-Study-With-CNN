import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Call the function to plot the results
file_path_rate = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Emissions/Emissions_model/output_emissions_file.csv'
file_path_latency = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions/Emissions_model/output_emissions_file.csv'
file_path_delta = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Emissions/Emissions_model/output_emissions_file.csv'

#data = pd.read_csv(file_path_latency, sep=';', decimal=',')
    
#data['energy_consumed'] = data['energy_consumed'] * 1000
#data['cpu_energy'] = data['cpu_energy'] * 1000
#data['gpu_energy'] = data['gpu_energy'] * 1000
#data['ram_energy'] = data['ram_energy'] * 1000

#print(data)


# Grouped Bar Chart: Energy Consumed vs. num_steps, grouped by gain
def grouped_bar_chart(data):

    palette = ['#ffa07a', '#20b2aa', '#778899']
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=data,
        x="num_steps",
        y="energy_consumed",
        hue="gain",
        palette=palette,
        errorbar=None
    )
    ax.bar_label(ax.containers[0], fmt='%.3f', fontsize=10)
    ax.bar_label(ax.containers[1], fmt='%.3f' , fontsize=10)
    plt.title("Energy Consumption by Encoding Parameters (Grouped by Gain)", fontsize=14)
    plt.xlabel("Number of Steps (num_steps)", fontsize=12)
    plt.ylabel("Energy Consumed (Wh)", fontsize=12)
    plt.ylim(0, data['energy_consumed'].max() * 1.2)  # Extend the y-axis by 20%
    plt.legend(title="Gain", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=2)
    # Adjust the legend position to be well above the title and outside the plot area
    #plt.legend(title="Gain", bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #                  ncols=2, mode="expand", borderaxespad=0.)
    sns.despine()
    plt.tight_layout()
    plt.show()

# Stacked Bar Chart: Breakdown of Energy Consumption
def stacked_bar_chart(data, encode):
    """
    Create a stacked bar chart to visualize the breakdown of energy consumption by CPU, GPU, and RAM.
    """
    
    if encode == 'rate':
    # Aggregate data
        energy_data = data[['num_steps', 'gain', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'gain']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
        bottom = None
        
    elif encode == 'latency':
        energy_data = data[['num_steps', 'tau', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'tau']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['tau'].astype(str)
        bottom = None
        
    elif encode == 'delta':
        energy_data = data[['threshold', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['threshold']).sum().reset_index()
        bar_positions = energy_data['threshold'].astype(str)
        bottom = None


    # Set up the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #plt.figure(figsize=(10, 6))
    #bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
    #bottom = None

    for energy_type, color in zip(['cpu_energy', 'gpu_energy', 'ram_energy'], ['#ffa07a', '#20b2aa', '#778899']):
        p = ax.bar(
            bar_positions,
            energy_data[energy_type],
            label=energy_type.replace('_energy', '').upper(),
            color=color,
            bottom=bottom
        )
        ax.bar_label(p, fmt='%.3f', label_type='center')
        bottom = energy_data[energy_type] if bottom is None else bottom + energy_data[energy_type]

    # Chart aesthetics
    #plt.title("Energy Consumption Breakdown by Encoding Parameters", fontsize=14)
    
    if encode == 'rate':
        plt.xlabel("Encoding Parameters (num_steps | Gain)", fontsize=12)
    elif encode == 'latency':
        plt.xlabel("Encoding Parameters (num_steps | Tau)", fontsize=12)
    elif encode == 'delta':
        plt.xlabel("Encoding Parameters (Threshold)", fontsize=12)
        
    
    plt.ylabel("Energy Consumed (Wh)", fontsize=12)
    plt.ylim(0, data['energy_consumed'].max() * 1.2)  # Extend the y-axis by 20%
    plt.legend(title="Hardware Type", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    #plt.legend(title="Energy Type", loc="upper right")
    plt.xticks(rotation=0, ha="center")
    sns.despine()
    plt.tight_layout()
    plt.show()
    

def grouped_bar_chart_with_secondary_emissions(data, encode):
    """
    Create a grouped bar chart for energy consumption with emissions as markers using a secondary Y-axis.
    """
    # Convert emissions from kg to g
    data['emissions_g'] = data['emissions'] * 1000  # Convert emissions to grams
    #print(data['emissions_g'])

    palette = ['#ffa07a', '#20b2aa', '#778899']

    # Create the main figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot energy consumption bars
    if encode == 'rate':
        bar = sns.barplot(
            data=data,
            x="num_steps",
            y="energy_consumed",
            hue="gain",
            palette=palette,
            errorbar=None,
            ax=ax1
        )
    elif encode == 'latency':
        bar = sns.barplot(
            data=data,
            x="num_steps",
            y="energy_consumed",
            hue="tau",
            palette=palette,
            errorbar=None,
            ax=ax1
        )
    elif encode == 'delta':
        bar = sns.barplot(
            data=data,
            x="threshold",
            y="energy_consumed",
            palette=palette,
            errorbar=None,
            ax=ax1
        )
    
    #ax1.set_title("Energy Consumption and Emissions by Encoding Parameters", fontsize=14)
    if encode == 'rate'or encode == 'latency':
        ax1.set_xlabel("Number of Steps (num_steps)", fontsize=12)
    elif encode == 'delta':
        ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Energy Consumed (Wh)", fontsize=12)
    ax1.set_ylim(0, data['energy_consumed'].max() * 1.2)  # Extend the primary Y-axis by 20%
    sns.despine(ax=ax1)

    # Add bar labels for energy consumption
    for container in bar.containers:
        ax1.bar_label(container , fmt='%.3f', fontsize=10)
        
    # Scale secondary Y-axis to match the primary Y-axis range
    primary_max = data['energy_consumed'].max()
    secondary_max = data['emissions_g'].max()
    scale_factor = primary_max / secondary_max

    # Create a secondary Y-axis for emissions
    ax2 = ax1.twinx()
    ax2.set_ylabel("CO2 Emissions (g)", fontsize=12)
    ax2.set_ylim(0, secondary_max * scale_factor * 1.2)  # Extend the secondary Y-axis by 20%

    # Overlay emissions markers on the bars
    for bar_rect, emission in zip(ax1.patches, data['emissions_g']):
        # Calculate the x and y position for the marker
        x_pos = bar_rect.get_x() + bar_rect.get_width() / 2  # Center of the bar
        y_pos = emission  # Directly use the emission value in grams
        ax2.plot(
            x_pos,
            y_pos,
            marker='o',
            color='black',
            markersize=12,
            label="Emissions" if 'Emissions' not in ax2.get_legend_handles_labels()[1] else ""
        )
        
        # Add labels for each marker
        ax2.text(
            x_pos,
            y_pos + (0.04 * primary_max),  # Slightly above the marker
            f"{emission:.1f}",
            color="black",
            ha="center",
            fontsize=10
        )

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    combined_handles = handles1 + handles2[:1]  # Add emissions marker
    combined_labels = labels1 + ["Emissions"]
    ax1.legend(combined_handles, combined_labels, title="Legend", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=3)

    plt.tight_layout()
    plt.show()


def stacked_bar_chart_excel(file_path, encode):
    """
    Create a stacked bar chart to visualize the breakdown of energy consumption by CPU, GPU, and RAM.
    """
    
    sheets = pd.read_excel(file_path, sheet_name=None)
    data = sheets['Encode CO2_Energy']
    
    data['energy_consumed'] = data['energy_consumed']
    data['cpu_energy'] = data['cpu_energy'] * 1000
    data['gpu_energy'] = data['gpu_energy'] * 1000
    data['ram_energy'] = data['ram_energy'] * 1000
    
    print(data)
    
    if encode == 'rate':
    # Aggregate data
        energy_data = data[['num_steps', 'gain', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'gain']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
        bottom = None
        
    elif encode == 'latency':
        energy_data = data[['num_steps', 'tau', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'tau']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['tau'].astype(str)
        bottom = None
        
    elif encode == 'delta':
        energy_data = data[['threshold', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['threshold']).sum().reset_index()
        bar_positions = energy_data['threshold'].astype(str)
        bottom = None


    # Set up the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #plt.figure(figsize=(10, 6))
    #bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
    #bottom = None

    for energy_type, color in zip(['cpu_energy', 'gpu_energy', 'ram_energy'], ['#ffa07a', '#20b2aa', '#778899']):
        p = ax.bar(
            bar_positions,
            energy_data[energy_type],
            label=energy_type.replace('_energy', '').upper(),
            color=color,
            bottom=bottom
        )
        ax.bar_label(p, fmt='%.3f', label_type='center')
        bottom = energy_data[energy_type] if bottom is None else bottom + energy_data[energy_type]

    # Chart aesthetics
    #plt.title("Energy Consumption Breakdown by Encoding Parameters", fontsize=14)
    
    if encode == 'rate':
        plt.xlabel("Encoding Parameters (num_steps | Gain)", fontsize=12)
    elif encode == 'latency':
        plt.xlabel("Encoding Parameters (num_steps | Tau)", fontsize=12)
    elif encode == 'delta':
        plt.xlabel("Encoding Parameters (Threshold)", fontsize=12)
        
    
    plt.ylabel("Energy Consumed (Wh)", fontsize=12)
    plt.ylim(0, data['energy_consumed'].max() * 1.2)  # Extend the y-axis by 20%
    plt.legend(title="Energy Type", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    #plt.legend(title="Energy Type", loc="upper right")
    plt.xticks(rotation=0, ha="center")
    sns.despine()
    plt.tight_layout()
    plt.show()
    
def grouped_bar_chart_with_secondary_emissions_excel(file_path, encode):
    """
    Create a grouped bar chart for energy consumption with emissions as markers using a secondary Y-axis.
    """
    sheets = pd.read_excel(file_path, sheet_name=None)
    data = sheets['Encode CO2_Energy']
    
    # Convert emissions from kg to g
    data['emissions_g'] = data['emissions'] * 1 # already in grams 
    #print(data['emissions_g'])

    palette = ['#ffa07a', '#20b2aa', '#778899']

    # Create the main figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot energy consumption bars
    if encode == 'rate':
        subset = data[data["Encoding"] == "Rate"]
        subset = subset.sort_values(by=["num_steps", "gain"])
        bar = sns.barplot(
            data=data,
            x="num_steps",
            y="energy_consumed",
            hue="gain",
            palette=palette,
            order=subset["num_steps"].unique(),
            hue_order=subset["gain"].unique(),
            errorbar=None,
            ax=ax1
        )
    elif encode == 'latency':
        subset = data[data["Encoding"] == "Latency"]
        subset = subset.sort_values(by=["num_steps", "tau"])
        bar = sns.barplot(
            data=data,
            x="num_steps",
            y="energy_consumed",
            hue="tau",
            palette=palette,
            order=subset["num_steps"].unique(),
            hue_order=subset["tau"].unique(),
            errorbar=None,
            ax=ax1
        )
    elif encode == 'delta':
        subset = data[data["Encoding"] == "Delta"]
        subset = subset.sort_values(by=["threshold"])
        bar = sns.barplot(
            data=data,
            x="threshold",
            y="energy_consumed",
            palette=palette,
            order=subset["threshold"].unique(),
            errorbar=None,
            ax=ax1
        )
    
    #ax1.set_title("Energy Consumption and Emissions by Encoding Parameters", fontsize=14)
    if encode == 'rate'or encode == 'latency':
        ax1.set_xlabel("Number of Steps (num_steps)", fontsize=12)
    elif encode == 'delta':
        ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Energy Consumed (Wh)", fontsize=12)
    ax1.set_ylim(0, subset['energy_consumed'].max() * 1.2)  # Extend the primary Y-axis by 20%
    sns.despine(ax=ax1)

    # Add bar labels for energy consumption
    for container in bar.containers:
        ax1.bar_label(container , fmt='%.3f', fontsize=10)
    
    # Scale secondary Y-axis to match the primary Y-axis range
    primary_max = subset['energy_consumed'].max()
    secondary_max =subset["emissions_g"].max()
    scale_factor = primary_max / secondary_max

    # Create a secondary Y-axis for emissions
    ax2 = ax1.twinx()
    ax2.set_ylabel("CO2 Emissions (g)", fontsize=12)
    ax2.set_ylim(0, secondary_max * scale_factor * 1.2)  # Extend the secondary Y-axis by 20%


    # Overlay emissions markers on the bars
    # Match emissions to bars using tolerance
    for bar_rect in ax1.patches:
        bar_height = bar_rect.get_height()
        # Find the corresponding row in the subset where energy_consumed matches the bar height within a tolerance
        match = subset.loc[np.isclose(subset["energy_consumed"], bar_height)]
        if not match.empty:
            emission = match["emissions_g"].values[0]
            x_pos = bar_rect.get_x() + bar_rect.get_width() / 2
            y_pos = emission
            ax2.plot(
                x_pos,
                y_pos,
                marker='o',
                color='black',
                markersize=12,
                label="Emissions" if 'Emissions' not in ax2.get_legend_handles_labels()[1] else ""
            )
        
        # Add labels for each marker
        ax2.text(
            x_pos,
            y_pos + (0.04 * primary_max),  # Slightly above the marker
            f"{emission:.3f}",
            color="black",
            ha="center",
            fontsize=10
        )

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    combined_handles = handles1 + handles2[:1]  # Add emissions marker
    combined_labels = labels1 + ["Emissions"]
    if encode == 'rate':
        ax1.legend(combined_handles, combined_labels, title="Gain", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=3)
    elif encode == 'latency':
        ax1.legend(combined_handles, combined_labels, title="Tau", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=3)
    elif encode == 'delta':
        ax1.legend(combined_handles, combined_labels, title=None, bbox_to_anchor=(0.5, 1), loc="upper center", ncols=3)

    plt.tight_layout()
    plt.show()

def stacked_bar_chart_exel(file_path, encode):
    """
    Create a stacked bar chart to visualize the breakdown of energy consumption by CPU, GPU, and RAM.
    """
    
    sheets = pd.read_excel(file_path, sheet_name=None)
    data = sheets['Encode CO2_Energy']
    
    if encode == 'rate':
        subset = data[data["Encoding"] == "Rate"]
        #subset['energy_consumed'] = subset['energy_consumed'] * 1000
        subset['cpu_energy'] = subset['cpu_energy'] * 1000
        subset['gpu_energy'] = subset['gpu_energy'] * 1000
        subset['ram_energy'] = subset['ram_energy'] * 1000
        subset = subset.sort_values(by=["num_steps", "gain"])
    # Aggregate data
        energy_data = subset[['num_steps', 'gain', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'gain']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
        bottom = None
        print(energy_data)
        
    elif encode == 'latency':
        subset = data[data["Encoding"] == "Latency"]
        #subset['energy_consumed'] = subset['energy_consumed'] * 1000
        subset['cpu_energy'] = subset['cpu_energy'] * 1000
        subset['gpu_energy'] = subset['gpu_energy'] * 1000
        subset['ram_energy'] = subset['ram_energy'] * 1000
        subset = subset.sort_values(by=["num_steps", "tau"])
        energy_data = subset[['num_steps', 'tau', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['num_steps', 'tau']).sum().reset_index()
        bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['tau'].astype(str)
        bottom = None
        
    elif encode == 'delta':
        subset = data[data["Encoding"] == "Delta"]
        #subset['energy_consumed'] = subset['energy_consumed'] * 1000
        subset['cpu_energy'] = subset['cpu_energy'] * 1000
        subset['gpu_energy'] = subset['gpu_energy'] * 1000
        subset['ram_energy'] = subset['ram_energy'] * 1000
        subset = subset.sort_values(by=["threshold"])
        energy_data = subset[['threshold', 'cpu_energy', 'gpu_energy', 'ram_energy']].groupby(
            ['threshold']).sum().reset_index()
        bar_positions = energy_data['threshold'].astype(str)
        bottom = None


    # Set up the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    #plt.figure(figsize=(10, 6))
    #bar_positions = energy_data['num_steps'].astype(str) + " | " + energy_data['gain'].astype(str)
    #bottom = None

    for energy_type, color in zip(['cpu_energy', 'gpu_energy', 'ram_energy'], ['#ffa07a', '#20b2aa', '#778899']):
        p = ax.bar(
            bar_positions,
            energy_data[energy_type],
            label=energy_type.replace('_energy', '').upper(),
            color=color,
            bottom=bottom
        )
        ax.bar_label(p, fmt='%.3f', label_type='center')
        bottom = energy_data[energy_type] if bottom is None else bottom + energy_data[energy_type]

    # Chart aesthetics
    #plt.title("Energy Consumption Breakdown by Encoding Parameters", fontsize=14)
    
    if encode == 'rate':
        plt.xlabel("Encoding Parameters (num_steps | Gain)", fontsize=12)
    elif encode == 'latency':
        plt.xlabel("Encoding Parameters (num_steps | Tau)", fontsize=12)
    elif encode == 'delta':
        plt.xlabel("Encoding Parameters (Threshold)", fontsize=12)
        
    
    plt.ylabel("Energy Consumed (Wh)", fontsize=12)
    plt.ylim(0, subset['energy_consumed'].max() * 1.2)  # Extend the y-axis by 20%
    plt.legend(title="Hardware Type", bbox_to_anchor=(0.5, 1), loc="upper center", ncols=4)
    #plt.legend(title="Energy Type", loc="upper right")
    plt.xticks(rotation=0, ha="center")
    sns.despine()
    plt.tight_layout()
    plt.show()
    


#grouped_bar_chart(data)

#stacked_bar_chart(data, 'latency')
#grouped_bar_chart_with_secondary_emissions(data,'rate')

#excel
file_path_excel = '/Users/fernando/Downloads/Todos los resultados.xlsx'
#stacked_bar_chart_excel(file_path_excel, 'delta')
#stacked_bar_chart_exel(file_path_excel, 'latency')
grouped_bar_chart_with_secondary_emissions_excel(file_path_excel, 'delta')
