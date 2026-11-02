import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file to examine the contents and prepare for analysis.
file_path_rate = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Rate/Emissions/output_emissions_file.csv'
data_rate = pd.read_csv(file_path_rate)

file_path_latency = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Latency/Emissions/output_emissions_file.csv'
data_latency = pd.read_csv(file_path_latency)

file_path_delta = '/Users/fernando/Documents/Doctorado/Udacity_Datset/Encoded_Images/Delta/Emissions/output_emissions_file.csv'
data_delta = pd.read_csv(file_path_delta)

#parameter = 'gain' #'tau'

def plot_encode_emissions_results(data, parameter):
    
    if parameter == 'rate' or parameter == 'latency':
          
    # Group by num_steps and gain to calculate the mean emissions and energy consumption
        grouped_data = data.groupby(['num_steps', parameter]).agg({
            'emissions': 'mean',
            'cpu_energy': 'mean',
            'gpu_energy': 'mean',
            'ram_energy': 'mean',
            'energy_consumed': 'mean',
            'duration': 'mean'
        }).reset_index()

        # Renaming the energy_consumed column for consistency in the summary
        grouped_data = grouped_data.rename(columns={'energy_consumed': 'total_energy_consumed'})
        #print(grouped_data)


        # Implementing the final improvements based on the feedback
        melted_data = grouped_data.melt(id_vars=['num_steps', parameter], 
                                        value_vars=['cpu_energy', 'gpu_energy', 'ram_energy'],
                                        var_name='Component', 
                                        value_name='Energy')

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # 1. Heatmap of Total Energy Consumption
        heatmap_data = grouped_data.pivot(index=parameter, columns='num_steps', values='total_energy_consumed')
        sns.heatmap(heatmap_data, annot=True, fmt=".1e", cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Heatmap of Total Energy Consumption')
        axes[0, 0].set_xlabel('Number of Steps')
        axes[0, 0].set_ylabel(parameter)


        # 2. Component-wise Energy Consumption Bar Chart
        sns.barplot(data=melted_data, x='num_steps', y='Energy', hue='Component', ax=axes[0, 1])
        axes[0, 1].set_title('Component-wise Energy by Steps')
        axes[0, 1].set_xlabel('Number of Steps')
        axes[0, 1].set_ylabel('Energy (J)')
        #axes[0, 1].set_yscale('log')

        # Ensure consistent placement for all legends using bbox_to_anchor and loc
        axes[0, 1].legend(loc='upper center', ncol=3)

        # 3. Box Plot of Energy Consumption by Gain (adjusted y-scale)
        sns.boxplot(data=grouped_data, x=parameter, y='total_energy_consumed', ax=axes[1, 0])
        axes[1, 0].set_title('Box Plot: Energy by Gain')
        axes[1, 0].set_xlabel(parameter)
        axes[1, 0].set_ylabel('Total Energy (J)')
        #axes[1, 0].set_yscale('log')
        #axes[1, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
        #axes[1, 0].legend(loc='upper center', ncol=3)

        # 4. Duration vs. Energy Consumption with Dual Axis (with visible legends)
        ax1 = axes[1, 1]
        sns.lineplot(data=grouped_data, x='num_steps', y='total_energy_consumed', hue=parameter, marker='o', ax=ax1)
        ax1.set_xlabel('Number of Steps')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title('Duration and Energy vs. Num Steps')
        #ax1.set_yscale('log')

        # Create second y-axis for duration
        ax2 = ax1.twinx()
        sns.lineplot(data=grouped_data, x='num_steps', y='duration', hue=parameter, marker='x', linestyle='--', ax=ax2)
        ax2.set_ylabel('Duration (s)')
        #ax2.set_yscale('log')

        # Ensure consistent placement for both legends using bbox_to_anchor and loc
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='upper center', ncol=4)
        ax2.legend(lines2, labels2, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.92))

        plt.show()
    
    else:
        grouped_data = data.groupby([parameter]).agg({
            'emissions': 'mean',
            'cpu_energy': 'mean',
            'gpu_energy': 'mean',
            'ram_energy': 'mean',
            'energy_consumed': 'mean',
            'duration': 'mean'
        }).reset_index()
        
        # Renaming the energy_consumed column for consistency in the summary
        grouped_data = grouped_data.rename(columns={'energy_consumed': 'total_energy_consumed'})
        #print(grouped_data)


        # Implementing the final improvements based on the feedback
        melted_data = grouped_data.melt(id_vars=[parameter], 
                                        value_vars=['cpu_energy', 'gpu_energy', 'ram_energy'],
                                        var_name='Component', 
                                        value_name='Energy')

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Heatmap of Total Energy Consumption
        heatmap_data = grouped_data.pivot( columns=parameter, values='total_energy_consumed')
        sns.heatmap(heatmap_data, annot=True, fmt=".1e", cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Heatmap of Total Energy Consumption')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel(parameter)


        # 2. Component-wise Energy Consumption Bar Chart
        sns.barplot(data=melted_data, x=parameter, y='Energy', hue='Component', ax=axes[0, 1])
        axes[0, 1].set_title('Component-wise Energy by Steps')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Energy (J)')
        #axes[0, 1].set_yscale('log')

        # Ensure consistent placement for all legends using bbox_to_anchor and loc
        axes[0, 1].legend(loc='upper center', ncol=3)

        # 3. Box Plot of Energy Consumption by Gain (adjusted y-scale)
        sns.boxplot(data=grouped_data, x=parameter, y='total_energy_consumed', ax=axes[1, 0])
        axes[1, 0].set_title('Box Plot: Energy by Gain')
        axes[1, 0].set_xlabel(parameter)
        axes[1, 0].set_ylabel('Total Energy (J)')
        #axes[1, 0].set_yscale('log')
        #axes[1, 0].get_yaxis().set_major_formatter(plt.ScalarFormatter())
        #axes[1, 0].legend(loc='upper center', ncol=3)

        # 4. Duration vs. Energy Consumption with Dual Axis (with visible legends)
        ax1 = axes[1, 1]
        sns.lineplot(data=grouped_data, x=parameter, y='total_energy_consumed',  marker='o', ax=ax1, legend=False)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Total Energy (J)')
        ax1.set_title('Duration and Energy vs. Threshold - Delta Encoding')
        #ax1.set_yscale('log')

        # Create second y-axis for duration
        ax2 = ax1.twinx()
        sns.lineplot(data=grouped_data, x=parameter, y='duration',  marker='o',  linestyle='--', ax=ax2, legend=False)
        ax2.set_ylabel('Duration (s)')
        #ax2.set_yscale('log')

        # Ensure consistent placement for both legends using bbox_to_anchor and loc
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,  ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))


        plt.show()
    
    
#plot_encode_emissions_results(data_rate, 'gain')
plot_encode_emissions_results(data_latency, 'tau')
#plot_encode_emissions_results(data_delta, 'threshold')


