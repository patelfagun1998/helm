import matplotlib.pyplot as plt
import numpy as np

# Try to import tueplots for styling, fall back to default if not available
try:
    from tueplots import bundles
    plt.rcParams.update(bundles.neurips2024())
    plt.rcParams['text.usetex'] = False
except ImportError:
    # Use matplotlib default styling if tueplots is not available
    plt.style.use('default')

# --- Helper function to plot a horizontal grouped bar chart ---
def plot_horizontal_grouped_bar_chart(ax, models, scores1, scores2, scores3, label1, label2, label3,
                                      color1, color2, color3, title, x_label, xlim_upper, bar_height,
                                      show_yticklabels=True):
    """
    Plots a horizontal grouped bar chart on the given axes for 3 age groups.
    """
    num_models = len(models)
    y_pos = np.arange(num_models)*1.75 # Positions for each model group on the y-axis

    # Three categories - use smaller bar height and adjust positions
    bar_height_3way = bar_height * 0.7  # Reduce bar thickness for 3-way comparison
    bar1 = ax.barh(y_pos - bar_height_3way, scores1, height=bar_height_3way,
                   label=label1, color=color1, align='center')
    bar2 = ax.barh(y_pos, scores2, height=bar_height_3way,
                   label=label2, color=color2, align='center')
    bar3 = ax.barh(y_pos + bar_height_3way, scores3, height=bar_height_3way,
                   label=label3, color=color3, align='center')

    # Set y-axis ticks and labels (model names)
    ax.set_yticks(y_pos)
    if show_yticklabels:
        ax.set_yticklabels(models, fontsize=16)
    else:
        ax.set_yticklabels([''] * len(models))

    ax.invert_yaxis()  # Display models from top to bottom
    ax.tick_params(axis='y', labelsize=16)

    # Set x-axis label and limits
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=16)

    # Set title for the subplot
    ax.set_title(title, fontsize=18, fontweight='bold')

    # Add gridlines (vertical gridlines along the x-axis)
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.7, axis='x')
    ax.set_axisbelow(True)  # Ensure grid is behind bars

    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# --- Data Dictionary Structure for Age Group Analysis ---
# Each model contains symptom classification scores and transcription WER scores for 3 age groups
age_group_data = {
    'gpt-4o-audio': {
        'disorder_type_classification': {'5 - 7 y/o': 0.561, '8 - 10 y/o': 0.336, '10+ y/o': 0.261},
        'transcription_wer': {'5 - 7 y/o': 5.630, '8 - 10 y/o': 6.052, '10+ y/o': 4.162}
    },
    'gemini-2.0-flash': {
        'disorder_type_classification': {'5 - 7 y/o': 0.475, '8 - 10 y/o': 0.269, '10+ y/o': 0.200},
        'transcription_wer': {'5 - 7 y/o': 4.346, '8 - 10 y/o': 4.452, '10+ y/o': 2.960}
    },
    'gemini-2.0-flash-lite': {
        'disorder_type_classification': {'5 - 7 y/o': 0.306, '8 - 10 y/o': 0.161, '10+ y/o': 0.106},
        'transcription_wer': {'5 - 7 y/o': 3.290, '8 - 10 y/o': 3.458, '10+ y/o': 2.650}
    },
    'gpt-4o-mini-audio': {
        'disorder_type_classification': {'5 - 7 y/o': 0.232, '8 - 10 y/o': 0.120, '10+ y/o': 0.072},
        'transcription_wer': {'5 - 7 y/o': 4.416, '8 - 10 y/o': 5.166, '10+ y/o': 5.008}
    },
    'gpt-4o-audio-transcribe': {
        'disorder_type_classification': {'5 - 7 y/o': 0.496, '8 - 10 y/o': 0.354, '10+ y/o': 0.289},
        'transcription_wer': {'5 - 7 y/o': 4.276, '8 - 10 y/o': 4.674, '10+ y/o': 3.342}
    },
    'gpt-4o-audio-mini-transcribe': {
        'disorder_type_classification': {'5 - 7 y/o': 0.491, '8 - 10 y/o': 0.343, '10+ y/o': 0.254},
        'transcription_wer': {'5 - 7 y/o': 5.404, '8 - 10 y/o': 5.466, '10+ y/o': 3.684}
    },
    'whispr+gpt4o': {
        'disorder_type_classification': {'5 - 7 y/o': 0.522, '8 - 10 y/o': 0.391, '10+ y/o': 0.393},
        'transcription_wer': {'5 - 7 y/o': 10.898, '8 - 10 y/o': 10.226, '10+ y/o': 4.866}
    }
}

# Extract model names from dictionary keys
models = list(age_group_data.keys())

# Age groups for display and keys
age_group_keys = ['5 - 7 y/o', '8 - 10 y/o', '10+ y/o']

# Helper function to extract data arrays from the dictionary
def extract_age_group_data(models, data_dict, metric_type, age_group_keys):
    """Extract data arrays for specified metric across age groups from the model data dictionary."""
    data_matrix = []
    for model in models:
        model_data = []
        for age_key in age_group_keys:
            model_data.append(data_dict[model][metric_type][age_key])
        data_matrix.append(model_data)
    return np.array(data_matrix)

# Extract data arrays for plotting
disorder_type_scores = extract_age_group_data(models, age_group_data, 'disorder_type_classification', age_group_keys)
transcription_wer = extract_age_group_data(models, age_group_data, 'transcription_wer', age_group_keys)

# Color scheme for the 3 age groups (matching extra_analysis_figure style)
colors = ['#edbf33', '#f46a9b', '#27aeef']  # Yellow, Pink, Blue - similar to language colors

# Chart parameters
bar_height = 0.6
x_lim_symptom = 0.8
x_lim_wer = 7.0

# --- Create the figure with two subplots ---
fig_width, fig_height = 10, 10
fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))

# --- Subplot 1: Disorder Type Classification Scores ---
plot_horizontal_grouped_bar_chart(
    axes[0], models, 
    disorder_type_scores[:, 0], disorder_type_scores[:, 1], disorder_type_scores[:, 2],
    '5-7 years', '8-10 years', '10+ years',
    colors[0], colors[1], colors[2],
    'Disorder Type Classification', 'Micro F1 Score ↑', x_lim_symptom, bar_height,
    show_yticklabels=True
)

# --- Subplot 2: Transcription Accuracy (WER) ---
plot_horizontal_grouped_bar_chart(
    axes[1], models,
    transcription_wer[:, 0], transcription_wer[:, 1], transcription_wer[:, 2],
    '5-7 years', '8-10 years', '10+ years',
    colors[0], colors[1], colors[2],
    'Transcription Accuracy', 'Word Error Rate (WER) ↓', x_lim_wer, bar_height,
    show_yticklabels=True
)

fig.legend(['5-7 years', '8-10 years', '10+ years'], 
          loc='center left', 
          ncol=1, fontsize=12, frameon=True)

# Adjust layout to prevent overlap and make room for legends

# Show plot
plt.savefig('Age.png', dpi=300)
plt.show()