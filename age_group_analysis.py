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
    y_pos = np.arange(num_models)  # Positions for each model group on the y-axis

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
        ax.set_yticklabels(models, fontsize=8.5)
    else:
        ax.set_yticklabels([''] * len(models))

    ax.invert_yaxis()  # Display models from top to bottom
    ax.tick_params(axis='y', labelsize=8.5)

    # Set x-axis label and limits
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=8.5)

    # Set title for the subplot
    ax.set_title(title, fontsize=10, fontweight='bold')

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
        'symptom_classification': {'5 - 7 y/o': 0.42, '8 - 10 y/o': 0.511, '10+ y/o': 0.60},
        'transcription_wer': {'5 - 7 y/o': 5.63, '8 - 10 y/o': 6.05, '10+ y/o': 4.16}
    },
    'gemini-2.0-flash': {
        'symptom_classification': {'5 - 7 y/o': 0.21, '8 - 10 y/o': 0.18, '10+ y/o': 0.13},
        'transcription_wer': {'5 - 7 y/o': 4.34, '8 - 10 y/o': 4.45, '10+ y/o': 2.96}
    },
    'gemini-2.0-flash-lite': {
        'symptom_classification': {'5 - 7 y/o': 0.32, '8 - 10 y/o': 0.43, '10+ y/o': 0.49},
        'transcription_wer': {'5 - 7 y/o': 3.39, '8 - 10 y/o': 3.45, '10+ y/o': 2.65}
    },
    'gpt-4o-mini-audio': {
        'symptom_classification': {'5 - 7 y/o': 0.32, '8 - 10 y/o': 0.416, '10+ y/o': 0.57},
        'transcription_wer': {'5 - 7 y/o': 4.4, '8 - 10 y/o': 5.2, '10+ y/o': 5.0}
    },
    'gpt-4o-audio-transcribe': {
        'symptom_classification': {'5 - 7 y/o': 0.34, '8 - 10 y/o': 0.34, '10+ y/o': 0.31},
        'transcription_wer': {'5 - 7 y/o': 4.27, '8 - 10 y/o': 4.67, '10+ y/o': 3.34}
    },
    'gpt-4o-audio-mini-transcribe': {
        'symptom_classification': {'5 - 7 y/o': 0.32, '8 - 10 y/o': 0.31, '10+ y/o': 0.27},
        'transcription_wer': {'5 - 7 y/o': 5.4, '8 - 10 y/o': 5.46, '10+ y/o': 3.68}
    },
    'whispr+gpt4o': {
        'symptom_classification': {'5 - 7 y/o': 0.33, '8 - 10 y/o': 0.4, '10+ y/o': 0.4},
        'transcription_wer': {'5 - 7 y/o': 10.89, '8 - 10 y/o': 10.26, '10+ y/o': 4.86}
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
symptom_scores = extract_age_group_data(models, age_group_data, 'symptom_classification', age_group_keys)
transcription_wer = extract_age_group_data(models, age_group_data, 'transcription_wer', age_group_keys)

# Color scheme for the 3 age groups (matching extra_analysis_figure style)
colors = ['#edbf33', '#f46a9b', '#27aeef']  # Yellow, Pink, Blue - similar to language colors

# Chart parameters
bar_height = 0.42
x_lim_symptom = 0.8
x_lim_wer = 7.0

# --- Create the figure with two subplots ---
fig_width, fig_height = 16, 6
fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

# --- Subplot 1: Symptom Classification Scores ---
plot_horizontal_grouped_bar_chart(
    axes[0], models, 
    symptom_scores[:, 0], symptom_scores[:, 1], symptom_scores[:, 2],
    '5-7 years', '8-10 years', '10+ years',
    colors[0], colors[1], colors[2],
    'Symptom Classification', 'Micro F1 Score ↑', x_lim_symptom, bar_height,
    show_yticklabels=True
)

# --- Subplot 2: Transcription Accuracy (WER) ---
plot_horizontal_grouped_bar_chart(
    axes[1], models,
    transcription_wer[:, 0], transcription_wer[:, 1], transcription_wer[:, 2],
    '5-7 years', '8-10 years', '10+ years',
    colors[0], colors[1], colors[2],
    'Transcription Accuracy', 'Word Error Rate (WER) ↓', x_lim_wer, bar_height,
    show_yticklabels=False
)

# Create legend at the bottom of the figure
fig.legend(['5-7 years', '8-10 years', '10+ years'], 
          loc='lower center', bbox_to_anchor=(0.5, 0.02), 
          ncol=3, fontsize=12, frameon=True)

# Adjust layout to prevent overlap and make room for legends
plt.tight_layout(pad=1.5, w_pad=2.5)
plt.subplots_adjust(bottom=0.18)  # Make room for legend at bottom

# Show plot
plt.savefig('age_group_analysis.png', dpi=300)

plt.show()

# --- Print summary statistics ---
print("\n" + "="*80)
print("AGE GROUP ANALYSIS SUMMARY - SYMPTOM CLASSIFICATION")
print("="*80)
print(f"{'Model':<25} {'5-7 years':<12} {'8-10 years':<12} {'10+ years':<12}")
print("-"*80)
for i, model in enumerate(models):
    print(f"{model:<25} {symptom_scores[i,0]:<12.3f} {symptom_scores[i,1]:<12.3f} {symptom_scores[i,2]:<12.3f}")

print("\n" + "="*80)
print("AGE GROUP ANALYSIS SUMMARY - TRANSCRIPTION ACCURACY (WER)")
print("="*80)
print(f"{'Model':<25} {'5-7 years':<12} {'8-10 years':<12} {'10+ years':<12}")
print("-"*80)
for i, model in enumerate(models):
    print(f"{model:<25} {transcription_wer[i,0]:<12.1f} {transcription_wer[i,1]:<12.1f} {transcription_wer[i,2]:<12.1f}")

# Best performing models per age group
print(f"\n{'='*50}")
print("BEST PERFORMING MODELS BY AGE GROUP")
print("="*50)
print("Symptom Classification (highest scores):")
age_group_names = ['5-7 years', '8-10 years', '10+ years']
for i, age_group in enumerate(age_group_names):
    best_model_idx = np.argmax(symptom_scores[:, i])
    print(f"  {age_group}: {models[best_model_idx]} ({symptom_scores[best_model_idx, i]:.3f})")

print("\nTranscription Accuracy (lowest WER):")
for i, age_group in enumerate(age_group_names):
    best_model_idx = np.argmin(transcription_wer[:, i])
    print(f"  {age_group}: {models[best_model_idx]} ({transcription_wer[best_model_idx, i]:.1f})")