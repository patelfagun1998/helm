import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed.
from tueplots import bundles
from tueplots import figsizes

# Apply tueplots styling (e.g., NeurIPS 2024 style)
plt.rcParams.update(bundles.neurips2024())
# Disable LaTeX for text rendering if not needed or to avoid setup issues
plt.rcParams['text.usetex'] = False

# --- Helper function to plot a horizontal grouped bar chart ---
def plot_horizontal_grouped_bar_chart(ax, models, scores1, scores2, label1, label2,
                                      color1, color2, title, x_label, xlim_upper, bar_height,
                                      show_yticklabels=True, return_legend_handles=False,
                                      scores3=None, label3=None, color3=None):
    """
    Plots a horizontal grouped bar chart on the given axes.
    Can handle 2 or 3 score categories.
    """
    num_models = len(models)
    y_pos = np.arange(num_models)  # Positions for each model group on the y-axis

    if scores3 is not None:
        # Three categories - use smaller bar height and adjust positions
        bar_height_3way = bar_height * 0.7  # Reduce bar thickness for 3-way comparison
        bar1 = ax.barh(y_pos - bar_height_3way, scores1, height=bar_height_3way,
                       label=label1, color=color1, align='center')
        bar2 = ax.barh(y_pos, scores2, height=bar_height_3way,
                       label=label2, color=color2, align='center')
        bar3 = ax.barh(y_pos + bar_height_3way, scores3, height=bar_height_3way,
                       label=label3, color=color3, align='center')
        legend_handles = [bar1, bar2, bar3]
    else:
        # Two categories - original positioning
        bar1 = ax.barh(y_pos - bar_height / 2, scores1, height=bar_height,
                       label=label1, color=color1, align='center')
        bar2 = ax.barh(y_pos + bar_height / 2, scores2, height=bar_height,
                       label=label2, color=color2, align='center')
        legend_handles = [bar1, bar2]

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
    
    if return_legend_handles:
        return legend_handles

# --- Data Organization with Dictionary Structure ---
model_data = {
    'gpt-4o-audio': {
        'reasoning': {'type_f1': 0.36, 'reasoning_type_f1': 0.20, 'symp_f1': 0.18, 'reasoning_symp_f1': 0.19},
        'perturbation': {'type_f1': 0.36, 'noisy_type_f1': 0.29, 'symp_f1': 0.18, 'noisy_symp_f1': 0.18},
        'gender': {'male_binary_f1': 0.58, 'female_binary_f1': 0.30, 'male_type_f1': 0.41, 'female_type_f1': 0.24},
        'language': {'english_f1': 0.16, 'french_f1': 0.30, 'dutch_f1': 0.33, 'english_wer': 2.25, 'french_wer': 4.45, 'dutch_wer': 5.66}
    },
    'gemini-2.0-flash': {
        'reasoning': {'type_f1': 0.46, 'reasoning_type_f1': 0.34, 'symp_f1': 0.18, 'reasoning_symp_f1': 0.25},
        'perturbation': {'type_f1': 0.46, 'noisy_type_f1': 0.33, 'symp_f1': 0.18, 'noisy_symp_f1': 0.18},
        'gender': {'male_binary_f1': 0.564, 'female_binary_f1': 0.30, 'male_type_f1': 0.44, 'female_type_f1': 0.24},
        'language': {'english_f1': 0.33, 'french_f1': 0.49, 'dutch_f1': 0.35, 'english_wer': 0.94, 'french_wer': 5.9, 'dutch_wer': 4.7}
    },
    'gemini-2.0-flash-lite': {
        'reasoning': {'type_f1': 0.19, 'reasoning_type_f1': 0.17, 'symp_f1': 0.41, 'reasoning_symp_f1': 0.19},
        'perturbation': {'type_f1': 0.19, 'noisy_type_f1': 0.14, 'symp_f1': 0.41, 'noisy_symp_f1': 0.4},
        'gender': {'male_binary_f1': 0.56, 'female_binary_f1': 0.31, 'male_type_f1': 0.24, 'female_type_f1': 0.09},
        'language': {'english_f1': 0.34, 'french_f1': 0.27, 'dutch_f1': 0.49, 'english_wer': 0.83, 'french_wer': 4.95, 'dutch_wer': 3.27}
    },
    'gpt-4o-mini-audio': {
        'reasoning': {'type_f1': 0.15, 'reasoning_type_f1': 0.04, 'symp_f1': 0.43, 'reasoning_symp_f1': 0.006},
        'perturbation': {'type_f1': 0.15, 'noisy_type_f1': 0.05, 'symp_f1': 0.43, 'noisy_symp_f1': 0.42},
        'gender': {'male_binary_f1': 0.22, 'female_binary_f1': 0.16, 'male_type_f1': 0.17, 'female_type_f1': 0.11},
        'language': {'english_f1': 0.21, 'french_f1': 0.10, 'dutch_f1': 0.02, 'english_wer': 1.4, 'french_wer': 5.27, 'dutch_wer': 5.94}
    },
    'gpt-4o-audio-transcribe': {
        'reasoning': {'type_f1': 0.41, 'reasoning_type_f1': 0.30, 'symp_f1': 0.31, 'reasoning_symp_f1': 0.39},
        'perturbation': {'type_f1': 0.41, 'noisy_type_f1': 0.36, 'symp_f1': 0.31, 'noisy_symp_f1': 0.30},
        'gender': {'male_binary_f1': 0.58, 'female_binary_f1': 0.34, 'male_type_f1': 0.49, 'female_type_f1': 0.34},
        'language': {'english_f1': 0.47, 'french_f1': 0.49, 'dutch_f1': 0.48, 'english_wer': 1.31, 'french_wer': 4.27, 'dutch_wer': 2.73}
    },
    'gpt-4o-audio-mini-transcribe': {
        'reasoning': {'type_f1': 0.42, 'reasoning_type_f1': 0.34, 'symp_f1': 0.31, 'reasoning_symp_f1': 0.35},
        'perturbation': {'type_f1': 0.42, 'noisy_type_f1': 0.34, 'symp_f1': 0.31, 'noisy_symp_f1': 0.31},
        'gender': {'male_binary_f1': 0.56, 'female_binary_f1': 0.29, 'male_type_f1': 0.5, 'female_type_f1': 0.34},
        'language': {'english_f1': 0.56, 'french_f1': 0.49, 'dutch_f1': 0.48, 'english_wer': 1.54, 'french_wer': 4.41, 'dutch_wer': 2.67}
    },
    'whispr+gpt4o': {
        'reasoning': {'type_f1': 0.43, 'reasoning_type_f1': 0.32, 'symp_f1': 0.36, 'reasoning_symp_f1': 0.44},
        'perturbation': {'type_f1': 0.43, 'noisy_type_f1': 0.48, 'symp_f1': 0.36, 'noisy_symp_f1': 0.40},
        'gender': {'male_binary_f1': 0.29, 'female_binary_f1': 0.35, 'male_type_f1': 0.512, 'female_type_f1': 0.38},
        'language': {'english_f1': 0.37, 'french_f1': 0.36, 'dutch_f1': 0.16, 'english_wer': 2.87, 'french_wer': 7.51, 'dutch_wer': 7.16}
    }
}

# Extract model names
models = list(model_data.keys())

# Color schemes
color_reasoning = ['#2CA4A2', '#D84546']  # Zero-shot vs Reasoning
color_perturbation = ['#1f77b4', '#ff7f0e']  # Unperturbed vs Perturbed
color_gender = ['#ea5545', '#87bc45']  # Male vs Female
color_language = ['#edbf33', '#f46a9b', '#27aeef']  # English vs French vs Dutch

bar_height = 0.42
x_lim = 0.6
xlim_upper_wer = 8.0

# Helper function to extract data arrays from the dictionary
def extract_data_arrays(models, data_dict, *keys):
    """Extract data arrays for specified keys from the model data dictionary."""
    arrays = []
    for key in keys:
        array = []
        for model in models:
            # Navigate through nested dictionary structure
            current_data = data_dict[model]
            for k in key.split('.'):
                current_data = current_data[k]
            array.append(current_data)
        arrays.append(np.array(array))
    return arrays if len(arrays) > 1 else arrays[0]

# Extract data arrays for plotting
type_f1, reasoning_type_f1 = extract_data_arrays(models, model_data, 'reasoning.type_f1', 'reasoning.reasoning_type_f1')
symp_f1, reasoning_symp_f1 = extract_data_arrays(models, model_data, 'reasoning.symp_f1', 'reasoning.reasoning_symp_f1')

type_f1_pert, noisy_type_f1 = extract_data_arrays(models, model_data, 'perturbation.type_f1', 'perturbation.noisy_type_f1')
symp_f1_pert, noisy_symp_f1 = extract_data_arrays(models, model_data, 'perturbation.symp_f1', 'perturbation.noisy_symp_f1')

male_binary_f1, female_binary_f1 = extract_data_arrays(models, model_data, 'gender.male_binary_f1', 'gender.female_binary_f1')
male_type_f1, female_type_f1 = extract_data_arrays(models, model_data, 'gender.male_type_f1', 'gender.female_type_f1')

english_scores_f1, french_scores_f1 = extract_data_arrays(models, model_data, 'language.english_f1', 'language.french_f1')
english_scores_wer, french_scores_wer = extract_data_arrays(models, model_data, 'language.english_wer', 'language.french_wer')

# Extract Dutch data arrays
dutch_scores_f1 = extract_data_arrays(models, model_data, 'language.dutch_f1')
dutch_scores_wer = extract_data_arrays(models, model_data, 'language.dutch_wer')

# --- Create the figure and subplots ---
fig_width, fig_height = 22, 6  # Made slightly taller to accommodate legends
fig, axes = plt.subplots(1, 8, figsize=(fig_width, fig_height))

# Plot configurations
plot_configs = [
    # Reasoning Analysis
    (type_f1, reasoning_type_f1, 'Zero-Shot', 'Zero-Shot w/ Reasoning', 
     color_reasoning, 'Disorder Type Diagnosis', 'Exact Match Score ↑', x_lim, True, None, None, None),
    (symp_f1, reasoning_symp_f1, 'Zero-Shot Inference', '0-Shot Inference w/ Reasoning', 
     color_reasoning, 'Symptom Diagnosis', 'Exact Match Score ↑', x_lim, False, None, None, None),
    
    # Perturbation Analysis
    (type_f1_pert, noisy_type_f1, 'Unperturbed', 'Perturbed', 
     color_perturbation, 'Disorder Type Diagnosis', 'Micro F1 Score ↑', x_lim, False, None, None, None),
    (symp_f1_pert, noisy_symp_f1, 'Unperturbed', 'Perturbed', 
     color_perturbation, 'Symptom Diagnosis', 'Micro F1 Score ↑', x_lim, False, None, None, None),
    
    # Gender Analysis
    (male_binary_f1, female_binary_f1, 'Male', 'Female', 
     color_gender, 'Disorder Diagnosis', 'Micro F1 Score ↑', x_lim, False, None, None, None),
    (male_type_f1, female_type_f1, 'Male', 'Female', 
     color_gender, 'Disorder Type Diagnosis', 'Micro F1 Score ↑', x_lim, False, None, None, None),
    
    # Language Analysis (3 languages)
    (english_scores_f1, french_scores_f1, 'English', 'French', 
     color_language, 'Disorder Diagnosis', 'Macro F1 Score ↑', x_lim, False, dutch_scores_f1, 'Dutch', color_language[2]),
    (english_scores_wer, french_scores_wer, 'English', 'French', 
     color_language, 'Transcription Accuracy', 'Word Error Rate (WER) ↓', xlim_upper_wer, False, dutch_scores_wer, 'Dutch', color_language[2]),
]

# Store legend handles for each unique color scheme
legend_handles = {}
legend_labels = {}

# Plot all subplots
for i, (scores1, scores2, label1, label2, colors, title, x_label, xlim, show_y, scores3, label3, color3) in enumerate(plot_configs):
    handles = plot_horizontal_grouped_bar_chart(
        axes[i], models, scores1, scores2, label1, label2,
        colors[0], colors[1], title, x_label, xlim, bar_height,
        show_yticklabels=show_y, return_legend_handles=True,
        scores3=scores3, label3=label3, color3=color3
    )
    
    # Store unique legend handles
    if scores3 is not None:
        # Three-category plot
        color_key = tuple(colors)
        if color_key not in legend_handles:
            legend_handles[color_key] = handles
            legend_labels[color_key] = [label1, label2, label3]
    else:
        # Two-category plot
        color_key = tuple(colors)
        if color_key not in legend_handles:
            legend_handles[color_key] = handles
            legend_labels[color_key] = [label1, label2]

# Create legends at the bottom of the figure
legend_positions = [0.125, 0.375, 0.625, 0.875]  # Centered positions for 4 legends
legend_keys = list(legend_handles.keys())

for i, (pos, key) in enumerate(zip(legend_positions, legend_keys)):
    if i < len(legend_keys):
        labels = legend_labels[key]
        ncol = len(labels)  # Use number of labels for ncol
        fig.legend(legend_handles[key], labels, 
                  loc='lower center', bbox_to_anchor=(pos, 0.02), 
                  ncol=ncol, fontsize=12, frameon=True)

# Adjust layout to prevent overlap and make room for legends
plt.tight_layout(pad=1.5, w_pad=2.5)
plt.subplots_adjust(bottom=0.18)  # Make more room for larger legends at bottom

# Show plot
plt.show()

# --- Create 4 Separate Figures with 2 Subplots Each ---
figure_titles = [
    'Reasoning Analysis',
    'Perturbation Analysis', 
    'Gender Analysis',
    'Language Analysis'
]

# Group plot configurations into pairs
plot_pairs = [
    plot_configs[0:2],  # Reasoning analysis (2 plots)
    plot_configs[2:4],  # Perturbation analysis (2 plots)
    plot_configs[4:6],  # Gender analysis (2 plots)
    plot_configs[6:8],  # Language analysis (2 plots)
]

# Create 4 separate figures
for fig_idx, (fig_title, plot_pair) in enumerate(zip(figure_titles, plot_pairs)):
    # Create figure with 2 subplots
    fig_pair, axes_pair = plt.subplots(1, 2, figsize=(16, 6))
    
    # Store legend handles for this figure
    pair_legend_handles = {}
    pair_legend_labels = {}
    
    # Plot both subplots in this figure
    for i, (scores1, scores2, label1, label2, colors, title, x_label, xlim, show_y, scores3, label3, color3) in enumerate(plot_pair):
        # Determine which subplot to show y-axis labels on (first one only)
        show_yticklabels = (i == 0)
        
        handles = plot_horizontal_grouped_bar_chart(
            axes_pair[i], models, scores1, scores2, label1, label2,
            colors[0], colors[1], title, x_label, xlim, bar_height,
            show_yticklabels=show_yticklabels, return_legend_handles=True,
            scores3=scores3, label3=label3, color3=color3
        )
        
        # Store legend handles for this figure
        if scores3 is not None:
            # Three-category plot
            color_key = tuple(colors)
            if color_key not in pair_legend_handles:
                pair_legend_handles[color_key] = handles
                pair_legend_labels[color_key] = [label1, label2, label3]
        else:
            # Two-category plot
            color_key = tuple(colors)
            if color_key not in pair_legend_handles:
                pair_legend_handles[color_key] = handles
                pair_legend_labels[color_key] = [label1, label2]
    
    # Create legend for this figure
    legend_keys = list(pair_legend_handles.keys())
    if len(legend_keys) > 0:
        key = legend_keys[0]  # Should only be one unique color scheme per figure pair
        labels = pair_legend_labels[key]
        ncol = len(labels)
        fig_pair.legend(pair_legend_handles[key], labels, 
                       loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                       ncol=ncol, fontsize=12, frameon=True)
    
    # Set overall figure title
    fig_pair.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)  # Make room for title and legend
    
    # Show the figure
    plt.show()