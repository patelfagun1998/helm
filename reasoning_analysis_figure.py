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
                                      show_yticklabels=True, return_legend_handles=False):
    """
    Plots a horizontal grouped bar chart on the given axes for reasoning analysis.
    """
    num_models = len(models)
    # Add spacing between model groups by using a larger multiplier
    spacing_factor = 1.5  # Increase spacing between model groups
    y_pos = np.arange(num_models) * spacing_factor  # Positions for each model group on the y-axis

    # Two categories - original positioning
    bar1 = ax.barh(y_pos - bar_height / 2, scores1, height=bar_height,
                   label=label1, color=color1, align='center')
    bar2 = ax.barh(y_pos + bar_height / 2, scores2, height=bar_height,
                   label=label2, color=color2, align='center')
    legend_handles = [bar1, bar2]

    # Set y-axis ticks and labels (model names)
    ax.set_yticks(y_pos)
    if show_yticklabels:
        ax.set_yticklabels(models, fontsize=5)
    else:
        ax.set_yticklabels([''] * len(models))

    ax.invert_yaxis()  # Display models from top to bottom
    ax.tick_params(axis='y', labelsize=6)

    # Set x-axis label and limits
    ax.set_xlabel(x_label, fontsize=5)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=5)

    # Set title for the subplot
    ax.set_title(title, fontsize=5, fontweight='bold')

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

# --- Data for Reasoning Analysis ---
reasoning_data = {
    'gpt-4o-audio': {'type': 0.37, 'reasoning_type': 0.21, 'symp': 0.49, 'reasoning_symp': 0.24},
    'gemini-2.0-flash': {'type': 0.36, 'reasoning_type': 0.46, 'symp': 0.22, 'reasoning_symp': 0.19},
    'gemini-2.0-flash-lite': {'type': 0.19, 'reasoning_type': 0.25, 'symp': 0.43, 'reasoning_symp': 0.25},
    'gpt-4o-mini-audio': {'type': 0.15, 'reasoning_type': 0.14, 'symp': 0.39, 'reasoning_symp': 0.04},
}

# Extract model names
models = list(reasoning_data.keys())

# Color scheme for reasoning analysis
color_reasoning = ['#2CA4A2', '#D84546']  # Zero-shot vs Reasoning

# Extract data arrays
type = [reasoning_data[model]['type'] for model in models]
reasoning_type = [reasoning_data[model]['reasoning_type'] for model in models]
symp = [reasoning_data[model]['symp'] for model in models]
reasoning_symp = [reasoning_data[model]['reasoning_symp'] for model in models]

# Convert to numpy arrays
type = np.array(type)
reasoning_type = np.array(reasoning_type)
symp = np.array(symp)
reasoning_symp = np.array(reasoning_symp)

# Plot parameters
bar_height = 0.42
x_lim = 0.6

# --- Create the figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(4, 3))

# Plot 1: Disorder Type Diagnosis
handles1 = plot_horizontal_grouped_bar_chart(
    axes[0], models, type, reasoning_type, 
    'Zero-Shot', 'Zero-Shot w/ Reasoning',
    color_reasoning[0], color_reasoning[1], 
    'Disorder Type Diagnosis', 'Exact Match Score ↑', x_lim, bar_height,
    show_yticklabels=True, return_legend_handles=True
)

# Plot 2: Symptom Diagnosis
handles2 = plot_horizontal_grouped_bar_chart(
    axes[1], models, symp, reasoning_symp, 
    'Zero-Shot Inference', '0-Shot Inference w/ Reasoning',
    color_reasoning[0], color_reasoning[1], 
    'Symptom Diagnosis', 'Exact Match Score ↑', x_lim, bar_height,
    show_yticklabels=False, return_legend_handles=True
)

# Create legend for the figure
labels = ['Zero-Shot', 'Zero-Shot w/ Reasoning']
fig.legend(handles1, labels, 
           loc='center right', bbox_to_anchor=(0.5, 0.02), 
           ncol=2, fontsize=4, frameon=True)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.88)  # Make room for title and legend

# Save and show the figure
plt.savefig('reasoning_analysis_figure.png', dpi=300, bbox_inches='tight')
plt.show() 