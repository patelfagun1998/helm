import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed.
from tueplots import bundles
from tueplots import figsizes

# Apply tueplots styling (e.g., NeurIPS 2024 style)
plt.rcParams.update(bundles.neurips2024())
# Disable LaTeX for text rendering if not needed or to avoid setup issues
plt.rcParams['text.usetex'] = False

# --- Helper function to plot a horizontal bar chart ---
def plot_horizontal_bar_chart(ax, models, scores, title, x_label, xlim_upper, bar_height, color):
    """
    Plots a horizontal bar chart on the given axes for a single language.
    """
    num_models = len(models)
    # Add spacing between model groups by using a larger multiplier
    spacing_factor = 1.5  # Increase spacing between model groups
    y_pos = np.arange(num_models) * spacing_factor  # Positions for each model group on the y-axis

    # Create horizontal bars
    bars = ax.barh(y_pos, scores, height=bar_height, color=color, align='center')

    # Set y-axis ticks and labels (model names)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=8.5)
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

# --- Data for Taiwanese and Cantonese ---
# Note: These are placeholder values - replace with actual data
model_data = {
    'gpt-4o-audio': {
        'taiwanese_micro_f1': 0.007,
        'cantonese_micro_f1': 0.001
    },
    'gemini-2.0-flash': {
        'taiwanese_micro_f1': 0.013,
        'cantonese_micro_f1': 0
    },
    'gemini-2.0-flash-lite': {
        'taiwanese_micro_f1': 0.08,
        'cantonese_micro_f1': 0.01
    },
    'gpt-4o-mini-audio': {
        'taiwanese_micro_f1': 0.0,
        'cantonese_micro_f1': 0
    },
    'gpt-4o-audio-transcribe': {
        'taiwanese_micro_f1': 0.1,
        'cantonese_micro_f1': 0.03
    },
    'gpt-4o-audio-mini-transcribe': {
        'taiwanese_micro_f1': 0.08,
        'cantonese_micro_f1': 0.02
    },
    'whispr+gpt4o': {
        'taiwanese_micro_f1': 0.07,
        'cantonese_micro_f1': 0.02
    }
}

# Extract model names
models = list(model_data.keys())

# Extract data arrays
taiwanese_scores = np.array([model_data[model]['taiwanese_micro_f1'] for model in models])
cantonese_scores = np.array([model_data[model]['cantonese_micro_f1'] for model in models])

# Colors for each language
color_taiwanese = '#27aeef'  # Blue
color_cantonese = '#ef476f'  # Pink/Red

# Plot configuration
bar_height = 0.6
x_lim = 0.5

# --- Create the figure with 2 subplots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot Taiwanese results
plot_horizontal_bar_chart(
    axes[0], models, taiwanese_scores, 
    'Taiwanese - Disorder Diagnosis', 'Micro F1 Score ↑', 
    x_lim, bar_height, color_taiwanese
)

# Plot Cantonese results
plot_horizontal_bar_chart(
    axes[1], models, cantonese_scores, 
    'Cantonese - Disorder Diagnosis', 'Micro F1 Score ↑', 
    x_lim, bar_height, color_cantonese
)

# Remove y-axis labels from the second subplot to avoid duplication
axes[1].set_yticklabels([''] * len(models))

# Set overall figure title
fig.suptitle('Disorder Diagnosis Performance: Taiwanese vs Cantonese', fontsize=14, fontweight='bold', y=0.95)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Make room for title

# Save and show the figure
plt.savefig('taiwanese_cantonese_analysis.png', dpi=300)
plt.show() 