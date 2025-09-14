import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed.
from tueplots import bundles
from tueplots import figsizes

# Apply tueplots styling (e.g., NeurIPS 2024 style)
plt.rcParams.update(bundles.neurips2024())
# Disable LaTeX for text rendering if not needed or to avoid setup issues
plt.rcParams['text.usetex'] = False

# --- Helper function to plot a combined horizontal bar chart ---
def plot_combined_horizontal_bar_chart(ax, models, taiwanese_scores, cantonese_scores, title, x_label, xlim_upper, bar_height):
    """
    Plots a combined horizontal bar chart showing both Taiwanese and Cantonese results.
    """
    num_models = len(models)
    # Reduced spacing between model groups for tighter layout
    spacing_factor = 1.0
    y_pos = np.arange(num_models) * spacing_factor
    
    # Colors for each language
    color_taiwanese = '#27aeef'  # Blue
    color_cantonese = '#ef476f'  # Pink/Red
    
    # Create horizontal bars for both languages side by side
    # Offset positions slightly to show both bars for each model
    offset = bar_height * 0.3  # Small offset to separate the bars
    
    bars_taiwanese = ax.barh(y_pos - offset/2, taiwanese_scores, height=bar_height*0.8, 
                            color=color_taiwanese, align='center', label='Taiwanese')
    bars_cantonese = ax.barh(y_pos + offset/2, cantonese_scores, height=bar_height*0.8, 
                            color=color_cantonese, align='center', label='Cantonese')

    # Set y-axis ticks and labels (model names) - increased font size
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=12)
    ax.invert_yaxis()  # Display models from top to bottom
    ax.tick_params(axis='y', labelsize=12)

    # Set x-axis label and limits - increased font size
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=11)

    # Set title for the plot - increased font size
    ax.set_title(title, fontsize=18, fontweight='bold')

    # Add gridlines (vertical gridlines along the x-axis)
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.7, axis='x')
    ax.set_axisbelow(True)  # Ensure grid is behind bars

    # Add legend
    ax.legend(fontsize=16, loc='lower right')

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

# Plot configuration - adjusted for better space utilization
bar_height = 0.7
x_lim = 0.12

# --- Create a single figure ---
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Plot combined results
plot_combined_horizontal_bar_chart(
    ax, models, taiwanese_scores, cantonese_scores,
    'Disorder Diagnosis Performance: Taiwanese vs Cantonese', 'Micro F1 Score ↑', 
    x_lim, bar_height
)

# Adjust layout with tighter spacing
plt.tight_layout()

# Save and show the figure
plt.savefig('taiwanese_cantonese_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
