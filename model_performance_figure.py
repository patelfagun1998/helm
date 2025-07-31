import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed.
from tueplots import bundles
from tueplots import figsizes

# Apply tueplots styling (e.g., NeurIPS 2024 style)
plt.rcParams.update(bundles.neurips2024())
# Disable LaTeX for text rendering if not needed or to avoid setup issues
plt.rcParams['text.usetex'] = False

# --- Data for Model Performance Analysis ---
# Each category has 4 models with correct and incorrect counts
performance_data = {
    'subgoal_setting': {
        'gpt-4o': {'correct': 10, 'incorrect': 6},
        'gpt-4o-mini': {'correct': 10, 'incorrect': 7},
        'gemini-2.0-flash': {'correct': 0, 'incorrect': 0},
        'gemini-2.0-flash-lite': {'correct': 3, 'incorrect': 14}
    },
    'rule_following': {
        'gpt-4o': {'correct': 30, 'incorrect': 46},
        'gpt-4o-mini': {'correct': 13, 'incorrect': 8},
        'gemini-2.0-flash': {'correct': 38, 'incorrect': 38},
        'gemini-2.0-flash-lite': {'correct': 30, 'incorrect': 77}
    },
    'error_detection': {
        'gpt-4o': {'correct': 26, 'incorrect': 29},
        'gpt-4o-mini': {'correct': 12, 'incorrect': 5},
        'gemini-2.0-flash': {'correct': 22, 'incorrect': 25},
        'gemini-2.0-flash-lite': {'correct': 30, 'incorrect': 81}
    }
}

# Model names and categories
models = ['gpt-4o', 'gpt-4o-mini', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']
model_labels = ['GPT-4o', 'GPT-4o Mini', 'Gemini 2.0 Flash', 'Gemini 2.0 Flash Lite']  # Verbose labels
categories = ['Subgoal Setting', 'Rule Following', 'Error Detection']
category_keys = ['subgoal_setting', 'rule_following', 'error_detection']

# Colors for each model (correct/incorrect pairs)
model_colors = {
    'gpt-4o': {'correct': '#2E8B57', 'incorrect': '#8FBC8F'},  # Sea Green / Light Sea Green
    'gpt-4o-mini': {'correct': '#4169E1', 'incorrect': '#87CEEB'},  # Royal Blue / Sky Blue
    'gemini-2.0-flash': {'correct': '#FF6347', 'incorrect': '#FFA07A'},  # Tomato / Light Salmon
    'gemini-2.0-flash-lite': {'correct': '#9932CC', 'incorrect': '#DDA0DD'}  # Dark Orchid / Plum
}

# --- Helper function to create stacked vertical bar chart ---
def create_stacked_bar_chart():
    """
    Creates a stacked vertical bar chart showing correct/incorrect counts
    for each model across three categories.
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    
    # Parameters for layout
    bar_width = 0.6
    category_spacing = 1.5  # Space between categories
    model_spacing = 0.8   # Space between models within a category
    
    # Calculate positions for each bar
    x_positions = []
    
    current_x = 0
    for cat_idx, (category, cat_key) in enumerate(zip(categories, category_keys)):
        for model_idx, model in enumerate(models):
            x_positions.append(current_x)
            current_x += model_spacing
        
        # Add extra space after each category (except the last one)
        if cat_idx < len(categories) - 1:
            current_x += category_spacing
    
    # Extract data for plotting
    correct_counts = []
    incorrect_counts = []
    bar_colors_correct = []
    bar_colors_incorrect = []
    
    for cat_key in category_keys:
        for model in models:
            correct_counts.append(performance_data[cat_key][model]['correct'])
            incorrect_counts.append(performance_data[cat_key][model]['incorrect'])
            bar_colors_correct.append(model_colors[model]['correct'])
            bar_colors_incorrect.append(model_colors[model]['incorrect'])
    
    # Create stacked vertical bars with unique colors for each model
    bars_correct = ax.bar(x_positions, correct_counts, width=bar_width, 
                         label='Correct', color=bar_colors_correct, alpha=0.8)
    bars_incorrect = ax.bar(x_positions, incorrect_counts, width=bar_width, 
                           bottom=correct_counts, label='Incorrect', color=bar_colors_incorrect, alpha=0.8)
    
    # Remove x-axis tick labels (no model names on x-axis)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([''] * len(x_positions))  # Empty labels
    
    # Customize the plot
    # Add category labels integrated into the x-axis
    category_x_positions = []
    for cat_idx in range(len(categories)):
        start_idx = cat_idx * len(models)
        end_idx = start_idx + len(models) - 1
        category_x_pos = (x_positions[start_idx] + x_positions[end_idx]) / 2
        category_x_positions.append(category_x_pos)
    
    # Add category labels below the x-axis
    for i, category in enumerate(categories):
        ax.text(category_x_positions[i], -max(incorrect_counts) * 0.05, category, 
                ha='center', va='top', fontsize=4, fontweight='bold')
    
    # Set y-axis label and limits
    ax.set_ylabel('Count', fontsize=4)
    max_total = max([c + i for c, i in zip(correct_counts, incorrect_counts)])
    ax.set_ylim(0, max_total * 1.1)
    
    # Add gridlines
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.7, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (correct, incorrect, x_pos) in enumerate(zip(correct_counts, incorrect_counts, x_positions)):
        # Label for correct count
        if correct > 0:  # Only show label if count > 0
            ax.text(x_pos, correct/2, str(correct), ha='center', va='center', 
                    fontsize=3, fontweight='bold', color='white')
        # Label for incorrect count
        if incorrect > 0:  # Only show label if count > 0
            ax.text(x_pos, correct + incorrect/2, str(incorrect), ha='center', va='center', 
                    fontsize=3, fontweight='bold', color='white')
    
    # Customize spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Create custom legend with individual correct/incorrect labels
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    
    # Create legend with individual labels for correct/incorrect for each model
    legend_elements = []
    
    for model, model_label in zip(models, model_labels):
        # Create separate entries for correct and incorrect
        correct_color = model_colors[model]['correct']
        incorrect_color = model_colors[model]['incorrect']
        
        # Add correct entry
        legend_elements.append(
            mpatches.Rectangle((0, 0), 1, 1, 
                             facecolor=correct_color, 
                             alpha=0.8,
                             label=f'{model_label} (Correct)')
        )
        
        # Add incorrect entry
        legend_elements.append(
            mpatches.Rectangle((0, 0), 1, 1, 
                             facecolor=incorrect_color, 
                             alpha=0.8,
                             label=f'{model_label} (Incorrect)')
        )
    
    # Create the legend
    ax.legend(handles=legend_elements, loc='upper left', 
              fontsize=3, frameon=True, ncol=2)
    
    # Add vertical lines to separate categories
    for cat_idx in range(len(categories) - 1):
        separator_x = x_positions[(cat_idx + 1) * len(models) - 1] + category_spacing / 2
        ax.axvline(x=separator_x, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Adjust bottom margin to accommodate category labels
    plt.subplots_adjust(bottom=0.2)
    
    return fig, ax

# --- Create and display the figure ---
fig, ax = create_stacked_bar_chart()

# Adjust layout
plt.tight_layout()

# Save and show the figure
plt.savefig('model_performance_figure.png', dpi=300, bbox_inches='tight')
plt.show() 