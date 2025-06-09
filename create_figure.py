import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed in your environment.
# You can typically install it with: pip install tueplots
from tueplots import bundles
from tueplots import figsizes # Good practice to import

plt.rcParams.update(bundles.neurips2024())
plt.rcParams['text.usetex'] = False

# Data for the plots
models = ['qwen2.5-omni-7b', 'qwen2.5-omni-3b', 'qwen2-audio-7b-instruct']
conditions = ['Base Model', 'Finetuned w/o Markers', 'Finetuned w/ Markers']

# Define colors for each condition
colors = ['#d62728', '#2ca02c', '#ff7f0e']  # Red, Green, Orange

# Sample data for each subplot (you should replace with actual data)
# Subplot 1: Disorder Diagnosis (Micro F1 Score)
disorder_diagnosis_data = {
    'Base Model': [0.71, 0.42, 0.45],
    'Finetuned w/o Markers': [0.67, 0.25, 0.36], 
    'Finetuned w/ Markers': [0.95, 0.89, 0.30]
}

# Subplot 2: Transcription Accuracy (Word Error Rate - lower is better)
transcription_accuracy_data = {
    'Base Model': [2.17, 4.98, 2.3],
    'Finetuned w/o Markers': [1.76, 0.95, 0.52],
    'Finetuned w/ Markers': [1.4, 0.97, 0.58]
}

# Subplot 3: Disorder Type Diagnosis (Micro F1 Score)
disorder_type_data = {
    'Base Model': [0.71, 0.44, 0.33],
    'Finetuned w/o Markers': [0.40, 0.63, 0.27],
    'Finetuned w/ Markers': [0.95, 0.89, 0.21]
}

# Subplot 4: Disorder Symptom Diagnosis (Micro F1 Score)  
disorder_symptom_data = {
    'Base Model': [0.71, 0.44, 0.10],
    'Finetuned w/o Markers': [0.34, 0.16, 0.08],
    'Finetuned w/ Markers': [0.95, 0.90, 0.07]
}

# Create figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot data for each subplot
subplot_data = [
    (disorder_diagnosis_data, 'Disorder Diagnosis', 'Micro F1 Score ↑', (0.0, 1.0)),
    (transcription_accuracy_data, 'Transcription Accuracy', 'Word Error Rate ↓', (0, 6)),
    (disorder_type_data, 'Disorder Type Diagnosis', 'Micro F1 Score ↑', (0.0, 1.0)),
    (disorder_symptom_data, 'Disorder Symptom Diagnosis', 'Micro F1 Score ↑', (0.0, 1.0))
]

# Create horizontal bar plots
for idx, (data, title, xlabel, xlim) in enumerate(subplot_data):
    ax = axes[idx]
    
    # Calculate bar positions
    y_pos = np.arange(len(models))
    bar_width = 0.25
    
    # Plot bars for each condition
    for i, condition in enumerate(conditions):
        # Order bars as: Base Model (top), w/o Markers (middle), w/ Markers (bottom)
        offset = (1 - i) * bar_width  # This gives offsets: bar_width, 0, -bar_width
        bars = ax.barh(y_pos + offset, data[condition], bar_width, 
                      color=colors[i], label=condition, alpha=0.8)
    
    # Customize subplot
    ax.set_yticks(y_pos)
    # Only show y-axis labels on the leftmost plot
    if idx == 0:
        ax.set_yticklabels(models)
    else:
        ax.set_yticklabels([])
    
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend only to the first subplot, positioned outside the plot area
    if idx == 0:
        ax.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right', frameon=True, fancybox=True, shadow=True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Optional: Save the figure
# plt.savefig('disorder_diagnosis_results.pdf', dpi=300, bbox_inches='tight') 