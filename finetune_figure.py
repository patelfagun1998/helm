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

# Sample data for each subplot (updated with actual data from table)
# Subplot 1: Disorder Diagnosis (Micro F1 Score)
disorder_diagnosis_data = {
    'Base Model': [0.455, 0.556, 0.460],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct
    'Finetuned w/o Markers': [0.482, 0.541, 0.366],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (no asterisk)
    'Finetuned w/ Markers': [0.504, 0.540, 0.314]   # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (with asterisk)
}

# Subplot 2: Transcription Accuracy (Word Error Rate - lower is better)
transcription_accuracy_data = {
    'Base Model': [2.084, 5.346, 2.449],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct
    'Finetuned w/o Markers': [1.762, 0.996, 0.572],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (no asterisk)
    'Finetuned w/ Markers': [1.206, 1.036, 0.574]   # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (with asterisk)
}

# Subplot 3: Disorder Type Diagnosis (Micro F1 Score)
disorder_type_data = {
    'Base Model': [0.347, 0.394, 0.284],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct
    'Finetuned w/o Markers': [0.413, 0.364, 0.271],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (no asterisk)
    'Finetuned w/ Markers': [0.390, 0.386, 0.207]   # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (with asterisk)
}

# Subplot 4: Disorder Symptom Diagnosis (Micro F1 Score)  
disorder_symptom_data = {
    'Base Model': [0.163, 0.155, 0.073],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct
    'Finetuned w/o Markers': [0.355, 0.158, 0.080],  # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (no asterisk)
    'Finetuned w/ Markers': [0.262, 0.155, 0.073]   # qwen2.5-omni-7b, qwen2.5-omni-3b, qwen2-audio-7b-instruct (with asterisk)
}

# Create figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot data for each subplot
subplot_data = [
    (disorder_diagnosis_data, 'Disorder Diag.', 'Micro F1 Score ↑', (0.0, 1.0)),
    (transcription_accuracy_data, 'Transcription Accuracy', 'Word Error Rate ↓', (0, 6)),
    (disorder_type_data, 'Disorder Type Diag.', 'Micro F1 Score ↑', (0.0, 1.0)),
    (disorder_symptom_data, 'Symptom Diag.', 'Micro F1 Score ↑', (0.0, 1.0))
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
        ax.set_yticklabels(models, fontsize=12)
    else:
        ax.set_yticklabels([])
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlim(xlim)
    ax.grid(axis='x', alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)  # Increase x-axis tick label font size

# Add common legend at bottom center
fig.legend(conditions, bbox_to_anchor=(0.5, 0.09), loc='upper center', ncol=3, frameon=True, fancybox=True, shadow=True, fontsize=10)

# Adjust layout to prevent overlap and make room for legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.savefig('Finetuning.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Save the figure
# plt.savefig('disorder_diagnosis_results.pdf', dpi=300, bbox_inches='tight') 