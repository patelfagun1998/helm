import matplotlib.pyplot as plt
import numpy as np
# Import tueplots for styling. Ensure this library is installed in your environment.
# You can typically install it with: pip install tueplots
from tueplots import bundles
from tueplots import figsizes

plt.rcParams.update(bundles.neurips2024())
plt.rcParams['text.usetex'] = False

# Model-wise data structure - much cleaner and easier to maintain
MODEL_DATA = {
    'gpt-4o-audio': {
        'disorder_diagnosis': 0.44,
        'asr_disorder_diagnosis': 0.02,
        'disorder_type_classification': 0.36,
        'disorder_symptom_classification': 0.18,
        'transcription_accuracy': 2.25
    },
    'gemini-2.0-flash': {
        'disorder_diagnosis': 0.46,
        'asr_disorder_diagnosis': 0.00,
        'disorder_type_classification': 0.46,
        'disorder_symptom_classification': 0.18,
        'transcription_accuracy': 0.94
    },
    'gemini-2.0-flash-lite': {
        'disorder_diagnosis': 0.46,
        'asr_disorder_diagnosis': 0.01,
        'disorder_type_classification': 0.19,
        'disorder_symptom_classification': 0.41,
        'transcription_accuracy': 0.83
    },
    'gpt-4o-mini-audio': {
        'disorder_diagnosis': 0.20,
        'asr_disorder_diagnosis': 0.07,
        'disorder_type_classification': 0.15,
        'disorder_symptom_classification': 0.43,
        'transcription_accuracy': 1.40
    },
    'gpt-4o-mini-transcribe': {
        'disorder_diagnosis': 0.56,
        'asr_disorder_diagnosis': 0.21,
        'disorder_type_classification': 0.35,
        'disorder_symptom_classification': 0.31,
        'transcription_accuracy': 1.54
    },
    'gpt-4o-transcribe': {
        'disorder_diagnosis': 0.47,
        'asr_disorder_diagnosis': 0.13,
        'disorder_type_classification': 0.38,
        'disorder_symptom_classification': 0.31,
        'transcription_accuracy': 1.31
    },
    'whispr+gpt4o': {
        'disorder_diagnosis': 0.47,
        'asr_disorder_diagnosis': 0.00,
        'disorder_type_classification': 0.43,
        'disorder_symptom_classification': 0.36,
        'transcription_accuracy': 2.84
    },
    'qwen2.5-omni-7b': {
        'disorder_diagnosis': 0.71,
        'asr_disorder_diagnosis': 0.07,
        'disorder_type_classification': 0.71,
        'disorder_symptom_classification': 0.71,
        'transcription_accuracy': 2.71
    },
    'qwen2.5-omni-3b': {
        'disorder_diagnosis': 0.42,
        'asr_disorder_diagnosis': 0.01,
        'disorder_type_classification': 0.44,
        'disorder_symptom_classification': 0.44,
        'transcription_accuracy': 4.98
    },
    'qwen2-audio-7b': {
        'disorder_diagnosis': 0.45,
        'asr_disorder_diagnosis': 0.03,
        'disorder_type_classification': 0.33,
        'disorder_symptom_classification': 0.10,
        'transcription_accuracy': 2.29
    },
    'qwen-audio-chat': {
        'disorder_diagnosis': 0.00,
        'asr_disorder_diagnosis': 0.00,
        'disorder_type_classification': 0.00,
        'disorder_symptom_classification': 0.00,
        'transcription_accuracy': 12.3
    },
    'Phi-4': {
        'disorder_diagnosis': 0.32,
        'asr_disorder_diagnosis': 0.06,
        'disorder_type_classification': 0.37,
        'disorder_symptom_classification': 0.13,
        'transcription_accuracy': 6.36
    },
    'granite-speech-3.3-8b': {
        'disorder_diagnosis': 0.00,
        'asr_disorder_diagnosis': 0.03,
        'disorder_type_classification': 0.00,
        'disorder_symptom_classification': 0.00,
        'transcription_accuracy': 0.00
    },
    'granite-speech-3.2-8b': {
        'disorder_diagnosis': 0.00,
        'asr_disorder_diagnosis': 0.03,
        'disorder_type_classification': 0.00,
        'disorder_symptom_classification': 0.00,
        'transcription_accuracy': 13.50
    },
    'granite-speech-3.3-3b': {
        'disorder_diagnosis': 0.00,
        'asr_disorder_diagnosis': 0.04,
        'disorder_type_classification': 0.01,
        'disorder_symptom_classification': 0.20,
        'transcription_accuracy': 6.64
    }
}

# Scenario configuration
SCENARIOS = {
    'disorder_diagnosis': {
        'title': 'Disorder Diagnosis \n Micro F1 ↑',
        'xlim_upper': 0.75
    },
    'asr_disorder_diagnosis': {
        'title': 'ASR-Based Disorder Diagnosis \n Micro F1 ↑',
        'xlim_upper': 0.75
    },
    'disorder_type_classification': {
        'title': 'Disorder Type Classification \n Micro F1 ↑',
        'xlim_upper': 0.75
    },
    'disorder_symptom_classification': {
        'title': 'Disorder Symptom Classification \n Micro F1 ↑',
        'xlim_upper': 0.75
    },
    'transcription_accuracy': {
        'title': 'Transcription Accuracy \n WER ↓',
        'xlim_upper': 7.5
    }
}

def get_scenario_data(scenario_key, model_order=None):
    """
    Extract data for a specific scenario from the model-wise dictionary.
    
    Args:
        scenario_key: Key for the scenario (e.g., 'disorder_diagnosis')
        model_order: Optional list to specify model order. If None, uses default order.
    
    Returns:
        tuple: (models list, scores list)
    """
    if model_order is None:
        model_order = list(MODEL_DATA.keys())
    
    models = []
    scores = []
    
    for model in model_order:
        if model in MODEL_DATA and scenario_key in MODEL_DATA[model]:
            models.append(model)
            scores.append(MODEL_DATA[model][scenario_key])
    
    return models, scores

def get_sorted_model_order(reference_scenario='disorder_diagnosis'):
    """
    Get models sorted by their performance in the reference scenario.
    
    Args:
        reference_scenario: Scenario to use for sorting (default: 'disorder_diagnosis')
    
    Returns:
        list: Models sorted by performance in descending order
    """
    model_scores = []
    for model, data in MODEL_DATA.items():
        if reference_scenario in data:
            model_scores.append((model, data[reference_scenario]))
    
    # Sort by score in descending order
    model_scores.sort(key=lambda x: x[1], reverse=True)
    return [model for model, _ in model_scores]

def plot_scenario(ax, models, scores, title_text, xlim_upper, show_labels=True):
    """
    Helper function to create a horizontal bar chart for a scenario,
    styled to match the reference image and tueplots settings.

    Args:
        ax: Matplotlib axis object
        models: List of model names
        scores: List of corresponding scores
        title_text: Title for the subplot
        xlim_upper: Upper limit for x-axis
        show_labels: Whether to show the model names on the y-axis
    """
    num_models = len(models)
    y_pos = np.arange(num_models)

    # Bar color similar to the reference image (Matplotlib's default blue)
    bar_color = '#1f77b4'
    bar_height = 0.85

    # Plot bars
    bars = ax.barh(y_pos, scores, align='center', color=bar_color, height=bar_height)

    # Set y-axis ticks and labels (model names)
    ax.set_yticks(y_pos)
    if show_labels:
        ax.set_yticklabels(models, fontsize=8.5)
    else:
        ax.set_yticklabels([''] * len(models))  # Empty labels but keep ticks
    ax.invert_yaxis()  # To display models from top to bottom

    # Set x-axis label and limits
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=8.5)

    # Set title for the subplot
    ax.set_title(f"{title_text}", fontsize=10, fontweight='bold')

    # Add both horizontal and vertical gridlines
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)
    ax.set_axisbelow(True)  # Ensure grid is behind bars

    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

def create_comparison_plot():
    """
    Create the main comparison plot with all scenarios.
    """
    # Get sorted model order based on disorder diagnosis performance
    sorted_models = get_sorted_model_order('disorder_diagnosis')
    
    # Create subplots
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    
    # Plot each scenario
    scenario_keys = list(SCENARIOS.keys())
    for i, scenario_key in enumerate(scenario_keys):
        models, scores = get_scenario_data(scenario_key, sorted_models)
        scenario_config = SCENARIOS[scenario_key]
        
        plot_scenario(
            axes[i], 
            models, 
            scores,
            scenario_config['title'],
            scenario_config['xlim_upper'],
            show_labels=(i == 0)  # Show labels only for first plot
        )
    
    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=3.0)
    return fig

# Create and display the plot
fig = create_comparison_plot()
plt.show()