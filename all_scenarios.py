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
    'gemini-2.0-flash-lite': {
        'disorder_diagnosis': 0.454,
        'asr_disorder_diagnosis': 0.4590,
        'disorder_type_classification': 0.180,
        'disorder_symptom_classification': 0.373,
        'transcription_accuracy': 0.83
    },
    'gemini-2.0-flash': {
        'disorder_diagnosis': 0.461,
        'asr_disorder_diagnosis': 0.4600,
        'disorder_type_classification': 0.298,
        'disorder_symptom_classification': 0.191,
        'transcription_accuracy': 0.931
    },
    'gpt-4o-mini-audio': {
        'disorder_diagnosis': 0.117,
        'asr_disorder_diagnosis': 0.4838,
        'disorder_type_classification': 0.131,
        'disorder_symptom_classification': 0.352,
        'transcription_accuracy': 2.92
    },
    'gpt-4o-audio': {
        'disorder_diagnosis': 0.458,
        'asr_disorder_diagnosis': 0.4590,
        'disorder_type_classification': 0.366,
        'disorder_symptom_classification': 0.538,
        'transcription_accuracy': 2.29
    },
    'gpt-4o-mini-transcribe': {
        'disorder_diagnosis': 0.466,
        'asr_disorder_diagnosis': 0.4600,
        'disorder_type_classification': 0.365,
        'disorder_symptom_classification': 0.297,
        'transcription_accuracy': 1.61
    },
    'gpt-4o-transcribe': {
        'disorder_diagnosis': 0.480,
        'asr_disorder_diagnosis': 0.4600,
        'disorder_type_classification': 0.381,
        'disorder_symptom_classification': 0.316,
        'transcription_accuracy': 1.267
    },
    'whispr+gpt4o': {
        'disorder_diagnosis': 0.484,
        'asr_disorder_diagnosis': 0.4611,
        'disorder_type_classification': 0.399,
        'disorder_symptom_classification': 0.378,
        'transcription_accuracy': 2.668
    },
    'qwen2.5-omni-7b': {
        'disorder_diagnosis': 0.455,
        'asr_disorder_diagnosis': 0.468,
        'disorder_type_classification': 0.347,
        'disorder_symptom_classification': 0.163,
        'transcription_accuracy': 2.084
    },
    'qwen2.5-omni-3b': {
        'disorder_diagnosis': 0.556,
        'asr_disorder_diagnosis': 0.459,
        'disorder_type_classification': 0.394,
        'disorder_symptom_classification': 0.155,
        'transcription_accuracy': 5.346
    },
    'qwen2-audio-7b': {
        'disorder_diagnosis': 0.460,
        'asr_disorder_diagnosis': 0.477,
        'disorder_type_classification': 0.284,
        'disorder_symptom_classification': 0.073,
        'transcription_accuracy': 2.449
    },
    'qwen-audio-chat': {
        'disorder_diagnosis': 0.000,
        'asr_disorder_diagnosis': 0.460,
        'disorder_type_classification': 0.000,
        'disorder_symptom_classification': 0.000,
        'transcription_accuracy': 9.549
    },
    'Phi-4': {
        'disorder_diagnosis': 0.552,
        'asr_disorder_diagnosis': 0.474,
        'disorder_type_classification': 0.318,
        'disorder_symptom_classification': 0.135,
        'transcription_accuracy': 2.288
    },
    'granite-speech-3.3-8b': {
        'disorder_diagnosis': 0.000,
        'asr_disorder_diagnosis': 0.473,
        'disorder_type_classification': 0.000,
        'disorder_symptom_classification': 0.000,
        'transcription_accuracy': 5.094
    },
    'granite-speech-3.3-3b': {
        'disorder_diagnosis': 0.000,
        'asr_disorder_diagnosis': 0.492,
        'disorder_type_classification': 0.000,
        'disorder_symptom_classification': 0.000,
        'transcription_accuracy': 2.564
    },
    'granite-speech-3.2-8b': {
        'disorder_diagnosis': 0.004,
        'asr_disorder_diagnosis': 0.484,
        'disorder_type_classification': 0.028,
        'disorder_symptom_classification': 0.268,
        'transcription_accuracy': 2.535
    }
}

# Scenario configuration
SCENARIOS = {
    'disorder_diagnosis': {
        'title': 'Disorder Diag. \n Micro F1 ↑',
        'xlim_upper': 0.9
    },
    'asr_disorder_diagnosis': {
        'title': 'ASR Disorder Diag. \n Micro F1 ↑',
        'xlim_upper': 0.9
    },
    'disorder_type_classification': {
        'title': 'Disorder Type Diag. \n Micro F1 ↑',
        'xlim_upper': 0.9
    },
    'disorder_symptom_classification': {
        'title': 'Symptom Diag. \n Micro F1 ↑',
        'xlim_upper': 0.9
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
        ax.set_yticklabels(models, fontsize=12)
    else:
        ax.set_yticklabels([''] * len(models))  # Empty labels but keep ticks
    ax.invert_yaxis()  # To display models from top to bottom

    # Set x-axis label and limits
    ax.set_xlim(0, xlim_upper)
    ax.tick_params(axis='x', labelsize=12)

    # Set title for the subplot
    ax.set_title(f"{title_text}", fontsize=14, fontweight='bold')

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
plt.savefig('AllModelsV2.png', dpi=300, bbox_inches='tight')
plt.show()