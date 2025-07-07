import numpy as np
from gpom import squash_probability
import matplotlib.pyplot as plt
import time
from typing import Dict
from sklearn.metrics import roc_curve, roc_auc_score, log_loss





def evaluate_gp_map(y_means, y_vars, ground_truth) -> Dict[str, float]:
    """Evaluate a map against ground truth, considering only observed regions.
    
    Args:
        predicted_map: Either a ternary occupancy grid (0=free, 1=occupied, 0.5=unknown)
                      or a probability map (0.0 to 1.0)
        ground_truth: Binary ground truth map (0=free, 1=occupied)
        observation_region: Boolean mask of observed regions
    """

    prob_occupied = squash_probability(y_means, y_vars)
    prob_occupied = prob_occupied.reshape(ground_truth.shape)
    
    # Flatten arrays for ROC calculation
    prob_occupied_flat = prob_occupied.flatten()
    ground_truth_flat = ground_truth.flatten()
    
    # Ensure ground truth is binary (0 or 1)
    ground_truth_flat = (ground_truth_flat > 0.5).astype(int)

    occupied_mask = prob_occupied > 0.65
    free_mask = prob_occupied < 0.35
    unknown_mask = (prob_occupied > 0.35) & (prob_occupied < 0.65)

    # True positives: correctly identified occupied cells
    tp = np.sum((occupied_mask) & (ground_truth == 1))
    
    # True negatives: correctly identified free cells
    tn = np.sum((free_mask) & (ground_truth == 0))
    
    # False positives: incorrectly identified as occupied
    fp = np.sum((occupied_mask) & (ground_truth == 0))
    
    # False negatives: incorrectly identified as free
    fn = np.sum((free_mask) & (ground_truth == 1))
    
    # Unknown cells in observed region
    unknown = np.sum((unknown_mask))
    
    # Calculate total valid cells (excluding unknown)
    total_valid = tp + tn + fp + fn
    
    # Calculate metrics only if there are valid cells to evaluate
    if total_valid > 0:
        accuracy = (tp + tn) / total_valid
        false_negatives = fn / total_valid
        false_positives = fp / total_valid
    else:
        accuracy = 0.0
        false_negatives = 0.0
        false_positives = 0.0
    
    unknown_percentage = unknown / (ground_truth.shape[0] * ground_truth.shape[1])
    observed_area_percentage = (tp + tn) / (ground_truth.shape[0] * ground_truth.shape[1])

    # Calculate ROC curve using flattened arrays
    fpr, tpr, thresholds = roc_curve(ground_truth_flat, prob_occupied_flat)
    roc_auc = roc_auc_score(ground_truth_flat, prob_occupied_flat)

    nll = log_loss(ground_truth_flat, prob_occupied_flat, labels=[0, 1])
    
    metrics = {
        'accuracy': accuracy,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'unknown_percentage': unknown_percentage,
        'observed_area_percentage': observed_area_percentage,
        'roc_auc': roc_auc,
        'nll': nll,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    return metrics








y_pred1 = np.load("y_pred_np1_large_10000.npy")
y_var1 = np.load("y_var_np1_large_10000.npy")


y_pred2 = np.load("y_pred_np2_large_10000.npy")
y_var2 = np.load("y_var_np2_large_10000.npy")

y_pred_merged = np.load("y_pred_merged_large_10000.npy")
y_var_merged = np.load("y_var_merged_large_10000.npy")

# Load ground truth data
ground_truth = np.load("ground_truth_large.npy")
print(ground_truth.shape)

# Analyze individual robot maps
print("=== Analyzing Robot 1 GPOM Map ===")
metrics_robot1 = evaluate_gp_map(y_pred1, y_var1, ground_truth)
print(f"Accuracy: {metrics_robot1['accuracy']:.4f}")
print(f"False Negatives: {metrics_robot1['false_negatives']:.4f}")
print(f"False Positives: {metrics_robot1['false_positives']:.4f}")
print(f"Unknown Percentage: {metrics_robot1['unknown_percentage']:.4f}")
print(f"Observed Area Percentage: {metrics_robot1['observed_area_percentage']:.4f}")
print(f"ROC AUC: {metrics_robot1['roc_auc']:.4f}")
print(f"Negative Log Likelihood: {metrics_robot1['nll']:.4f}")

print("\n=== Analyzing Robot 2 GPOM Map ===")
metrics_robot2 = evaluate_gp_map(y_pred2, y_var2, ground_truth)
print(f"Accuracy: {metrics_robot2['accuracy']:.4f}")
print(f"False Negatives: {metrics_robot2['false_negatives']:.4f}")
print(f"False Positives: {metrics_robot2['false_positives']:.4f}")
print(f"Unknown Percentage: {metrics_robot2['unknown_percentage']:.4f}")
print(f"Observed Area Percentage: {metrics_robot2['observed_area_percentage']:.4f}")
print(f"ROC AUC: {metrics_robot2['roc_auc']:.4f}")
print(f"Negative Log Likelihood: {metrics_robot2['nll']:.4f}")

print("\n=== Analyzing Merged GPOM Map ===")
metrics_merged = evaluate_gp_map(y_pred_merged, y_var_merged, ground_truth)
print(f"Accuracy: {metrics_merged['accuracy']:.4f}")
print(f"False Negatives: {metrics_merged['false_negatives']:.4f}")
print(f"False Positives: {metrics_merged['false_positives']:.4f}")
print(f"Unknown Percentage: {metrics_merged['unknown_percentage']:.4f}")
print(f"Observed Area Percentage: {metrics_merged['observed_area_percentage']:.4f}")
print(f"ROC AUC: {metrics_merged['roc_auc']:.4f}")
print(f"Negative Log Likelihood: {metrics_merged['nll']:.4f}")

# Create comparison plots
def plot_comparison_analysis():
    """Create comprehensive comparison plots for GPOM analysis."""
    
    # Prepare data for plotting
    robots = ['Robot 1', 'Robot 2', 'Merged']
    accuracies = [metrics_robot1['accuracy'], metrics_robot2['accuracy'], metrics_merged['accuracy']]
    false_negatives = [metrics_robot1['false_negatives'], metrics_robot2['false_negatives'], metrics_merged['false_negatives']]
    false_positives = [metrics_robot1['false_positives'], metrics_robot2['false_positives'], metrics_merged['false_positives']]
    unknown_percentages = [metrics_robot1['unknown_percentage'], metrics_robot2['unknown_percentage'], metrics_merged['unknown_percentage']]
    observed_areas = [metrics_robot1['observed_area_percentage'], metrics_robot2['observed_area_percentage'], metrics_merged['observed_area_percentage']]
    roc_aucs = [metrics_robot1['roc_auc'], metrics_robot2['roc_auc'], metrics_merged['roc_auc']]
    nlls = [metrics_robot1['nll'], metrics_robot2['nll'], metrics_merged['nll']]
    
    # Create subplots
    # fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle('GPOM Map Analysis Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    plt.figure()
    plt.bar(robots, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # False Negatives comparison
    plt.figure()
    plt.bar(robots, false_negatives, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('False Negatives Comparison')
    plt.ylabel('False Negatives Rate')
    plt.ylim(0, max(false_negatives) * 1.2)
    for i, v in enumerate(false_negatives):
        plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # False Positives comparison
    plt.figure()
    plt.bar(robots, false_positives, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('False Positives Comparison')
    plt.ylabel('False Positives Rate')
    plt.ylim(0, max(false_positives) * 1.2)
    for i, v in enumerate(false_positives):
        plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # Unknown Percentage comparison
    plt.figure()
    plt.bar(robots, unknown_percentages, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('Unknown Area Percentage')
    plt.ylabel('Unknown Percentage')
    plt.ylim(0, max(unknown_percentages) * 1.2)
    for i, v in enumerate(unknown_percentages):
        plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # ROC AUC comparison
    plt.figure()
    plt.bar(robots, roc_aucs, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('ROC AUC Comparison')
    plt.ylabel('ROC AUC')
    plt.ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Negative Log Likelihood comparison
    plt.figure()
    plt.bar(robots, nlls, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('Negative Log Likelihood Comparison')
    plt.ylabel('NLL (lower is better)')
    plt.ylim(0, max(nlls) * 1.2)
    for i, v in enumerate(nlls):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # plt.tight_layout()
    # plt.savefig('gpom_analysis_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()

# def plot_roc_curves():
#     """Plot ROC curves for all three maps."""
#     plt.figure(figsize=(10, 8))


    plt.figure()
    plt.plot(metrics_robot1['fpr'], metrics_robot1['tpr'], 
             label=f'Robot 1 (AUC = {metrics_robot1["roc_auc"]:.3f})', linewidth=2)
    plt.plot(metrics_robot2['fpr'], metrics_robot2['tpr'], 
             label=f'Robot 2 (AUC = {metrics_robot2["roc_auc"]:.3f})', linewidth=2)
    plt.plot(metrics_merged['fpr'], metrics_merged['tpr'], 
             label=f'Merged (AUC = {metrics_merged["roc_auc"]:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('gpom_roc_curves.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Plot ROC AUC
    plt.figure()
    plt.bar(robots, roc_aucs, color=['blue', 'green', 'red'], alpha=0.7)
    plt.title('ROC AUC Comparison')
    plt.ylabel('ROC AUC')
    plt.ylim(0, 1)
    for i, v in enumerate(roc_aucs):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    

# def plot_probability_maps():


    """Plot the probability maps for visual comparison."""
    prob_robot1 = squash_probability(y_pred1, y_var1)
    prob_robot2 = squash_probability(y_pred2, y_var2)
    prob_merged = squash_probability(y_pred_merged, y_var_merged)
    
    # Reshape to 2D for plotting
    prob_robot1 = prob_robot1.reshape(ground_truth.shape)
    prob_robot2 = prob_robot2.reshape(ground_truth.shape)
    prob_merged = prob_merged.reshape(ground_truth.shape)
    
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # fig.suptitle('GPOM Probability Maps Comparison', fontsize=16, fontweight='bold')

    plt.figure()
    
    # Ground truth
    im1 = plt.imshow(ground_truth, cmap='gray_r', vmin=0, vmax=1)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    
    # Robot 1
    plt.figure()
    im2 = plt.imshow(prob_robot1, cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Robot 1 (Accuracy: {metrics_robot1["accuracy"]:.3f})')
    plt.axis('off')
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    
    # Robot 2
    plt.figure()
    im3 = plt.imshow(prob_robot2, cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Robot 2 (Accuracy: {metrics_robot2["accuracy"]:.3f})')
    plt.axis('off')
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    
    # Merged
    plt.figure()
    im4 = plt.imshow(prob_merged, cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Merged (Accuracy: {metrics_merged["accuracy"]:.3f})')
    plt.axis('off')
    plt.colorbar(im4, fraction=0.046, pad=0.04)
    
    # plt.tight_layout()
    # plt.savefig('gpom_probability_maps.png', dpi=300, bbox_inches='tight')
    # plt.show()

    plt.show()

def save_analysis_results():
    """Save analysis results to a CSV file."""
    import pandas as pd
    
    results_data = {
        'Metric': ['Accuracy', 'False_Negatives', 'False_Positives', 'Unknown_Percentage', 
                   'Observed_Area_Percentage', 'ROC_AUC', 'NLL'],
        'Robot_1': [metrics_robot1['accuracy'], metrics_robot1['false_negatives'], 
                   metrics_robot1['false_positives'], metrics_robot1['unknown_percentage'],
                   metrics_robot1['observed_area_percentage'], metrics_robot1['roc_auc'], 
                   metrics_robot1['nll']],
        'Robot_2': [metrics_robot2['accuracy'], metrics_robot2['false_negatives'], 
                   metrics_robot2['false_positives'], metrics_robot2['unknown_percentage'],
                   metrics_robot2['observed_area_percentage'], metrics_robot2['roc_auc'], 
                   metrics_robot2['nll']],
        'Merged': [metrics_merged['accuracy'], metrics_merged['false_negatives'], 
                  metrics_merged['false_positives'], metrics_merged['unknown_percentage'],
                  metrics_merged['observed_area_percentage'], metrics_merged['roc_auc'], 
                  metrics_merged['nll']]
    }
    
    df = pd.DataFrame(results_data)
    df.to_csv('gpom_analysis_results.csv', index=False)
    print("\nAnalysis results saved to 'gpom_analysis_results.csv'")

# Run the analysis
if __name__ == "__main__":
    print("Starting GPOM Map Analysis...")
    
    # Generate all plots and analysis
    plot_comparison_analysis()

    
    print("\n=== Summary ===")
    print(f"Best Accuracy: {'Merged' if metrics_merged['accuracy'] > max(metrics_robot1['accuracy'], metrics_robot2['accuracy']) else 'Individual Robot'}")
    print(f"Best ROC AUC: {'Merged' if metrics_merged['roc_auc'] > max(metrics_robot1['roc_auc'], metrics_robot2['roc_auc']) else 'Individual Robot'}")
    print(f"Lowest NLL: {'Merged' if metrics_merged['nll'] < min(metrics_robot1['nll'], metrics_robot2['nll']) else 'Individual Robot'}")
    
    print("\nAnalysis complete! Check the generated plots and CSV file for detailed results.")


