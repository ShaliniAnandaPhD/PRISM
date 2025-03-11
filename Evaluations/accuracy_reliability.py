"""
ACCURACY & RELIABILITY EVALUATION SCRIPT
This script measures the accuracy and reliability of a given function or model by
comparing its outputs against a ground truth dataset. 
The goal is to ensure consistent performance over multiple test cases, detect deviations,
and provide a structured audit trail for legal review.
"""
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

# Configure logging for auditability
logging.basicConfig(filename='accuracy_reliability.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def compute_accuracy(predictions, ground_truth):
    """
    Compute accuracy as the percentage of correct predictions.
    
    :param predictions: List of predicted values.
    :param ground_truth: List of true values.
    :return: Accuracy percentage.
    """
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    accuracy = (correct / len(ground_truth)) * 100
    return accuracy

def compute_precision_recall_f1(predictions, ground_truth):
    """
    Compute precision, recall, and F1 score for classification models.
    
    :param predictions: List of predicted values (0 or 1).
    :param ground_truth: List of true values (0 or 1).
    :return: Dictionary containing precision, recall, and F1 score.
    """
    # True positives, false positives, false negatives
    tp = sum((p == 1 and gt == 1) for p, gt in zip(predictions, ground_truth))
    fp = sum((p == 1 and gt == 0) for p, gt in zip(predictions, ground_truth))
    fn = sum((p == 0 and gt == 1) for p, gt in zip(predictions, ground_truth))
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100
    }

def compute_regression_metrics(predictions, ground_truth):
    """
    Compute common regression metrics (MAE, MSE, RMSE, R²).
    
    :param predictions: List of predicted values.
    :param ground_truth: List of true values.
    :return: Dictionary containing regression metrics.
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate errors
    errors = predictions - ground_truth
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Mean Absolute Error
    mae = np.mean(abs_errors)
    
    # Mean Squared Error
    mse = np.mean(squared_errors)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R² (coefficient of determination)
    mean_gt = np.mean(ground_truth)
    ss_total = np.sum((ground_truth - mean_gt) ** 2)
    ss_residual = np.sum(squared_errors)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r_squared': r_squared
    }

def evaluate_reliability(predictions, expected_variance=0.05):
    """
    Assess reliability by measuring the variance of predictions.
    
    :param predictions: List of predicted values.
    :param expected_variance: Threshold for acceptable variance.
    :return: Tuple of (boolean indicating reliability, variance, coefficient of variation).
    """
    predictions = np.array(predictions)
    variance = np.var(predictions)
    std_dev = np.std(predictions)
    
    # Coefficient of variation (relative standard deviation)
    mean = np.mean(predictions)
    cv = std_dev / mean if mean != 0 else float('inf')
    
    reliable = variance <= expected_variance
    
    return reliable, variance, cv

def check_time_series_drift(predictions, ground_truth):
    """
    Check for drift in time series data by analyzing residuals over time.
    
    :param predictions: List of predicted values in time order.
    :param ground_truth: List of true values in time order.
    :return: Dictionary with drift analysis results.
    """
    residuals = np.array(ground_truth) - np.array(predictions)
    time_indices = np.arange(len(residuals))
    
    # Linear regression on residuals to detect drift
    n = len(residuals)
    sum_x = np.sum(time_indices)
    sum_y = np.sum(residuals)
    sum_xy = np.sum(time_indices * residuals)
    sum_xx = np.sum(time_indices * time_indices)
    
    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate drift significance (simplified)
    drift_significant = abs(slope) > 0.01
    
    return {
        'slope': slope,
        'intercept': intercept,
        'significant_drift': drift_significant
    }

def compute_confidence_interval(accuracy, n_samples, confidence=0.95):
    """
    Compute confidence interval for accuracy using normal approximation.
    
    :param accuracy: Accuracy as a percentage.
    :param n_samples: Number of samples used to compute accuracy.
    :param confidence: Confidence level (default: 0.95 for 95%).
    :return: Tuple of (lower bound, upper bound).
    """
    from scipy import stats
    
    # Convert accuracy from percentage to proportion
    p = accuracy / 100
    
    # Standard error
    se = np.sqrt(p * (1 - p) / n_samples)
    
    # Critical value for the given confidence level
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Margin of error
    margin = z * se
    
    # Confidence interval (as percentage)
    lower = max(0, (p - margin)) * 100
    upper = min(1, (p + margin)) * 100
    
    return lower, upper

def plot_results(predictions, ground_truth, output_dir="evaluation_results"):
    """
    Create visualizations of model performance.
    
    :param predictions: List of predicted values.
    :param ground_truth: List of true values.
    :param output_dir: Directory to save plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    
    # Add diagonal reference line (perfect predictions)
    min_val = min(np.min(predictions), np.min(ground_truth))
    max_val = max(np.max(predictions), np.max(ground_truth))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
    plt.close()
    
    # Plot residuals
    residuals = ground_truth - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # Plot residuals over time (for time series data)
    plt.figure(figsize=(10, 6))
    plt.plot(residuals, marker='o', linestyle='-', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
    plt.xlabel('Time Index')
    plt.ylabel('Residuals')
    plt.title('Residuals Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'residuals_time.png'))
    plt.close()
    
    # Plot histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'residuals_histogram.png'))
    plt.close()
    
    logging.info(f"Plots saved to {output_dir}")

def cross_validate(model_func, inputs, ground_truth, n_folds=5):
    """
    Perform k-fold cross-validation for more robust performance estimation.
    
    :param model_func: Function or model to evaluate.
    :param inputs: List of input values.
    :param ground_truth: List of ground truth values.
    :param n_folds: Number of folds for cross-validation.
    :return: Dictionary of performance metrics across folds.
    """
    # Convert to numpy arrays
    inputs = np.array(inputs)
    ground_truth = np.array(ground_truth)
    
    # Create folds
    fold_size = len(inputs) // n_folds
    fold_results = []
    
    for i in range(n_folds):
        # Define test indices for this fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(inputs)
        test_indices = list(range(test_start, test_end))
        
        # Define train indices (all other indices)
        train_indices = [j for j in range(len(inputs)) if j not in test_indices]
        
        # Split data
        X_train = inputs[train_indices]
        y_train = ground_truth[train_indices]
        X_test = inputs[test_indices]
        y_test = ground_truth[test_indices]
        
        # Train model if it has a fit method
        if hasattr(model_func, 'fit'):
            model_func.fit(X_train, y_train)
        
        # Make predictions
        predictions = [model_func(x) for x in X_test]
        
        # Compute metrics
        accuracy = compute_accuracy(predictions, y_test)
        regression_metrics = compute_regression_metrics(predictions, y_test)
        reliable, variance, cv = evaluate_reliability(predictions)
        
        # Store results for this fold
        fold_results.append({
            'accuracy': accuracy,
            'mae': regression_metrics['mae'],
            'mse': regression_metrics['mse'],
            'rmse': regression_metrics['rmse'],
            'r_squared': regression_metrics['r_squared'],
            'variance': variance,
            'reliable': reliable
        })
    
    # Compute average metrics across folds
    avg_results = {}
    for key in fold_results[0].keys():
        if key != 'reliable':
            avg_results[key] = sum(fold[key] for fold in fold_results) / n_folds
        else:
            # For boolean values, take the majority vote
            avg_results[key] = sum(fold[key] for fold in fold_results) >= n_folds / 2
    
    # Compute standard deviations for metrics
    std_results = {}
    for key in fold_results[0].keys():
        if key != 'reliable':
            std_results[key] = np.std([fold[key] for fold in fold_results])
    
    return {
        'average_metrics': avg_results,
        'std_metrics': std_results,
        'fold_results': fold_results
    }

def generate_summary_report(metrics, reliability_results, drift_results=None, 
                           confidence_interval=None, cross_val_results=None,
                           output_file="accuracy_reliability_summary.json"):
    """
    Generate a comprehensive summary report of all evaluation results.
    
    :param metrics: Dictionary of performance metrics.
    :param reliability_results: Tuple of (reliable flag, variance, cv).
    :param drift_results: Optional dictionary with drift analysis results.
    :param confidence_interval: Optional tuple of (lower, upper) bounds.
    :param cross_val_results: Optional cross-validation results.
    :param output_file: File path to save the JSON report.
    """
    # Unpack reliability results
    reliable, variance, cv = reliability_results
    
    # Construct summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance_metrics": metrics,
        "reliability": {
            "status": "Pass" if reliable else "Fail",
            "variance": variance,
            "coefficient_of_variation": cv
        }
    }
    
    # Add confidence interval if provided
    if confidence_interval:
        summary["confidence_interval"] = {
            "lower_bound": confidence_interval[0],
            "upper_bound": confidence_interval[1]
        }
    
    # Add drift analysis if provided
    if drift_results:
        summary["drift_analysis"] = drift_results
    
    # Add cross-validation results if provided
    if cross_val_results:
        summary["cross_validation"] = {
            "average_metrics": cross_val_results["average_metrics"],
            "std_metrics": cross_val_results["std_metrics"],
            "fold_count": len(cross_val_results["fold_results"])
        }
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    logging.info(f"Summary report saved to {output_file}")
    return summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Accuracy & Reliability Evaluation")
    
    parser.add_argument("--model-type", choices=["classification", "regression", "time_series"],
                      default="regression", help="Type of model being evaluated")
    
    parser.add_argument("--expected-variance", type=float, default=0.05,
                      help="Threshold for acceptable variance")
    
    parser.add_argument("--cross-validate", action="store_true",
                      help="Perform cross-validation")
    
    parser.add_argument("--folds", type=int, default=5,
                      help="Number of folds for cross-validation")
    
    parser.add_argument("--output-dir", default="evaluation_results",
                      help="Directory to save plots and results")
    
    parser.add_argument("--confidence-level", type=float, default=0.95,
                      help="Confidence level for interval calculation")
    
    return parser.parse_args()

# Example model for testing
def example_model(x):
    return x * 2

# Example test case
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Example data
    test_inputs = [1, 2, 3, 4, 5]
    ground_truth = [2, 4, 6, 8, 10]
    
    # Get predictions
    predictions = [example_model(x) for x in test_inputs]
    
    # Collect all metrics based on model type
    all_metrics = {}
    
    # Compute accuracy for all model types
    accuracy = compute_accuracy(predictions, ground_truth)
    all_metrics["accuracy"] = accuracy
    
    # Model-specific metrics
    if args.model_type == "classification":
        # Compute classification metrics
        class_metrics = compute_precision_recall_f1(predictions, ground_truth)
        all_metrics.update(class_metrics)
        
    elif args.model_type in ["regression", "time_series"]:
        # Compute regression metrics
        reg_metrics = compute_regression_metrics(predictions, ground_truth)
        all_metrics.update(reg_metrics)
    
    # Compute confidence interval for accuracy
    ci_lower, ci_upper = compute_confidence_interval(accuracy, len(test_inputs), args.confidence_level)
    
    # Evaluate reliability
    reliability_results = evaluate_reliability(predictions, args.expected_variance)
    
    # Time series specific analysis
    drift_results = None
    if args.model_type == "time_series":
        drift_results = check_time_series_drift(predictions, ground_truth)
    
    # Cross-validation if requested
    cross_val_results = None
    if args.cross_validate:
        cross_val_results = cross_validate(example_model, test_inputs, ground_truth, args.folds)
    
    # Generate plots
    plot_results(predictions, ground_truth, args.output_dir)
    
    # Generate and save summary report
    summary = generate_summary_report(
        all_metrics, 
        reliability_results,
        drift_results,
        (ci_lower, ci_upper),
        cross_val_results,
        os.path.join(args.output_dir, "accuracy_reliability_summary.json")
    )
    
    # Log results
    logging.info(f"Model Accuracy: {accuracy:.2f}% (95% CI: {ci_lower:.2f}% - {ci_upper:.2f}%)")
    logging.info(f"Reliability: {'Pass' if reliability_results[0] else 'Fail'} | Variance: {reliability_results[1]:.6f}")
    
    # Print summary
    print("Accuracy & Reliability evaluation complete.")
    print(f"Accuracy: {accuracy:.2f}% (95% CI: {ci_lower:.2f}% - {ci_upper:.2f}%)")
    print(f"Reliability: {'Pass' if reliability_results[0] else 'Fail'} (Variance: {reliability_results[1]:.6f})")
    print(f"Check '{args.output_dir}/accuracy_reliability_summary.json' for detailed results.")
    print(f"Visualizations saved to '{args.output_dir}'.")

"""
TODO:
- Extend the reliability test to support time-series data with autocorrelation analysis.
- Implement statistical significance testing for accuracy comparisons between models.
- Add support for multi-class classification with confusion matrix visualization.
- Enhance precision, recall, and F1-score evaluation with PR curves for classification models.
- Automate ground truth verification using external databases or reference standards.
- Add anomaly detection for identifying outliers in model predictions.
- Implement fairness metrics to evaluate model bias across different subgroups.
- Add support for ensemble evaluation by combining results from multiple models.
- Extend visualization capabilities with interactive plots and dashboards.
- Implement model calibration assessment for probabilistic predictions.
"""
