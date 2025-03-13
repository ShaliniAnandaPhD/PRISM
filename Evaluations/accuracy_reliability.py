"""
A script to help you evaluate how well your model is performing against ground truth data.
I built this to save time when checking if my models are reliable enough for production.

It handles classification, regression, and time series models with various metrics
and generates nice visualizations to help with reporting.

Usage:
    python model_evaluator.py --model-type regression --cross-validate
"""
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

# Set up logging - helpful for keeping an audit trail for compliance folks
logging.basicConfig(filename='accuracy_reliability.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def compute_accuracy(predictions, ground_truth):
    """How many did we get right?"""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    return (correct / len(ground_truth)) * 100

def compute_precision_recall_f1(predictions, ground_truth):
    """
    For classification tasks - the standard metrics everyone asks for.
    Precision: When we predict yes, how often are we right?
    Recall: How many actual yeses did we catch?
    F1: Harmonic mean of the two (balances both concerns)
    """
    # Count the true positives, false positives, and false negatives
    tp = sum((p == 1 and gt == 1) for p, gt in zip(predictions, ground_truth))
    fp = sum((p == 1 and gt == 0) for p, gt in zip(predictions, ground_truth))
    fn = sum((p == 0 and gt == 1) for p, gt in zip(predictions, ground_truth))
    
    # Avoid division by zero (learned this the hard way!)
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
    For regression tasks - all the usual suspects.
    I've found RMSE most useful in practice, but included the others
    because stakeholders always ask for them.
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate various error metrics
    errors = predictions - ground_truth
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Mean Absolute Error - easy to explain to non-technical folks
    mae = np.mean(abs_errors)
    
    # Mean Squared Error
    mse = np.mean(squared_errors)
    
    # Root Mean Squared Error - my go-to for most regression problems
    rmse = np.sqrt(mse)
    
    # R² (coefficient of determination) - management loves this one
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
    Check if predictions are consistent enough for production.
    High variance = unreliable model = unhappy users.
    
    I picked 0.05 as the default threshold based on our past projects,
    but you might want to adjust it for your specific case.
    """
    predictions = np.array(predictions)
    variance = np.var(predictions)
    std_dev = np.std(predictions)
    
    # Coefficient of variation - useful for comparing different scales
    mean = np.mean(predictions)
    cv = std_dev / mean if mean != 0 else float('inf')
    
    # Simple pass/fail check
    reliable = variance <= expected_variance
    
    return reliable, variance, cv

def check_time_series_drift(predictions, ground_truth):
    """
    My simple check for drift in time series predictions.
    If the residuals trend over time, something's probably wrong.
    """
    residuals = np.array(ground_truth) - np.array(predictions)
    time_indices = np.arange(len(residuals))
    
    # Quick and dirty linear regression on residuals
    n = len(residuals)
    sum_x = np.sum(time_indices)
    sum_y = np.sum(residuals)
    sum_xy = np.sum(time_indices * residuals)
    sum_xx = np.sum(time_indices * time_indices)
    
    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Arbitrary threshold - adjust to your needs
    drift_significant = abs(slope) > 0.01
    
    return {
        'slope': slope,
        'intercept': intercept,
        'significant_drift': drift_significant
    }

def compute_confidence_interval(accuracy, n_samples, confidence=0.95):
    """
    How confident are we in our accuracy estimate?
    Good for small test sets where one lucky guess can skew things.
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
    Create some pretty plots for the report.
    
    The residual plot is usually the most telling - look for patterns
    that might reveal where your model is struggling.
    """
    # Make sure we have somewhere to save the plots
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Plot 1: Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    
    # Add the perfect-prediction line
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
    
    # Plot 2: Residuals vs Actual
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
    
    # Plot 3: Residuals over time
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
    
    # Plot 4: Histogram of residuals
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
    K-fold cross-validation to get a more realistic picture.
    
    If you're getting very different results across folds,
    your model might be overfitting or your dataset might be imbalanced.
    """
    inputs = np.array(inputs)
    ground_truth = np.array(ground_truth)
    
    # Split data into folds
    fold_size = len(inputs) // n_folds
    fold_results = []
    
    for i in range(n_folds):
        # Figure out which data goes in which fold
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(inputs)
        test_indices = list(range(test_start, test_end))
        
        # Everything else is training data
        train_indices = [j for j in range(len(inputs)) if j not in test_indices]
        
        # Split the data
        X_train = inputs[train_indices]
        y_train = ground_truth[train_indices]
        X_test = inputs[test_indices]
        y_test = ground_truth[test_indices]
        
        # Train the model if it has a fit method
        if hasattr(model_func, 'fit'):
            model_func.fit(X_train, y_train)
        
        # Get predictions
        predictions = [model_func(x) for x in X_test]
        
        # Calculate metrics for this fold
        accuracy = compute_accuracy(predictions, y_test)
        regression_metrics = compute_regression_metrics(predictions, y_test)
        reliable, variance, cv = evaluate_reliability(predictions)
        
        # Save results
        fold_results.append({
            'accuracy': accuracy,
            'mae': regression_metrics['mae'],
            'mse': regression_metrics['mse'],
            'rmse': regression_metrics['rmse'],
            'r_squared': regression_metrics['r_squared'],
            'variance': variance,
            'reliable': reliable
        })
    
    # Calculate average across folds
    avg_results = {}
    for key in fold_results[0].keys():
        if key != 'reliable':
            avg_results[key] = sum(fold[key] for fold in fold_results) / n_folds
        else:
            # For boolean values, majority rules
            avg_results[key] = sum(fold[key] for fold in fold_results) >= n_folds / 2
    
    # Calculate standard deviations (helpful to see consistency)
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
    Create a nice JSON report with all our findings.
    Great for audits or sharing with the team.
    """
    reliable, variance, cv = reliability_results
    
    # Put everything together
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance_metrics": metrics,
        "reliability": {
            "status": "Pass" if reliable else "Fail",
            "variance": variance,
            "coefficient_of_variation": cv
        }
    }
    
    # Add confidence interval if we calculated it
    if confidence_interval:
        summary["confidence_interval"] = {
            "lower_bound": confidence_interval[0],
            "upper_bound": confidence_interval[1]
        }
    
    # Add drift analysis for time series data
    if drift_results:
        summary["drift_analysis"] = drift_results
    
    # Add cross-validation results if we did that
    if cross_val_results:
        summary["cross_validation"] = {
            "average_metrics": cross_val_results["average_metrics"],
            "std_metrics": cross_val_results["std_metrics"],
            "fold_count": len(cross_val_results["fold_results"])
        }
    
    # Save to a file
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    logging.info(f"Summary report saved to {output_file}")
    return summary

def parse_args():
    """Handle command line arguments."""
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

# A super simple model for testing
def example_model(x):
    # Just doubles the input - perfect for our test data
    return x * 2

# Main script
if __name__ == "__main__":
    # Get command line args
    args = parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Example data for testing
    # In real life, you'd load this from a file
    test_inputs = [1, 2, 3, 4, 5]
    ground_truth = [2, 4, 6, 8, 10]
    
    # Run the model on our test data
    predictions = [example_model(x) for x in test_inputs]
    
    # Store all our metrics
    all_metrics = {}
    
    # Basic accuracy works for everything
    accuracy = compute_accuracy(predictions, ground_truth)
    all_metrics["accuracy"] = accuracy
    
    # Add metrics specific to the model type
    if args.model_type == "classification":
        # For classification, we care about precision, recall, etc.
        class_metrics = compute_precision_recall_f1(predictions, ground_truth)
        all_metrics.update(class_metrics)
        
    elif args.model_type in ["regression", "time_series"]:
        # For regression, we want error metrics
        reg_metrics = compute_regression_metrics(predictions, ground_truth)
        all_metrics.update(reg_metrics)
    
    # How confident are we in our accuracy?
    ci_lower, ci_upper = compute_confidence_interval(accuracy, len(test_inputs), args.confidence_level)
    
    # Is the model reliable?
    reliability_results = evaluate_reliability(predictions, args.expected_variance)
    
    # Check for drift in time series
    drift_results = None
    if args.model_type == "time_series":
        drift_results = check_time_series_drift(predictions, ground_truth)
    
    # Cross-validation if requested
    cross_val_results = None
    if args.cross_validate:
        cross_val_results = cross_validate(example_model, test_inputs, ground_truth, args.folds)
    
    # Generate pretty plots
    plot_results(predictions, ground_truth, args.output_dir)
    
    # Create the final report
    summary = generate_summary_report(
        all_metrics, 
        reliability_results,
        drift_results,
        (ci_lower, ci_upper),
        cross_val_results,
        os.path.join(args.output_dir, "accuracy_reliability_summary.json")
    )
    
    # Log everything for posterity
    logging.info(f"Model Accuracy: {accuracy:.2f}% (95% CI: {ci_lower:.2f}% - {ci_upper:.2f}%)")
    logging.info(f"Reliability: {'Pass' if reliability_results[0] else 'Fail'} | Variance: {reliability_results[1]:.6f}")
    
    # Show a nice summary in the console
    print("Accuracy & Reliability evaluation complete!")
    print(f"Accuracy: {accuracy:.2f}% (95% CI: {ci_lower:.2f}% - {ci_upper:.2f}%)")
    print(f"Reliability: {'Pass' if reliability_results[0] else 'Fail'} (Variance: {reliability_results[1]:.6f})")
    print(f"Check '{args.output_dir}/accuracy_reliability_summary.json' for detailed results.")
    print(f"Visualizations saved to '{args.output_dir}'.")

# My to-do list for future improvements:
"""
TODO:
- Add autocorrelation analysis for time-series reliability testing
- Add statistical significance testing to compare models 
- Build a nice confusion matrix viz for multi-class problems
- Create PR curves for classification models
- Add ground truth verification against reference data
- Add anomaly detection to spot weird predictions
- Add fairness metrics to check for bias issues
- Support ensemble models by combining results
- Make interactive dashboards with something like Dash or Streamlit
- Add model calibration checks for probabilistic predictions
"""
