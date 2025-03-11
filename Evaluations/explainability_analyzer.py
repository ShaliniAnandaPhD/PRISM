"""
This framework evaluates and enhances the explainability and interpretability of machine learning
models and algorithmic functions. It provides:

1. Comprehensive explainability assessment for different model types
2. Integration with state-of-the-art XAI techniques (SHAP, LIME, etc.)
3. Visualization of feature importance and decision boundaries
4. Quantitative metrics for measuring interpretability
5. Structured reporting for regulatory and legal compliance
6. Human-friendliness evaluation of model explanations
7. Output format optimization for stakeholder understanding

The framework supports various model types including:
- Tree-based models (Random Forests, XGBoost, etc.)
- Linear models (Linear/Logistic Regression, etc.)
- Neural Networks (with activation visualization)
- Black-box models (through post-hoc explanation techniques)
"""

import logging
import json
import inspect
import os
import re
import sys
import argparse
import datetime
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import wraps
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging with both file and console handlers
def setup_logging(log_file='explainability_interpretability.log', console_level=logging.INFO):
    """Setup logging to file and console with different levels."""
    logger = logging.getLogger('explainability')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Utility function to check if a model is a specific type
def is_tree_based_model(model):
    """Check if the model is tree-based."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['tree', 'forest', 'boost', 'xgb', 'lgbm', 'catboost'])

def is_linear_model(model):
    """Check if the model is a linear model."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['linear', 'logistic', 'regression', 'lasso', 'ridge', 'elasticnet'])

def is_neural_network(model):
    """Check if the model is a neural network."""
    # Check for TensorFlow/Keras models
    if TF_AVAILABLE and isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return True
    
    # Check for PyTorch models
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        return True
    
    # Check by class name
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['neural', 'deep', 'mlp', 'keras', 'tensorflow', 'torch', 'nn.'])

# Core explainability assessment functions
def check_docstring_quality(function):
    """
    Evaluate the quality of a function's docstring.
    
    :param function: The function to be checked.
    :return: Dictionary with docstring quality metrics.
    """
    if not function.__doc__:
        return {
            "has_docstring": False,
            "quality_score": 0,
            "length": 0,
            "has_parameters": False,
            "has_return_info": False,
            "has_examples": False,
            "suggestions": ["Add a docstring to explain the function's purpose and usage."]
        }
    
    docstring = function.__doc__.strip()
    
    # Basic metrics
    has_parameters = ":param" in docstring or "Parameters:" in docstring
    has_return_info = ":return" in docstring or "Returns:" in docstring
    has_examples = "Example" in docstring
    length = len(docstring)
    
    # Calculate quality score (0-100)
    quality_score = 0
    if length > 0:
        quality_score += min(length / 5, 30)  # Up to 30 points for length
    if has_parameters:
        quality_score += 25  # 25 points for parameter documentation
    if has_return_info:
        quality_score += 25  # 25 points for return value documentation
    if has_examples:
        quality_score += 20  # 20 points for examples
    
    # Generate suggestions
    suggestions = []
    if not has_parameters:
        suggestions.append("Add parameter descriptions.")
    if not has_return_info:
        suggestions.append("Document return values.")
    if not has_examples:
        suggestions.append("Include usage examples.")
    if length < 50:
        suggestions.append("Expand the description to be more detailed.")
        
    return {
        "has_docstring": True,
        "quality_score": quality_score,
        "length": length,
        "has_parameters": has_parameters,
        "has_return_info": has_return_info,
        "has_examples": has_examples,
        "suggestions": suggestions
    }

def analyze_function_complexity(function):
    """
    Analyze function complexity and structure for explainability.
    
    :param function: The function to be analyzed.
    :return: Dictionary with complexity metrics.
    """
    try:
        source_code = inspect.getsource(function)
    except (TypeError, OSError) as e:
        logger.warning(f"Could not retrieve source code: {e}")
        return {
            "complexity_score": None,
            "lines_of_code": None,
            "nested_depth": None,
            "branching_factor": None,
            "error": str(e)
        }
    
    # Count lines of code
    lines = source_code.strip().split('\n')
    lines_of_code = len(lines)
    
    # Count conditional statements (if, elif, else)
    conditional_counts = len(re.findall(r'\bif\b|\belif\b|\belse\b', source_code))
    
    # Count loops (for, while)
    loop_counts = len(re.findall(r'\bfor\b|\bwhile\b', source_code))
    
    # Estimate nesting depth
    indent_levels = [len(line) - len(line.lstrip()) for line in lines]
    max_indent = max(indent_levels) if indent_levels else 0
    nested_depth = max_indent // 4  # Assuming 4 spaces per indentation level
    
    # Calculate complexity score (higher means more complex, less interpretable)
    complexity_score = (
        conditional_counts * 2 + 
        loop_counts * 3 + 
        nested_depth * 5 + 
        lines_of_code / 10
    )
    
    # Calculate branching factor
    branching_factor = conditional_counts / lines_of_code if lines_of_code > 0 else 0
    
    # Interpretability assessment
    if complexity_score < 10:
        interpretability = "High"
    elif complexity_score < 25:
        interpretability = "Medium"
    else:
        interpretability = "Low"
        
    return {
        "complexity_score": complexity_score,
        "interpretability": interpretability,
        "lines_of_code": lines_of_code,
        "nested_depth": nested_depth,
        "branching_factor": branching_factor,
        "conditional_counts": conditional_counts,
        "loop_counts": loop_counts
    }

def extract_feature_importance(model, feature_names=None, X=None, y=None):
    """
    Extract feature importance from a model using appropriate method.
    
    :param model: The trained model to analyze.
    :param feature_names: List of feature names.
    :param X: Input data for permutation importance (optional).
    :param y: Target data for permutation importance (optional).
    :return: Dictionary with feature importance information.
    """
    importance_methods = []
    importance_values = {}
    error_messages = []
    
    # Try built-in feature importance
    if hasattr(model, "feature_importances_"):
        importance_methods.append("built_in")
        if feature_names and len(model.feature_importances_) == len(feature_names):
            importance_values["built_in"] = dict(zip(feature_names, model.feature_importances_.tolist()))
        else:
            importance_values["built_in"] = model.feature_importances_.tolist()
    
    # Try coefficients for linear models
    if hasattr(model, "coef_"):
        importance_methods.append("coefficients")
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = np.mean(np.abs(coefs), axis=0)  # For multi-class models
            
        if feature_names and len(coefs) == len(feature_names):
            importance_values["coefficients"] = dict(zip(feature_names, coefs.tolist()))
        else:
            importance_values["coefficients"] = coefs.tolist()
    
    # Try permutation importance if sklearn is available and data is provided
    if SKLEARN_AVAILABLE and X is not None and y is not None:
        try:
            importance_methods.append("permutation")
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            
            if feature_names and len(perm_importance.importances_mean) == len(feature_names):
                importance_values["permutation"] = dict(zip(feature_names, perm_importance.importances_mean.tolist()))
            else:
                importance_values["permutation"] = perm_importance.importances_mean.tolist()
        except Exception as e:
            error_messages.append(f"Permutation importance failed: {str(e)}")
    
    # Try SHAP if available and appropriate
    if SHAP_AVAILABLE and X is not None:
        try:
            sample_X = X[:100] if hasattr(X, "__len__") and len(X) > 100 else X
            
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
                importance_methods.append("shap_tree")
            else:
                explainer = shap.Explainer(model)
                importance_methods.append("shap")
                
            shap_values = explainer.shap_values(sample_X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For multi-class models, take mean absolute value across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            # Calculate feature importance as mean absolute SHAP value
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            if feature_names and len(mean_abs_shap) == len(feature_names):
                importance_values["shap"] = dict(zip(feature_names, mean_abs_shap.tolist()))
            else:
                importance_values["shap"] = mean_abs_shap.tolist()
                
        except Exception as e:
            error_messages.append(f"SHAP analysis failed: {str(e)}")
    
    return {
        "methods_available": importance_methods,
        "feature_importance": importance_values,
        "errors": error_messages
    }

def analyze_model_structure(model):
    """
    Analyze the structure of a model to assess interpretability.
    
    :param model: The model to analyze.
    :return: Dictionary with model structure information.
    """
    model_type = "unknown"
    interpretability_score = 0
    structure_info = {}
    
    # Determine model type
    if is_tree_based_model(model):
        model_type = "tree_based"
        
        # Extract tree structure information
        if hasattr(model, "tree_") or hasattr(model, "estimators_"):
            n_nodes = 0
            max_depth = 0
            
            if hasattr(model, "tree_"):
                # Single tree model
                n_nodes = model.tree_.node_count
                max_depth = model.get_depth() if hasattr(model, "get_depth") else model.tree_.max_depth
                structure_info["n_nodes"] = n_nodes
                structure_info["max_depth"] = max_depth
            
            elif hasattr(model, "estimators_"):
                # Ensemble of trees
                n_estimators = len(model.estimators_)
                structure_info["n_estimators"] = n_estimators
                
                depths = []
                nodes = []
                
                for estimator in model.estimators_:
                    if hasattr(estimator, "tree_"):
                        nodes.append(estimator.tree_.node_count)
                        if hasattr(estimator, "get_depth"):
                            depths.append(estimator.get_depth())
                        else:
                            depths.append(estimator.tree_.max_depth)
                
                if depths:
                    max_depth = max(depths)
                    structure_info["max_depth"] = max_depth
                    structure_info["mean_depth"] = sum(depths) / len(depths)
                
                if nodes:
                    n_nodes = sum(nodes)
                    structure_info["total_nodes"] = n_nodes
                    structure_info["mean_nodes_per_tree"] = n_nodes / n_estimators
            
            # Calculate interpretability score for tree models (higher is more interpretable)
            if max_depth > 0:
                depth_penalty = max(0, (max_depth - 3) * 10)  # Deeper trees are less interpretable
                interpretability_score = max(0, 100 - depth_penalty)
                
                if "n_estimators" in structure_info:
                    # Ensemble models are less interpretable
                    ensemble_penalty = min(80, structure_info["n_estimators"] * 2)
                    interpretability_score = max(0, interpretability_score - ensemble_penalty)
        
    elif is_linear_model(model):
        model_type = "linear"
        
        # Linear models are generally interpretable
        interpretability_score = 80
        
        # Check for sparsity in linear models
        if hasattr(model, "coef_"):
            coefs = model.coef_
            n_features = coefs.size if len(coefs.shape) == 1 else coefs.shape[1]
            n_nonzero = np.count_nonzero(coefs)
            sparsity = 1.0 - (n_nonzero / n_features) if n_features > 0 else 0
            
            structure_info["n_features"] = n_features
            structure_info["n_nonzero_coefficients"] = n_nonzero
            structure_info["sparsity"] = sparsity
            
            # Sparse models are more interpretable
            if sparsity > 0.5:
                interpretability_score += 15
            elif sparsity < 0.2:
                interpretability_score -= 10
    
    elif is_neural_network(model):
        model_type = "neural_network"
        
        # Neural networks are generally less interpretable
        interpretability_score = 30
        
        # Analyze network architecture
        if TF_AVAILABLE and isinstance(model, tf.keras.Model):
            # For TensorFlow/Keras models
            structure_info["layers"] = []
            total_params = 0
            
            for i, layer in enumerate(model.layers):
                layer_info = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape)
                }
                
                if hasattr(layer, "count_params"):
                    params = layer.count_params()
                    layer_info["parameters"] = params
                    total_params += params
                
                structure_info["layers"].append(layer_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(model.layers) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
        
        elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            # For PyTorch models
            structure_info["modules"] = []
            total_params = 0
            
            for name, module in model.named_modules():
                if list(module.children()):  # Skip container modules
                    continue
                    
                module_info = {
                    "name": name,
                    "type": module.__class__.__name__
                }
                
                params = sum(p.numel() for p in module.parameters())
                module_info["parameters"] = params
                total_params += params
                
                structure_info["modules"].append(module_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(structure_info["modules"]) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
    
    else:
        # Unknown model type
        model_type = "unknown"
        interpretability_score = 40  # Default score for unknown models
        
        # Try to extract some generic information
        model_attributes = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
        for attr in model_attributes:
            try:
                val = getattr(model, attr)
                if isinstance(val, (int, float, str, bool)):
                    structure_info[attr] = val
            except:
                pass
    
    # Determine interpretability category
    if interpretability_score >= 70:
        interpretability_category = "High"
    elif interpretability_score >= 40:
        interpretability_category = "Medium"
    else:
        interpretability_category = "Low"
        
    return {
        "model_type": model_type,
        "interpretability_score": interpretability_score,
        "interpretability_category": interpretability_category,
        "structure_info": structure_info
    }

def generate_lime_explanation(model, X, feature_names=None, class_names=None, instance_idx=0):
    """
    Generate a LIME explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param class_names: List of class names for classification.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with LIME explanation results.
    """
    if not LIME_AVAILABLE:
        return {"error": "LIME package is not available. Install with 'pip install lime'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx].values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx]
        
        # Determine if we're doing classification or regression
        # This is a simplification - in practice, you might need to check model type
        mode = "classification"
        if class_names is None:
            # Try to guess if it's binary classification
            mode = "classification"
            class_names = [0, 1]
        
        # Create the LIME explainer
        explainer = LimeTabularExplainer(
            X if isinstance(X, np.ndarray) else X.values,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode
        )
        
        # Define prediction function based on expected input
        if mode == "classification":
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:
                # Fallback to non-probabilistic prediction
                def predict_fn(x):
                    preds = model.predict(x)
                    # Convert to pseudo-probabilities for binary classification
                    if len(preds.shape) == 1 and len(class_names) == 2:
                        return np.vstack([(1-preds), preds]).T
                    return preds
        else:
            predict_fn = model.predict
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance, 
            predict_fn,
            num_features=min(10, len(feature_names) if feature_names else 10)
        )
        
        # Extract explanation data
        if mode == "classification":
            # For classification, get explanation for top predicted class
            try:
                prediction = model.predict([instance])[0]
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.argmax()
                exp_data = explanation.as_list(label=prediction)
            except:
                # Fallback to first class
                exp_data = explanation.as_list(label=0)
        else:
            # For regression
            exp_data = explanation.as_list()
        
        # Format the explanation data
        lime_features = []
        for feature, weight in exp_data:
            lime_features.append({
                "feature": feature,
                "weight": weight
            })
        
        return {
            "explanation_type": "lime",
            "mode": mode,
            "features": lime_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return {"error": f"Failed to generate LIME explanation: {str(e)}"}

def generate_shap_explanation(model, X, feature_names=None, instance_idx=0):
    """
    Generate a SHAP explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with SHAP explanation results.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP package is not available. Install with 'pip install shap'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
        
        # Choose the right explainer based on model type
        if is_tree_based_model(model):
            explainer = shap.TreeExplainer(model)
        elif is_neural_network(model) and hasattr(model, 'predict'):
            # For neural networks, use KernelExplainer with a small background
            sample_X = X[:50] if hasattr(X, "__len__") and len(X) > 50 else X
            explainer = shap.KernelExplainer(model.predict, sample_X)
        else:
            explainer = shap.Explainer(model)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, use values for predicted class
            prediction = model.predict(instance)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.argmax()
            values = shap_values[prediction]
        else:
            values = shap_values
        
        # Format the explanation
        if len(values.shape) > 1:
            values = values[0]  # Get values for first (only) instance
        
        # Pair with feature names
        if feature_names and len(feature_names) == len(values):
            shap_features = [{"feature": name, "value": float(val)} 
                            for name, val in zip(feature_names, values)]
        else:
            shap_features = [{"feature": f"Feature {i}", "value": float(val)} 
                            for i, val in enumerate(values)]
        
        # Sort by absolute value for importance
        shap_features.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "explanation_type": "shap",
            "features": shap_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return {"error": f"Failed to generate SHAP explanation: {str(e)}"}

def evaluate_explanation_quality(explanation_data):
    """
    Evaluate the quality of model explanations.
    
    :param explanation_data: Dictionary with explanation results.
    :return: Dictionary with explanation quality metrics.
    """
    metrics = {
        "completeness": 0,
        "consistency": 0,
        "compactness": 0,
        "clarity": 0,
        "overall_quality": 0
    }
    
    explanation_type = explanation_data.get("explanation_type", "unknown")
    
    if "error" in explanation_data:
        # Failed explanation
        metrics["error"] = explanation_data["error"]
        metrics["overall_quality"] = 0
        return metrics
    
    # Evaluate LIME explanations
    if explanation_type == "lime":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are weights consistent?
        if features:
            weights = [abs(f["weight"]) for f in features]
            weight_std = np.std(weights)
            weight_mean = np.mean(weights)
            if weight_mean > 0:
                cv = weight_std / weight_mean
                metrics["consistency"] = max(0, 100 - cv * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Evaluate SHAP explanations
    elif explanation_type == "shap":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are values distributed reasonably?
        if features:
            values = [abs(f["value"]) for f in features]
            # Check if there's a good distribution of importance
            sorted_values = sorted(values, reverse=True)
            if sorted_values[0] > 0:
                # Measure how quickly importance drops off
                importance_ratio = sum(sorted_values[1:]) / sorted_values[0]
                metrics["consistency"] = min(100, importance_ratio * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Calculate overall quality
    weights = {
        "completeness": 0.3,
        "consistency": 0.3,
        "compactness": 0.2,
        "clarity": 0.2
    }
    
    metrics["overall_quality"] = sum(metrics[k] * weights[k] for k in weights)
    
    return metrics

def generate_decision_rules(model, feature_names=None):
    """
    Extract interpretable decision rules from a model.
    
    :param model: The model to extract rules from.
    :param feature_names: List of feature names.
    :return: Dictionary with decision rules.
    """
    rules = []
    
    # Only tree-based models and some linear models can generate rules
    if not is_tree_based_model(model) and not is_linear_model(model):
        return {
            "rules": [],
            "error": "Rule extraction is only supported for tree-based and linear models"
        }
    
    try:
        if is_tree_based_model(model):
            # Try to extract rules from tree-based model
            if hasattr(model, "estimators_") and SKLEARN_AVAILABLE:
                # For random forests, extract from first few trees
                from sklearn.tree import _tree
                
                max_trees = min(3, len(model.estimators_))
                for i in range(max_trees):
                    tree = model.estimators_[i]
                    tree_rules = _extract_rules_from_tree(tree, feature_names)
                    rules.extend([f"Tree {i+1}: {rule}" for rule in tree_rules[:5]])  # Limit rules per tree
            
            elif hasattr(model, "tree_") and SKLEARN_AVAILABLE:
                # For single decision tree
                from sklearn.tree import _tree
                rules = _extract_rules_from_tree(model, feature_names)
        
        elif is_linear_model(model):
            # For linear models, create rules based on coefficients
            if hasattr(model, "coef_") and feature_names:
                coefs = model.coef_
                intercept = model.intercept_ if hasattr(model, "intercept_") else 0
                
                # For binary classification or regression
                if len(coefs.shape) == 1:
                    # Sort features by importance (coefficient magnitude)
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    
                    # Create rules for top features
                    for idx in sorted_idx[:5]:  # Limit to 5 most important features
                        feature = feature_names[idx]
                        coef = coefs[idx]
                        if coef > 0:
                            rules.append(f"Higher values of '{feature}' increase the prediction")
                        else:
                            rules.append(f"Higher values of '{feature}' decrease the prediction")
                    
                    # Add intercept information
                    if intercept != 0:
                        rules.append(f"Base value (intercept) is {intercept:.4f}")
                
                # For multi-class classification
                elif len(coefs.shape) == 2:
                    n_classes = coefs.shape[0]
                    for class_idx in range(min(n_classes, 3)):  # Limit to first 3 classes
                        class_rules = []
                        
                        # Sort features by importance for this class
                        sorted_idx = np.argsort(np.abs(coefs[class_idx]))[::-1]
                        
                        # Create rules for top features
                        for idx in sorted_idx[:3]:  # Limit to 3 most important features per class
                            feature = feature_names[idx]
                            coef = coefs[class_idx, idx]
                            if coef > 0:
                                class_rules.append(f"Higher '{feature}' increases probability")
                            else:
                                class_rules.append(f"Higher '{feature}' decreases probability")
                        
                        # Add class-specific intercept
                        if isinstance(intercept, np.ndarray) and len(intercept) > class_idx:
                            class_rules.append(f"Base value: {intercept[class_idx]:.4f}")
                        
                        rules.append(f"Class {class_idx}: " + ", ".join(class_rules))
    
    except Exception as e:
        logger.warning(f"Error extracting decision rules: {str(e)}")
        return {"rules": [], "error": f"Rule extraction failed: {str(e)}"}
    
    return {
        "rules": rules[:10],  # Limit total rules
        "rule_count": len(rules)
    }

def _extract_rules_from_tree(tree_model, feature_names):
    """Helper function to extract rules from a decision tree."""
    if not SKLEARN_AVAILABLE:
        return []
        
    from sklearn.tree import _tree
    
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if feature_names and i != _tree.TREE_UNDEFINED else f"feature {i}"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left path: feature <= threshold
            left_path = path.copy()
            left_path.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], depth + 1, left_path)
            
            # Right path: feature > threshold
            right_path = path.copy()
            right_path.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], depth + 1, right_path)
        else:
            # Leaf node
            if tree_.n_outputs == 1:
                value = tree_.value[node][0][0]
                rule = " AND ".join(path) + f" → {value:.2f}"
                rules.append(rule)
            else:
                # Multi-output: get class with highest probability
                class_idx = np.argmax(tree_.value[node])
                value = tree_.value[node][0][class_idx]
                rule = " AND ".join(path) + f" → Class {class_idx} (prob: {value:.2f})"
                rules.append(rule)
    
    rules = []
    recurse(0, 1, [])
    
    # Sort rules by complexity (number of conditions)
    rules.sort(key=lambda x: x.count("AND"))
    
    return rules

def generate_visualizations(model, X, y=None, feature_names=None, output_dir="explainability_visualizations"):
    """
    Generate visualizations to aid in model understanding.
    
    :param model: The model to visualize.
    :param X: The input data.
    :param y: The target data (optional).
    :param feature_names: List of feature names.
    :param output_dir: Directory to save visualizations.
    :return: Dictionary with visualization information.
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []
    
    # Convert X to numpy if it's pandas
    if hasattr(X, 'values'):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
    
    # Generate feature importance visualization if available
    importance_data = extract_feature_importance(model, feature_names, X, y)
    
    if importance_data["methods_available"]:
        # Use the first available method
        method = importance_data["methods_available"][0]
        importances = importance_data["feature_importance"].get(method, [])
        
        if importances and isinstance(importances, dict):
            # Create feature importance bar chart
            features = list(importances.keys())
            values = list(importances.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            sorted_features = [features[i] for i in sorted_idx[-15:]]  # Top 15 features
            sorted_values = [values[i] for i in sorted_idx[-15:]]
            
            plt.figure(figsize=(10, 8))
            plt.barh(sorted_features, sorted_values)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance ({method})')
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(output_dir, f"feature_importance_{method}.png")
            plt.savefig(viz_path)
            plt.close()
            
            visualizations.append({
                "type": "feature_importance",
                "method": method,
                "path": viz_path
            })
    
    # Generate SHAP visualizations if available
    if SHAP_AVAILABLE and len(X_values) > 0:
        try:
            # Use a sample of the data for SHAP
            sample_size = min(100, len(X_values))
            sample_indices = np.random.choice(len(X_values), sample_size, replace=False)
            X_sample = X_values[sample_indices]
            
            # Create the explainer
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
            else:
                # For other models, use Kernel explainer with background data
                background = shap.kmeans(X_values, 5)  # Use k-means for background data
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class, use class 0
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot, 
                X_sample, 
                feature_names=feature_names,
                show=False
            )
            
            viz_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(viz_path, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                "type": "shap_summary",
                "path": viz_path
            })
            
            # Dependence plot for most important feature
            if feature_names and len(feature_names) > 0:
                # Find most important feature by mean absolute SHAP value
                mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
                top_feature_idx = mean_abs_shap.argmax()
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    top_feature_idx, 
                    shap_values_plot, 
                    X_sample, 
                    feature_names=feature_names,
                    show=False
                )
                
                viz_path = os.path.join(output_dir, "shap_dependence.png")
                plt.savefig(viz_path, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    "type": "shap_dependence",
                    "feature": feature_names[top_feature_idx],
                    "path": viz_path
                })
        
        except Exception as e:
            logger.warning(f"Error generating SHAP visualizations: {str(e)}")
    
    # Generate decision tree visualization if applicable
    if is_tree_based_model(model) and SKLEARN_AVAILABLE:
        try:
            from sklearn.tree import export_graphviz
            import subprocess
            
            # If it's a random forest, visualize the first tree
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                tree_to_viz = model.estimators_[0]
            else:
                tree_to_viz = model
            
            # Export tree as dot file
            dot_path = os.path.join(output_dir, "decision_tree.dot")
            export_graphviz(
                tree_to_viz,
                out_file=dot_path,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                max_depth=3  # Limit depth for readability
            )
            
            # Try to convert to PNG if graphviz is installed
            png_path = os.path.join(output_dir, "decision_tree.png")
            try:
                subprocess.run(
                    ["dot", "-Tpng", dot_path, "-o", png_path],
                    check=True,
                    stderr=subprocess.PIPE
                )
                
                visualizations.append({
                    "type": "decision_tree",
                    "path": png_path
                })
            except (subprocess.SubprocessError, FileNotFoundError):
                # Graphviz not installed or error occurred
                visualizations.append({
                    "type": "decision_tree_dot",
                    "path": dot_path,
                    "note": "Install Graphviz to convert to PNG"
                })
        
        except Exception as e:
            logger.warning(f"Error generating decision tree visualization: {str(e)}")
    
    # Generate partial dependence plots if sklearn is available
    if SKLEARN_AVAILABLE and feature_names and len(feature_names) > 0:
        try:
            from sklearn.inspection import plot_partial_dependence
            
            # Use feature importance to find the top features
            importance_data = extract_feature_importance(model, feature_names, X, y)
            
            if importance_data["methods_available"]:
                method = importance_data["methods_available"][0]
                importances = importance_data["feature_importance"].get(method, [])
                
                if importances:
                    # Get top 2 features
                    if isinstance(importances, dict):
                        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        top_features = [feature_names.index(f[0]) for f in sorted_features[:2] 
                                       if f[0] in feature_names]
                    else:
                        top_features = np.argsort(importances)[-2:]
                    
                    # Generate partial dependence plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_partial_dependence(
                        model, X_values, top_features,
                        feature_names=feature_names,
                        ax=ax
                    )
                    
                    viz_path = os.path.join(output_dir, "partial_dependence.png")
                    plt.savefig(viz_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "partial_dependence",
                        "features": [feature_names[i] for i in top_features],
                        "path": viz_path
                    })
        
        except Exception as e:
            logger.warning(f"Error generating partial dependence plots: {str(e)}")
    
    return {
        "visualizations": visualizations,
        "count": len(visualizations),
        "output_dir": output_dir
    }

def create_human_readable_explanation(model, X, y=None, feature_names=None, instance_idx=0):
    """
    Generate a human-readable explanation of model prediction.
    
    :param model: The model to explain.
    :param X: Input data.
    :param y: Target data (optional).
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with human-readable explanation.
    """
    explanation = {}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
            
            # Get feature values as a dict
            feature_values = instance.iloc[0].to_dict()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
            
            # Create feature value dict
            if feature_names:
                feature_values = {name: instance[0, i] for i, name in enumerate(feature_names)}
            else:
                feature_values = {f"Feature {i}": val for i, val in enumerate(instance[0])}
        
        # Get the model's prediction
        prediction = model.predict(instance)[0]
        
        # Try to get probability if it's a classifier
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(instance)[0]
                probability = np.max(probs)
            except:
                pass
        
        # Add basic prediction info
        explanation["prediction"] = {
            "value": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            "probability": float(probability) if probability is not None else None
        }
        
        # Add instance values
        explanation["instance"] = {
            "index": instance_idx,
            "features": feature_values
        }
        
        # Try to generate LIME explanation
        if LIME_AVAILABLE:
            lime_explanation = generate_lime_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in lime_explanation:
                top_features = lime_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add LIME insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["weight"],
                        "direction": "increases" if feature["weight"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If LIME failed or is not available, try SHAP
        if "key_factors" not in explanation and SHAP_AVAILABLE:
            shap_explanation = generate_shap_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in shap_explanation:
                top_features = shap_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add SHAP insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["value"],
                        "direction": "increases" if feature["value"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If we have key factors, create a natural language explanation
        if "key_factors" in explanation:
            factors = explanation["key_factors"]
            
            # Create natural language summary
            if is_tree_based_model(model):
                model_type = "decision tree" if not hasattr(model, "estimators_") else "random forest"
            elif is_linear_model(model):
                model_type = "linear model"
            elif is_neural_network(model):
                model_type = "neural network"
            else:
                model_type = "model"
                
            # Start with prediction statement
            if "probability" in explanation["prediction"] and explanation["prediction"]["probability"] is not None:
                prob = explanation["prediction"]["probability"]
                confidence = "high" if prob > 0.8 else "moderate" if prob > 0.6 else "low"
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']} with {confidence} confidence ({prob:.2f})."
            else:
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']}."
            
            # Add key factors
            nl_explanation += " This prediction is based on the following factors:\n"
            
            for i, factor in enumerate(factors):
                feature = factor["feature"]
                contrib = abs(factor["contribution"])
                direction = factor["direction"]
                
                # Get the feature value if available
                feature_value = feature_values.get(feature, None)
                value_str = f" (value: {feature_value:.2f})" if feature_value is not None else ""
                
                nl_explanation += f"\n{i+1}. {feature}{value_str} {direction} the prediction"
                
                # Add magnitude description
                if i == 0:  # First factor
                    nl_explanation += " (primary factor)"
                elif contrib < factors[0]["contribution"] * 0.2:  # Small contribution
                    nl_explanation += " (minor factor)"
            
            explanation["natural_language"] = nl_explanation
        
        # Add counterfactual example if we have feature values
        if feature_values:
            # Find the most important feature to change
            if "key_factors" in explanation and explanation["key_factors"]:
                # Use the top factor from our explanation
                top_feature = explanation["key_factors"][0]["feature"]
                direction = explanation["key_factors"][0]["direction"]
                
                # Create counterfactual by modifying the value
                counterfactual = feature_values.copy()
                
                if direction == "increases":
                    # Decrease the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 0.5
                else:
                    # Increase the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 1.5
                
                explanation["counterfactual"] = {
                    "feature_changed": top_feature,
                    "original_value": feature_values[top_feature],
                    "new_value": counterfactual[top_feature],
                    "note": f"Changing {top_feature} might lead to a different prediction"
                }

    except Exception as e:
        logger.error(f"Error creating human readable explanation: {str(e)}")
        explanation["error"] = f"Failed to create explanation: {str(e)}"
    
    return explanation

def generate_structured_report(results, model=None, function=None, output_file="explainability_report.json"):
    """
    Generate a structured report with all explainability results.
    
    :param results: Dictionary of explainability assessment results.
    :param model: The model that was analyzed (optional).
    :param function: The function that was analyzed (optional).
    :param output_file: Path to save the JSON report.
    :return: Dictionary with the full report.
    """
    # Create report structure
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "assessment_type": "model" if model else "function",
        "summary": {},
        "details": results
    }
    
    # Add model or function information
    if model:
        model_info = {
            "type": type(model).__name__,
            "module": model.__module__
        }
        
        # Add additional info based on model type
        if is_tree_based_model(model):
            model_info["category"] = "tree_based"
            
            if hasattr(model, "estimators_"):
                model_info["n_estimators"] = len(model.estimators_)
            
            if hasattr(model, "get_depth"):
                model_info["max_depth"] = model.get_depth()
            elif hasattr(model, "max_depth"):
                model_info["max_depth"] = model.max_depth
            
        elif is_linear_model(model):
            model_info["category"] = "linear"
            
        elif is_neural_network(model):
            model_info["category"] = "neural_network"
        
        report["model_info"] = model_info
    
    if function:
        function_info = {
            "name": function.__name__,
            "module": function.__module__
        }
        
        if "function_complexity" in results:
            function_info.update({
                "complexity_score": results["function_complexity"]["complexity_score"],
                "interpretability": results["function_complexity"]["interpretability"]
            })
        
        report["function_info"] = function_info
    
    # Create summary of key metrics
    summary = {}
    
    # Extract key metrics from results
    if "docstring_quality" in results:
        summary["docstring_quality_score"] = results["docstring_quality"]["quality_score"]
    
    if "function_complexity" in results:
        summary["complexity_score"] = results["function_complexity"]["complexity_score"]
        summary["interpretability"] = results["function_complexity"]["interpretability"]
    
    if "model_structure" in results:
        summary["interpretability_score"] = results["model_structure"]["interpretability_score"]
        summary["interpretability_category"] = results["model_structure"]["interpretability_category"]
    
    if "explanation_quality" in results:
        summary["explanation_quality"] = results["explanation_quality"]["overall_quality"]
    
    if "feature_importance" in results:
        summary["feature_importance_available"] = bool(results["feature_importance"]["methods_available"])
    
    if "decision_rules" in results:
        summary["rule_count"] = results["decision_rules"].get("rule_count", 0)
    
    if "visualizations" in results:
        summary["visualization_count"] = results["visualizations"]["count"]
    
    # Calculate overall explainability score
    explainability_score = 0
    score_components = 0
    
    if "docstring_quality_score" in summary:
        explainability_score += summary["docstring_quality_score"] * 0.1
        score_components += 0.1
    
    if "interpretability" in summary and summary["interpretability"] == "High":
        explainability_score += 100 * 0.2
        score_components += 0.2
    elif "interpretability" in summary and summary["interpretability"] == "Medium":
        explainability_score += 50 * 0.2
        score_components += 0.2
    elif "interpretability" in summary:
        explainability_score += 20 * 0.2
        score_components += 0.2
    
    if "interpretability_score" in summary:
        explainability_score += summary["interpretability_score"] * 0.3
        score_components += 0.3
    
    if "explanation_quality" in summary:
        explainability_score += summary["explanation_quality"] * 0.2
        score_components += 0.2
    
    if "feature_importance_available" in summary and summary["feature_importance_available"]:
        explainability_score += 100 * 0.1
        score_components += 0.1
    
    if "rule_count" in summary and summary["rule_count"] > 0:
        explainability_score += min(100, summary["rule_count"] * 10) * 0.1
        score_components += 0.1
    
    # Normalize score if we have components
    if score_components > 0:
        explainability_score = explainability_score / score_components
    
    # Determine explainability category
    if explainability_score >= 80:
        explainability_category = "High"
    elif explainability_score >= 50:
        explainability_category = "Medium"
    else:
        explainability_category = "Low"
    
    # Add overall score to summary
    summary["explainability_score"] = explainability_score
    summary["explainability_category"] = explainability_category
    
    # Add recommendations based on results
    recommendations = []
    
    if "docstring_quality" in results and results["docstring_quality"]["suggestions"]:
        recommendations.extend(results["docstring_quality"]["suggestions"])
    
    if "function_complexity" in results and results["function_complexity"]["complexity_score"] > 20:
        recommendations.append("Simplify function logic for better interpretability")
    
    if "model_structure" in results and results["model_structure"]["interpretability_category"] == "Low":
        recommendations.append("Consider using a more interpretable model type")
    
    if not summary.get("feature_importance_available", False):
        recommendations.append("Implement feature importance mechanisms")
    
    if "decision_rules" in results and results["decision_rules"].get("rule_count", 0) == 0:
        recommendations.append("Consider models that can provide decision rules")
    
    summary["recommendations"] = recommendations
    
    # Add summary to report
    report["summary"] = summary
    
    # Save report to file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return report

def generate_html_report(results, output_file="explainability_report.html"):
    """
    Generate an HTML report for easier readability.
    
    :param results: Explainability results dictionary.
    :param output_file: Path to save HTML report.
    :return: True if successful, False otherwise.
    """
    try:
        # Define HTML template (simplified version)
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Explainability & Interpretability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2c3e50; }
                .summary { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .metric { margin: 10px 0; }
                .high { color: #27ae60; }
                .medium { color: #f39c12; }
                .low { color: #e74c3c; }
                .recommendations { background-color: #eaf5fb; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .recommendation { margin: 5px 0; }
                .explanation { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .code { font-family: monospace; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Explainability & Interpretability Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                {summary_html}
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
            
            {details_html}
            
        </body>
        </html>
        """
        
        # Format summary section
        summary_html = ""
        if "summary" in results:
            summary = results["summary"]
            
            # Overall score
            if "explainability_score" in summary:
                score = summary["explainability_score"]
                category = summary["explainability_category"]
                summary_html += f"<div class='metric'><strong>Overall Explainability:</strong> "
                summary_html += f"<span class='{category.lower()}'>{score:.1f}/100 ({category})</span></div>"
            
            # Other key metrics
            metrics_to_show = [
                ("docstring_quality_score", "Docstring Quality"),
                ("complexity_score", "Function Complexity"),
                ("interpretability_score", "Model Interpretability"),
                ("explanation_quality", "Explanation Quality")
            ]
            
            for key, label in metrics_to_show:
                if key in summary:
                    summary_html += f"<div class='metric'><strong>{label}:</strong> {summary[key]:.1f}/100</div>"
        
        # Format recommendations section
        recommendations_html = "<ul>"
        if "summary" in results and "recommendations" in results["summary"]:
            for rec in results["summary"]["recommendations"]:
                recommendations_html += f"<li class='recommendation'>{rec}</li>"
        else:
            recommendations_html += "<li>No specific recommendations available.</li>"
        recommendations_html += "</ul>"
        
        # Format details section
        details_html = ""
        
        # Docstring information
        if "docstring_quality" in results:
            details_html += "<h2>Docstring Analysis</h2>"
            doc_quality = results["docstring_quality"]
            
            details_html += "<div class='explanation'>"
            if doc_quality["has_docstring"]:
                details_html += f"<p>Docstring quality score: {doc_quality['quality_score']:.1f}/100</p>"
                details_html += "<ul>"
                details_html += f"<li>Length: {doc_quality['length']} characters</li>"
                details_html += f"<li>Parameters documented: {'Yes' if doc_quality['has_parameters'] else 'No'}</li>"
                details_html += f"<li>Return values documented: {'Yes' if doc_quality['has_return_info'] else 'No'}</li>"
                details_html += f"<li>Examples included: {'Yes' if doc_quality['has_examples'] else 'No'}</li>"
                details_html += "</ul>"
            else:
                details_html += "<p>No docstring found.</p>"
            details_html += "</div>"
        
        # Function complexity
        if "function_complexity" in results:
            details_html += "<h2>Function Complexity Analysis</h2>"
            complexity = results["function_complexity"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Complexity score: {complexity['complexity_score']:.1f}</p>"
            details_html += f"<p>Interpretability: <span class='{complexity['interpretability'].lower()}'>{complexity['interpretability']}</span></p>"
            details_html += "<ul>"
            details_html += f"<li>Lines of code: {complexity['lines_of_code']}</li>"
            details_html += f"<li>Nested depth: {complexity['nested_depth']}</li>"
            details_html += f"<li>Branching factor: {complexity['branching_factor']:.2f}</li>"
            details_html += f"<li>Conditional statements: {complexity.get('conditional_counts', 'N/A')}</li>"
            details_html += f"<li>Loops: {complexity.get('loop_counts', 'N/A')}</li>"
            details_html += "</ul>"
            details_html += "</div>"
        
        # Model structure
        if "model_structure" in results:
            details_html += "<h2>Model Structure Analysis</h2>"
            structure = results["model_structure"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Model type: {structure['model_type']}</p>"
            details_html += f"<p>Interpretability score: {structure['interpretability_score']:.1f}/100</p>"
            details_html += f"<p>Interpretability category: <span class='{structure['interpretability_category'].lower()}'>{structure['interpretability_category']}</span></p>"
            
            # Add structure details
            if "structure_info" in structure and structure["structure_info"]:
                details_html += "<h3>Structure Details</h3>"
                details_html += "<table>"
                details_html += "<tr><th>Property</th><th>Value</th></tr>"
                
                for key, value in structure["structure_info"].items():
                    # Skip complex nested structures
                    if isinstance(value, (dict, list)):
                        continue
                    details_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
                details_html += "</table>"
            
            details_html += "</div>"
        
        # Feature importance
        if "feature_importance" in results:
            details_html += "<h2>Feature Importance Analysis</h2>"
            importance = results["feature_importance"]
            
            details_html += "<div class='explanation'>"
            if importance["methods_available"]:
                details_html += f"<p>Available methods: {', '.join(importance['methods_available'])}</p>"
                
                # Show feature importance for each method
                for method, values in importance["feature_importance"].items():
                    details_html += f"<h3>Method: {method}</h3>"
                    
                    if isinstance(values, dict):
                        # Create a table for feature importance values
                        details_html += "<table>"
                        details_html += "<tr><th>Feature</th><th>Importance</th></tr>"
                        
                        # Sort by importance value
                        sorted_features = sorted(values.items(), key=lambda x: x[1], reverse=True)
                        
                        for feature, value in sorted_features[:10]:  # Show top 10
                            details_html += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>"
                        
                        details_html += "</table>"
                    else:
                        details_html += "<p>Feature importance values available but feature names not provided.</p>"
            else:
                details_html += "<p>No feature importance methods available for this model.</p>"
            
            if "errors" in importance and importance["errors"]:
                details_html += "<h3>Errors</h3>"
                details_html += "<ul>"
                for error in importance["errors"]:
                    details_html += f"<li>{error}</li>"
                details_html += "</ul>"
            
            details_html += "</div>"
        
        # Decision rules
        if "decision_rules" in results:
            details_html += "<h2>Decision Rules</h2>"
            rules = results["decision_rules"]
            
            details_html += "<div class='explanation'>"
            if "rules" in rules and rules["rules"]:
                details_html += f"<p>Number of rules: {rules.get('rule_count', len(rules['rules']))}</p>"
                details_html += "<ol>"
                for rule in rules["rules"]:
                    details_html += f"<li><code>{rule}</code></li>"
                details_html += "</ol>"
            else:
                details_html += "<p>No decision rules available or applicable for this model.</p>"
                if "error" in rules:
                    details_html += f"<p>Error: {rules['error']}</p>"
            details_html += "</div>"
        
        # Human-readable explanation
        if "human_explanation" in results:
            details_html += "<h2>Human-Readable Explanation</h2>"
            explanation = results["human_explanation"]
            
            details_html += "<div class='explanation'>"
            if "error" not in explanation:
                if "natural_language" in explanation:
                    details_html += f"<p>{explanation['natural_language'].replace('\n', '<br>')}</p>"
                
                if "key_factors" in explanation:
                    details_html += "<h3>Key Factors</h3>"
                    details_html += "<table>"
                    details_html += "<tr><th>Feature</th><th>Contribution</th><th>Direction</th></tr>"
                    
                    for factor in explanation["key_factors"]:
                        details_html += (f"<tr><td>{factor['feature']}</td>"
                                        f"<td>{factor['contribution']:.4f}</td>"
                                        f"<td>{factor['direction']}</td></tr>")
                    
                    details_html += "</table>"
                
                if "counterfactual" in explanation:
                    details_html += "<h3>Counterfactual Example</h3>"
                    cf = explanation["counterfactual"]
                    details_html += f"<p>If <strong>{cf['feature_changed']}</strong> were "
                    details_html += f"changed from {cf['original_value']} to {cf['new_value']}, "
                    details_html += "the prediction might change.</p>"
            else:
                details_html += f"<p>Error generating explanation: {explanation['error']}</p>"
            details_html += "</div>"
        
        # Visualizations
        if "visualizations" in results:
            details_html += "<h2>Visualizations</h2>"
            visualizations = results["visualizations"]
            
            if "visualizations" in visualizations and visualizations["visualizations"]:
                for viz in visualizations["visualizations"]:
                    details_html += "<div class='explanation'>"
                    details_html += f"<h3>{viz['type'].replace('_', ' ').title()}</h3>"
                    
                    if "path" in viz:
                        # Check if the path exists and is accessible
                        if os.path.exists(viz["path"]):
                            # For HTML report, use relative paths
                            rel_path = os.path.relpath(viz["path"], os.path.dirname(output_file))
                            details_html += f"<img src='{rel_path}' alt='{viz['type']}' />"
                        else:
                            details_html += f"<p>Visualization file not found: {viz['path']}</p>"
                    
                    if "feature" in viz:
                        details_html += f"<p>Feature: {viz['feature']}</p>"
                    
                    if "note" in viz:
                        details_html += f"<p>Note: {viz['note']}</p>"
                    
                    details_html += "</div>"
            else:
                details_html += "<p>No visualizations available.</p>"
        
        # Fill in the template
        html_content = html_template.format(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_html=summary_html,
            recommendations_html=recommendations_html,
            details_html=details_html
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return False

def evaluate_model_explainability(model, X=None, y=None, feature_names=None, output_dir="explainability_results"):
    """
    Comprehensive assessment of model explainability.
    
    :param model: The model to evaluate.
    :param X: Input data (optional, enables advanced assessments).
    :param y: Target data (optional, enables advanced assessments).
    :param feature_names: List of feature names (optional).
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "model_type": type(model).__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Analyze model structure
    logger.info("Analyzing model structure...")
    results["model_structure"] = analyze_model_structure(model)
    
    # Extract feature importance
    logger.info("Extracting feature importance...")
    results["feature_importance"] = extract_feature_importance(model, feature_names, X, y)
    
    # Generate decision rules if possible
    logger.info("Generating decision rules...")
    results["decision_rules"] = generate_decision_rules(model, feature_names)
    
    # If we have input data, perform more advanced evaluations
    if X is not None:
        # Generate visualizations
        logger.info("Generating visualizations...")
        results["visualizations"] = generate_visualizations(
            model, X, y, feature_names, 
            output_dir=os.path.join(output_dir, "visualizations")
        )
        
        # Generate example-specific explanations
        if len(X) > 0:
            logger.info("Generating instance explanation...")
            # Use the first instance as an example
            results["human_explanation"] = create_human_readable_explanation(
                model, X, y, feature_names, instance_idx=0
            )
            
            # Generate explanation quality assessment
            if LIME_AVAILABLE:
                lime_explanation = generate_lime_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(lime_explanation)
            elif SHAP_AVAILABLE:
                shap_explanation = generate_shap_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(shap_explanation)
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, model=model, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def evaluate_function_explainability(function, output_dir="explainability_results"):
    """
    Comprehensive assessment of function explainability.
    
    :param function: The function to evaluate.
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "function_name": function.__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Check docstring quality
    logger.info("Checking docstring quality...")
    results["docstring_quality"] = check_docstring_quality(function)
    
    # Analyze function complexity
    logger.info("Analyzing function complexity...")
    results["function_complexity"] = analyze_function_complexity(function)
    
    # Extract function source code
    try:
        results["function_source"] = inspect.getsource(function)
    except:
        results["function_source"] = "Source code not available"
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, function=function, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Explainability & Interpretability Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--function",
        help="Function name to analyze"
    )
    
    parser.add_argument(
        "--module",
        help="Module containing the function or model"
    )
    
    parser.add_argument(
        "--model",
        help="Name of model object to analyze (must be in the specified module)"
    )
    
    parser.add_argument(
        "--data",
        help="Path to data file for model analysis (CSV, pickle, etc.)"
    )
    
    parser.add_argument(
        "--target-column",
        help="Name of target column in the data file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="explainability_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# Example function to be analyzed
def example_function(x):
    """
    Simple function that squares a number.
    
    This function takes an input number and returns its square.
    
    :param x: The number to be squared.
    :return: The square of the input number.
    
    Example:
        >>> example_function(4)
        16
    """
    return x ** 2

# Run explainability evaluation
if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Handle function evaluation
    if args.function:
        try:
            if args.module:
                # Import the module containing the function
                module = importlib.import_module(args.module)
                function = getattr(module, args.function)
            else:
                # Use the example function if no module specified
                function = example_function
            
            print(f"Evaluating explainability of function '{function.__name__}'...")
            evaluate_function_explainability(function, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating function: {e}")
            sys.exit(1)
    
    # Handle model evaluation
    elif args.model:
        try:
            if not args.module:
                logger.error("Module must be specified when analyzing a model")
                sys.exit(1)
            
            # Import the module containing the model
            module = importlib.import_module(args.module)
            model = getattr(module, args.model)
            
            # Load data if provided
            X = None
            y = None
            feature_names = None
            
            if args.data:
                print(f"Loading data from {args.data}...")
                
                if args.data.endswith('.csv'):
                    if not PANDAS_AVAILABLE:
                        logger.error("Pandas is required to load CSV files")
                        sys.exit(1)
                    
                    data = pd.read_csv(args.data)
                    
                    if args.target_column and args.target_column in data.columns:
                        y = data[args.target_column]
                        X = data.drop(columns=[args.target_column])
                    else:
                        # Assume all columns are features
                        X = data
                    
                    feature_names = X.columns.tolist()
                
                elif args.data.endswith('.pkl') or args.data.endswith('.pickle'):
                    with open(args.data, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        X = data.get('X') or data.get('data')
                        y = data.get('y') or data.get('target')
                        feature_names = data.get('feature_names')
                    else:
                        # Assume it's just the features
                        X = data
            
            print(f"Evaluating explainability of model '{args.model}'...")
            evaluate_model_explainability(model, X, y, feature_names, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            sys.exit(1)
    
    else:
        # No specific function or model specified, run on example function
        print("No function or model specified, using example function...")
        evaluate_function_explainability(example_function, args.output_dir)
    
    print(f"Explainability assessment complete. Results saved to {args.output_dir}")

"""
TODO:
- Add support for more advanced model types (e.g., ensemble methods, deep learning frameworks)
- Implement benchmarking against "gold standard" human interpretations
- Add support for explainability of embeddings and representations
- Include more advanced LIME and SHAP visualization options
- Implement counterfactual explanation generation with optimization
- Add evaluation of explanation faithfulness and stability
- Support for multi-modal explanations (text + visualization)
- Add natural language explanation generation with templates
- Integrate with popular model registries and governance frameworks
- Extend report generation with more detailed legal and compliance reporting
"""#!/usr/bin/env python3
"""
EXPLAINABILITY & INTERPRETABILITY EVALUATION FRAMEWORK
======================================================

This framework evaluates and enhances the explainability and interpretability of machine learning
models and algorithmic functions. It provides:

1. Comprehensive explainability assessment for different model types
2. Integration with state-of-the-art XAI techniques (SHAP, LIME, etc.)
3. Visualization of feature importance and decision boundaries
4. Quantitative metrics for measuring interpretability
5. Structured reporting for regulatory and legal compliance
6. Human-friendliness evaluation of model explanations
7. Output format optimization for stakeholder understanding

The framework supports various model types including:
- Tree-based models (Random Forests, XGBoost, etc.)
- Linear models (Linear/Logistic Regression, etc.)
- Neural Networks (with activation visualization)
- Black-box models (through post-hoc explanation techniques)
"""

import logging
import json
import inspect
import os
import re
import sys
import argparse
import datetime
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import wraps
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging with both file and console handlers
def setup_logging(log_file='explainability_interpretability.log', console_level=logging.INFO):
    """Setup logging to file and console with different levels."""
    logger = logging.getLogger('explainability')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Utility function to check if a model is a specific type
def is_tree_based_model(model):
    """Check if the model is tree-based."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['tree', 'forest', 'boost', 'xgb', 'lgbm', 'catboost'])

def is_linear_model(model):
    """Check if the model is a linear model."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['linear', 'logistic', 'regression', 'lasso', 'ridge', 'elasticnet'])

def is_neural_network(model):
    """Check if the model is a neural network."""
    # Check for TensorFlow/Keras models
    if TF_AVAILABLE and isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return True
    
    # Check for PyTorch models
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        return True
    
    # Check by class name
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['neural', 'deep', 'mlp', 'keras', 'tensorflow', 'torch', 'nn.'])

# Core explainability assessment functions
def check_docstring_quality(function):
    """
    Evaluate the quality of a function's docstring.
    
    :param function: The function to be checked.
    :return: Dictionary with docstring quality metrics.
    """
    if not function.__doc__:
        return {
            "has_docstring": False,
            "quality_score": 0,
            "length": 0,
            "has_parameters": False,
            "has_return_info": False,
            "has_examples": False,
            "suggestions": ["Add a docstring to explain the function's purpose and usage."]
        }
    
    docstring = function.__doc__.strip()
    
    # Basic metrics
    has_parameters = ":param" in docstring or "Parameters:" in docstring
    has_return_info = ":return" in docstring or "Returns:" in docstring
    has_examples = "Example" in docstring
    length = len(docstring)
    
    # Calculate quality score (0-100)
    quality_score = 0
    if length > 0:
        quality_score += min(length / 5, 30)  # Up to 30 points for length
    if has_parameters:
        quality_score += 25  # 25 points for parameter documentation
    if has_return_info:
        quality_score += 25  # 25 points for return value documentation
    if has_examples:
        quality_score += 20  # 20 points for examples
    
    # Generate suggestions
    suggestions = []
    if not has_parameters:
        suggestions.append("Add parameter descriptions.")
    if not has_return_info:
        suggestions.append("Document return values.")
    if not has_examples:
        suggestions.append("Include usage examples.")
    if length < 50:
        suggestions.append("Expand the description to be more detailed.")
        
    return {
        "has_docstring": True,
        "quality_score": quality_score,
        "length": length,
        "has_parameters": has_parameters,
        "has_return_info": has_return_info,
        "has_examples": has_examples,
        "suggestions": suggestions
    }

def analyze_function_complexity(function):
    """
    Analyze function complexity and structure for explainability.
    
    :param function: The function to be analyzed.
    :return: Dictionary with complexity metrics.
    """
    try:
        source_code = inspect.getsource(function)
    except (TypeError, OSError) as e:
        logger.warning(f"Could not retrieve source code: {e}")
        return {
            "complexity_score": None,
            "lines_of_code": None,
            "nested_depth": None,
            "branching_factor": None,
            "error": str(e)
        }
    
    # Count lines of code
    lines = source_code.strip().split('\n')
    lines_of_code = len(lines)
    
    # Count conditional statements (if, elif, else)
    conditional_counts = len(re.findall(r'\bif\b|\belif\b|\belse\b', source_code))
    
    # Count loops (for, while)
    loop_counts = len(re.findall(r'\bfor\b|\bwhile\b', source_code))
    
    # Estimate nesting depth
    indent_levels = [len(line) - len(line.lstrip()) for line in lines]
    max_indent = max(indent_levels) if indent_levels else 0
    nested_depth = max_indent // 4  # Assuming 4 spaces per indentation level
    
    # Calculate complexity score (higher means more complex, less interpretable)
    complexity_score = (
        conditional_counts * 2 + 
        loop_counts * 3 + 
        nested_depth * 5 + 
        lines_of_code / 10
    )
    
    # Calculate branching factor
    branching_factor = conditional_counts / lines_of_code if lines_of_code > 0 else 0
    
    # Interpretability assessment
    if complexity_score < 10:
        interpretability = "High"
    elif complexity_score < 25:
        interpretability = "Medium"
    else:
        interpretability = "Low"
        
    return {
        "complexity_score": complexity_score,
        "interpretability": interpretability,
        "lines_of_code": lines_of_code,
        "nested_depth": nested_depth,
        "branching_factor": branching_factor,
        "conditional_counts": conditional_counts,
        "loop_counts": loop_counts
    }

def extract_feature_importance(model, feature_names=None, X=None, y=None):
    """
    Extract feature importance from a model using appropriate method.
    
    :param model: The trained model to analyze.
    :param feature_names: List of feature names.
    :param X: Input data for permutation importance (optional).
    :param y: Target data for permutation importance (optional).
    :return: Dictionary with feature importance information.
    """
    importance_methods = []
    importance_values = {}
    error_messages = []
    
    # Try built-in feature importance
    if hasattr(model, "feature_importances_"):
        importance_methods.append("built_in")
        if feature_names and len(model.feature_importances_) == len(feature_names):
            importance_values["built_in"] = dict(zip(feature_names, model.feature_importances_.tolist()))
        else:
            importance_values["built_in"] = model.feature_importances_.tolist()
    
    # Try coefficients for linear models
    if hasattr(model, "coef_"):
        importance_methods.append("coefficients")
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = np.mean(np.abs(coefs), axis=0)  # For multi-class models
            
        if feature_names and len(coefs) == len(feature_names):
            importance_values["coefficients"] = dict(zip(feature_names, coefs.tolist()))
        else:
            importance_values["coefficients"] = coefs.tolist()
    
    # Try permutation importance if sklearn is available and data is provided
    if SKLEARN_AVAILABLE and X is not None and y is not None:
        try:
            importance_methods.append("permutation")
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            
            if feature_names and len(perm_importance.importances_mean) == len(feature_names):
                importance_values["permutation"] = dict(zip(feature_names, perm_importance.importances_mean.tolist()))
            else:
                importance_values["permutation"] = perm_importance.importances_mean.tolist()
        except Exception as e:
            error_messages.append(f"Permutation importance failed: {str(e)}")
    
    # Try SHAP if available and appropriate
    if SHAP_AVAILABLE and X is not None:
        try:
            sample_X = X[:100] if hasattr(X, "__len__") and len(X) > 100 else X
            
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
                importance_methods.append("shap_tree")
            else:
                explainer = shap.Explainer(model)
                importance_methods.append("shap")
                
            shap_values = explainer.shap_values(sample_X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For multi-class models, take mean absolute value across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            # Calculate feature importance as mean absolute SHAP value
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            if feature_names and len(mean_abs_shap) == len(feature_names):
                importance_values["shap"] = dict(zip(feature_names, mean_abs_shap.tolist()))
            else:
                importance_values["shap"] = mean_abs_shap.tolist()
                
        except Exception as e:
            error_messages.append(f"SHAP analysis failed: {str(e)}")
    
    return {
        "methods_available": importance_methods,
        "feature_importance": importance_values,
        "errors": error_messages
    }

def analyze_model_structure(model):
    """
    Analyze the structure of a model to assess interpretability.
    
    :param model: The model to analyze.
    :return: Dictionary with model structure information.
    """
    model_type = "unknown"
    interpretability_score = 0
    structure_info = {}
    
    # Determine model type
    if is_tree_based_model(model):
        model_type = "tree_based"
        
        # Extract tree structure information
        if hasattr(model, "tree_") or hasattr(model, "estimators_"):
            n_nodes = 0
            max_depth = 0
            
            if hasattr(model, "tree_"):
                # Single tree model
                n_nodes = model.tree_.node_count
                max_depth = model.get_depth() if hasattr(model, "get_depth") else model.tree_.max_depth
                structure_info["n_nodes"] = n_nodes
                structure_info["max_depth"] = max_depth
            
            elif hasattr(model, "estimators_"):
                # Ensemble of trees
                n_estimators = len(model.estimators_)
                structure_info["n_estimators"] = n_estimators
                
                depths = []
                nodes = []
                
                for estimator in model.estimators_:
                    if hasattr(estimator, "tree_"):
                        nodes.append(estimator.tree_.node_count)
                        if hasattr(estimator, "get_depth"):
                            depths.append(estimator.get_depth())
                        else:
                            depths.append(estimator.tree_.max_depth)
                
                if depths:
                    max_depth = max(depths)
                    structure_info["max_depth"] = max_depth
                    structure_info["mean_depth"] = sum(depths) / len(depths)
                
                if nodes:
                    n_nodes = sum(nodes)
                    structure_info["total_nodes"] = n_nodes
                    structure_info["mean_nodes_per_tree"] = n_nodes / n_estimators
            
            # Calculate interpretability score for tree models (higher is more interpretable)
            if max_depth > 0:
                depth_penalty = max(0, (max_depth - 3) * 10)  # Deeper trees are less interpretable
                interpretability_score = max(0, 100 - depth_penalty)
                
                if "n_estimators" in structure_info:
                    # Ensemble models are less interpretable
                    ensemble_penalty = min(80, structure_info["n_estimators"] * 2)
                    interpretability_score = max(0, interpretability_score - ensemble_penalty)
        
    elif is_linear_model(model):
        model_type = "linear"
        
        # Linear models are generally interpretable
        interpretability_score = 80
        
        # Check for sparsity in linear models
        if hasattr(model, "coef_"):
            coefs = model.coef_
            n_features = coefs.size if len(coefs.shape) == 1 else coefs.shape[1]
            n_nonzero = np.count_nonzero(coefs)
            sparsity = 1.0 - (n_nonzero / n_features) if n_features > 0 else 0
            
            structure_info["n_features"] = n_features
            structure_info["n_nonzero_coefficients"] = n_nonzero
            structure_info["sparsity"] = sparsity
            
            # Sparse models are more interpretable
            if sparsity > 0.5:
                interpretability_score += 15
            elif sparsity < 0.2:
                interpretability_score -= 10
    
    elif is_neural_network(model):
        model_type = "neural_network"
        
        # Neural networks are generally less interpretable
        interpretability_score = 30
        
        # Analyze network architecture
        if TF_AVAILABLE and isinstance(model, tf.keras.Model):
            # For TensorFlow/Keras models
            structure_info["layers"] = []
            total_params = 0
            
            for i, layer in enumerate(model.layers):
                layer_info = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape)
                }
                
                if hasattr(layer, "count_params"):
                    params = layer.count_params()
                    layer_info["parameters"] = params
                    total_params += params
                
                structure_info["layers"].append(layer_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(model.layers) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
        
        elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            # For PyTorch models
            structure_info["modules"] = []
            total_params = 0
            
            for name, module in model.named_modules():
                if list(module.children()):  # Skip container modules
                    continue
                    
                module_info = {
                    "name": name,
                    "type": module.__class__.__name__
                }
                
                params = sum(p.numel() for p in module.parameters())
                module_info["parameters"] = params
                total_params += params
                
                structure_info["modules"].append(module_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(structure_info["modules"]) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
    
    else:
        # Unknown model type
        model_type = "unknown"
        interpretability_score = 40  # Default score for unknown models
        
        # Try to extract some generic information
        model_attributes = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
        for attr in model_attributes:
            try:
                val = getattr(model, attr)
                if isinstance(val, (int, float, str, bool)):
                    structure_info[attr] = val
            except:
                pass
    
    # Determine interpretability category
    if interpretability_score >= 70:
        interpretability_category = "High"
    elif interpretability_score >= 40:
        interpretability_category = "Medium"
    else:
        interpretability_category = "Low"
        
    return {
        "model_type": model_type,
        "interpretability_score": interpretability_score,
        "interpretability_category": interpretability_category,
        "structure_info": structure_info
    }

def generate_lime_explanation(model, X, feature_names=None, class_names=None, instance_idx=0):
    """
    Generate a LIME explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param class_names: List of class names for classification.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with LIME explanation results.
    """
    if not LIME_AVAILABLE:
        return {"error": "LIME package is not available. Install with 'pip install lime'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx].values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx]
        
        # Determine if we're doing classification or regression
        # This is a simplification - in practice, you might need to check model type
        mode = "classification"
        if class_names is None:
            # Try to guess if it's binary classification
            mode = "classification"
            class_names = [0, 1]
        
        # Create the LIME explainer
        explainer = LimeTabularExplainer(
            X if isinstance(X, np.ndarray) else X.values,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode
        )
        
        # Define prediction function based on expected input
        if mode == "classification":
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:
                # Fallback to non-probabilistic prediction
                def predict_fn(x):
                    preds = model.predict(x)
                    # Convert to pseudo-probabilities for binary classification
                    if len(preds.shape) == 1 and len(class_names) == 2:
                        return np.vstack([(1-preds), preds]).T
                    return preds
        else:
            predict_fn = model.predict
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance, 
            predict_fn,
            num_features=min(10, len(feature_names) if feature_names else 10)
        )
        
        # Extract explanation data
        if mode == "classification":
            # For classification, get explanation for top predicted class
            try:
                prediction = model.predict([instance])[0]
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.argmax()
                exp_data = explanation.as_list(label=prediction)
            except:
                # Fallback to first class
                exp_data = explanation.as_list(label=0)
        else:
            # For regression
            exp_data = explanation.as_list()
        
        # Format the explanation data
        lime_features = []
        for feature, weight in exp_data:
            lime_features.append({
                "feature": feature,
                "weight": weight
            })
        
        return {
            "explanation_type": "lime",
            "mode": mode,
            "features": lime_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return {"error": f"Failed to generate LIME explanation: {str(e)}"}

def generate_shap_explanation(model, X, feature_names=None, instance_idx=0):
    """
    Generate a SHAP explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with SHAP explanation results.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP package is not available. Install with 'pip install shap'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
        
        # Choose the right explainer based on model type
        if is_tree_based_model(model):
            explainer = shap.TreeExplainer(model)
        elif is_neural_network(model) and hasattr(model, 'predict'):
            # For neural networks, use KernelExplainer with a small background
            sample_X = X[:50] if hasattr(X, "__len__") and len(X) > 50 else X
            explainer = shap.KernelExplainer(model.predict, sample_X)
        else:
            explainer = shap.Explainer(model)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, use values for predicted class
            prediction = model.predict(instance)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.argmax()
            values = shap_values[prediction]
        else:
            values = shap_values
        
        # Format the explanation
        if len(values.shape) > 1:
            values = values[0]  # Get values for first (only) instance
        
        # Pair with feature names
        if feature_names and len(feature_names) == len(values):
            shap_features = [{"feature": name, "value": float(val)} 
                            for name, val in zip(feature_names, values)]
        else:
            shap_features = [{"feature": f"Feature {i}", "value": float(val)} 
                            for i, val in enumerate(values)]
        
        # Sort by absolute value for importance
        shap_features.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "explanation_type": "shap",
            "features": shap_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return {"error": f"Failed to generate SHAP explanation: {str(e)}"}

def evaluate_explanation_quality(explanation_data):
    """
    Evaluate the quality of model explanations.
    
    :param explanation_data: Dictionary with explanation results.
    :return: Dictionary with explanation quality metrics.
    """
    metrics = {
        "completeness": 0,
        "consistency": 0,
        "compactness": 0,
        "clarity": 0,
        "overall_quality": 0
    }
    
    explanation_type = explanation_data.get("explanation_type", "unknown")
    
    if "error" in explanation_data:
        # Failed explanation
        metrics["error"] = explanation_data["error"]
        metrics["overall_quality"] = 0
        return metrics
    
    # Evaluate LIME explanations
    if explanation_type == "lime":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are weights consistent?
        if features:
            weights = [abs(f["weight"]) for f in features]
            weight_std = np.std(weights)
            weight_mean = np.mean(weights)
            if weight_mean > 0:
                cv = weight_std / weight_mean
                metrics["consistency"] = max(0, 100 - cv * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Evaluate SHAP explanations
    elif explanation_type == "shap":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are values distributed reasonably?
        if features:
            values = [abs(f["value"]) for f in features]
            # Check if there's a good distribution of importance
            sorted_values = sorted(values, reverse=True)
            if sorted_values[0] > 0:
                # Measure how quickly importance drops off
                importance_ratio = sum(sorted_values[1:]) / sorted_values[0]
                metrics["consistency"] = min(100, importance_ratio * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Calculate overall quality
    weights = {
        "completeness": 0.3,
        "consistency": 0.3,
        "compactness": 0.2,
        "clarity": 0.2
    }
    
    metrics["overall_quality"] = sum(metrics[k] * weights[k] for k in weights)
    
    return metrics

def generate_decision_rules(model, feature_names=None):
    """
    Extract interpretable decision rules from a model.
    
    :param model: The model to extract rules from.
    :param feature_names: List of feature names.
    :return: Dictionary with decision rules.
    """
    rules = []
    
    # Only tree-based models and some linear models can generate rules
    if not is_tree_based_model(model) and not is_linear_model(model):
        return {
            "rules": [],
            "error": "Rule extraction is only supported for tree-based and linear models"
        }
    
    try:
        if is_tree_based_model(model):
            # Try to extract rules from tree-based model
            if hasattr(model, "estimators_") and SKLEARN_AVAILABLE:
                # For random forests, extract from first few trees
                from sklearn.tree import _tree
                
                max_trees = min(3, len(model.estimators_))
                for i in range(max_trees):
                    tree = model.estimators_[i]
                    tree_rules = _extract_rules_from_tree(tree, feature_names)
                    rules.extend([f"Tree {i+1}: {rule}" for rule in tree_rules[:5]])  # Limit rules per tree
            
            elif hasattr(model, "tree_") and SKLEARN_AVAILABLE:
                # For single decision tree
                from sklearn.tree import _tree
                rules = _extract_rules_from_tree(model, feature_names)
        
        elif is_linear_model(model):
            # For linear models, create rules based on coefficients
            if hasattr(model, "coef_") and feature_names:
                coefs = model.coef_
                intercept = model.intercept_ if hasattr(model, "intercept_") else 0
                
                # For binary classification or regression
                if len(coefs.shape) == 1:
                    # Sort features by importance (coefficient magnitude)
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    
                    # Create rules for top features
                    for idx in sorted_idx[:5]:  # Limit to 5 most important features
                        feature = feature_names[idx]
                        coef = coefs[idx]
                        if coef > 0:
                            rules.append(f"Higher values of '{feature}' increase the prediction")
                        else:
                            rules.append(f"Higher values of '{feature}' decrease the prediction")
                    
                    # Add intercept information
                    if intercept != 0:
                        rules.append(f"Base value (intercept) is {intercept:.4f}")
                
                # For multi-class classification
                elif len(coefs.shape) == 2:
                    n_classes = coefs.shape[0]
                    for class_idx in range(min(n_classes, 3)):  # Limit to first 3 classes
                        class_rules = []
                        
                        # Sort features by importance for this class
                        sorted_idx = np.argsort(np.abs(coefs[class_idx]))[::-1]
                        
                        # Create rules for top features
                        for idx in sorted_idx[:3]:  # Limit to 3 most important features per class
                            feature = feature_names[idx]
                            coef = coefs[class_idx, idx]
                            if coef > 0:
                                class_rules.append(f"Higher '{feature}' increases probability")
                            else:
                                class_rules.append(f"Higher '{feature}' decreases probability")
                        
                        # Add class-specific intercept
                        if isinstance(intercept, np.ndarray) and len(intercept) > class_idx:
                            class_rules.append(f"Base value: {intercept[class_idx]:.4f}")
                        
                        rules.append(f"Class {class_idx}: " + ", ".join(class_rules))
    
    except Exception as e:
        logger.warning(f"Error extracting decision rules: {str(e)}")
        return {"rules": [], "error": f"Rule extraction failed: {str(e)}"}
    
    return {
        "rules": rules[:10],  # Limit total rules
        "rule_count": len(rules)
    }

def _extract_rules_from_tree(tree_model, feature_names):
    """Helper function to extract rules from a decision tree."""
    if not SKLEARN_AVAILABLE:
        return []
        
    from sklearn.tree import _tree
    
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if feature_names and i != _tree.TREE_UNDEFINED else f"feature {i}"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left path: feature <= threshold
            left_path = path.copy()
            left_path.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], depth + 1, left_path)
            
            # Right path: feature > threshold
            right_path = path.copy()
            right_path.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], depth + 1, right_path)
        else:
            # Leaf node
            if tree_.n_outputs == 1:
                value = tree_.value[node][0][0]
                rule = " AND ".join(path) + f" → {value:.2f}"
                rules.append(rule)
            else:
                # Multi-output: get class with highest probability
                class_idx = np.argmax(tree_.value[node])
                value = tree_.value[node][0][class_idx]
                rule = " AND ".join(path) + f" → Class {class_idx} (prob: {value:.2f})"
                rules.append(rule)
    
    rules = []
    recurse(0, 1, [])
    
    # Sort rules by complexity (number of conditions)
    rules.sort(key=lambda x: x.count("AND"))
    
    return rules

def generate_visualizations(model, X, y=None, feature_names=None, output_dir="explainability_visualizations"):
    """
    Generate visualizations to aid in model understanding.
    
    :param model: The model to visualize.
    :param X: The input data.
    :param y: The target data (optional).
    :param feature_names: List of feature names.
    :param output_dir: Directory to save visualizations.
    :return: Dictionary with visualization information.
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []
    
    # Convert X to numpy if it's pandas
    if hasattr(X, 'values'):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
    
    # Generate feature importance visualization if available
    importance_data = extract_feature_importance(model, feature_names, X, y)
    
    if importance_data["methods_available"]:
        # Use the first available method
        method = importance_data["methods_available"][0]
        importances = importance_data["feature_importance"].get(method, [])
        
        if importances and isinstance(importances, dict):
            # Create feature importance bar chart
            features = list(importances.keys())
            values = list(importances.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            sorted_features = [features[i] for i in sorted_idx[-15:]]  # Top 15 features
            sorted_values = [values[i] for i in sorted_idx[-15:]]
            
            plt.figure(figsize=(10, 8))
            plt.barh(sorted_features, sorted_values)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance ({method})')
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(output_dir, f"feature_importance_{method}.png")
            plt.savefig(viz_path)
            plt.close()
            
            visualizations.append({
                "type": "feature_importance",
                "method": method,
                "path": viz_path
            })
    
    # Generate SHAP visualizations if available
    if SHAP_AVAILABLE and len(X_values) > 0:
        try:
            # Use a sample of the data for SHAP
            sample_size = min(100, len(X_values))
            sample_indices = np.random.choice(len(X_values), sample_size, replace=False)
            X_sample = X_values[sample_indices]
            
            # Create the explainer
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
            else:
                # For other models, use Kernel explainer with background data
                background = shap.kmeans(X_values, 5)  # Use k-means for background data
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class, use class 0
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot, 
                X_sample, 
                feature_names=feature_names,
                show=False
            )
            
            viz_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(viz_path, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                "type": "shap_summary",
                "path": viz_path
            })
            
            # Dependence plot for most important feature
            if feature_names and len(feature_names) > 0:
                # Find most important feature by mean absolute SHAP value
                mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
                top_feature_idx = mean_abs_shap.argmax()
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    top_feature_idx, 
                    shap_values_plot, 
                    X_sample, 
                    feature_names=feature_names,
                    show=False
                )
                
                viz_path = os.path.join(output_dir, "shap_dependence.png")
                plt.savefig(viz_path, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    "type": "shap_dependence",
                    "feature": feature_names[top_feature_idx],
                    "path": viz_path
                })
        
        except Exception as e:
            logger.warning(f"Error generating SHAP visualizations: {str(e)}")
    
    # Generate decision tree visualization if applicable
    if is_tree_based_model(model) and SKLEARN_AVAILABLE:
        try:
            from sklearn.tree import export_graphviz
            import subprocess
            
            # If it's a random forest, visualize the first tree
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                tree_to_viz = model.estimators_[0]
            else:
                tree_to_viz = model
            
            # Export tree as dot file
            dot_path = os.path.join(output_dir, "decision_tree.dot")
            export_graphviz(
                tree_to_viz,
                out_file=dot_path,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                max_depth=3  # Limit depth for readability
            )
            
            # Try to convert to PNG if graphviz is installed
            png_path = os.path.join(output_dir, "decision_tree.png")
            try:
                subprocess.run(
                    ["dot", "-Tpng", dot_path, "-o", png_path],
                    check=True,
                    stderr=subprocess.PIPE
                )
                
                visualizations.append({
                    "type": "decision_tree",
                    "path": png_path
                })
            except (subprocess.SubprocessError, FileNotFoundError):
                # Graphviz not installed or error occurred
                visualizations.append({
                    "type": "decision_tree_dot",
                    "path": dot_path,
                    "note": "Install Graphviz to convert to PNG"
                })
        
        except Exception as e:
            logger.warning(f"Error generating decision tree visualization: {str(e)}")
    
    # Generate partial dependence plots if sklearn is available
    if SKLEARN_AVAILABLE and feature_names and len(feature_names) > 0:
        try:
            from sklearn.inspection import plot_partial_dependence
            
            # Use feature importance to find the top features
            importance_data = extract_feature_importance(model, feature_names, X, y)
            
            if importance_data["methods_available"]:
                method = importance_data["methods_available"][0]
                importances = importance_data["feature_importance"].get(method, [])
                
                if importances:
                    # Get top 2 features
                    if isinstance(importances, dict):
                        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        top_features = [feature_names.index(f[0]) for f in sorted_features[:2] 
                                       if f[0] in feature_names]
                    else:
                        top_features = np.argsort(importances)[-2:]
                    
                    # Generate partial dependence plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_partial_dependence(
                        model, X_values, top_features,
                        feature_names=feature_names,
                        ax=ax
                    )
                    
                    viz_path = os.path.join(output_dir, "partial_dependence.png")
                    plt.savefig(viz_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "partial_dependence",
                        "features": [feature_names[i] for i in top_features],
                        "path": viz_path
                    })
        
        except Exception as e:
            logger.warning(f"Error generating partial dependence plots: {str(e)}")
    
    return {
        "visualizations": visualizations,
        "count": len(visualizations),
        "output_dir": output_dir
    }

def create_human_readable_explanation(model, X, y=None, feature_names=None, instance_idx=0):
    """
    Generate a human-readable explanation of model prediction.
    
    :param model: The model to explain.
    :param X: Input data.
    :param y: Target data (optional).
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with human-readable explanation.
    """
    explanation = {}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
            
            # Get feature values as a dict
            feature_values = instance.iloc[0].to_dict()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
            
            # Create feature value dict
            if feature_names:
                feature_values = {name: instance[0, i] for i, name in enumerate(feature_names)}
            else:
                feature_values = {f"Feature {i}": val for i, val in enumerate(instance[0])}
        
        # Get the model's prediction
        prediction = model.predict(instance)[0]
        
        # Try to get probability if it's a classifier
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(instance)[0]
                probability = np.max(probs)
            except:
                pass
        
        # Add basic prediction info
        explanation["prediction"] = {
            "value": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            "probability": float(probability) if probability is not None else None
        }
        
        # Add instance values
        explanation["instance"] = {
            "index": instance_idx,
            "features": feature_values
        }
        
        # Try to generate LIME explanation
        if LIME_AVAILABLE:
            lime_explanation = generate_lime_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in lime_explanation:
                top_features = lime_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add LIME insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["weight"],
                        "direction": "increases" if feature["weight"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If LIME failed or is not available, try SHAP
        if "key_factors" not in explanation and SHAP_AVAILABLE:
            shap_explanation = generate_shap_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in shap_explanation:
                top_features = shap_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add SHAP insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["value"],
                        "direction": "increases" if feature["value"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If we have key factors, create a natural language explanation
        if "key_factors" in explanation:
            factors = explanation["key_factors"]
            
            # Create natural language summary
            if is_tree_based_model(model):
                model_type = "decision tree" if not hasattr(model, "estimators_") else "random forest"
            elif is_linear_model(model):
                model_type = "linear model"
            elif is_neural_network(model):
                model_type = "neural network"
            else:
                model_type = "model"
                
            # Start with prediction statement
            if "probability" in explanation["prediction"] and explanation["prediction"]["probability"] is not None:
                prob = explanation["prediction"]["probability"]
                confidence = "high" if prob > 0.8 else "moderate" if prob > 0.6 else "low"
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']} with {confidence} confidence ({prob:.2f})."
            else:
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']}."
            
            # Add key factors
            nl_explanation += " This prediction is based on the following factors:\n"
            
            for i, factor in enumerate(factors):
                feature = factor["feature"]
                contrib = abs(factor["contribution"])
                direction = factor["direction"]
                
                # Get the feature value if available
                feature_value = feature_values.get(feature, None)
                value_str = f" (value: {feature_value:.2f})" if feature_value is not None else ""
                
                nl_explanation += f"\n{i+1}. {feature}{value_str} {direction} the prediction"
                
                # Add magnitude description
                if i == 0:  # First factor
                    nl_explanation += " (primary factor)"
                elif contrib < factors[0]["contribution"] * 0.2:  # Small contribution
                    nl_explanation += " (minor factor)"
            
            explanation["natural_language"] = nl_explanation
        
        # Add counterfactual example if we have feature values
        if feature_values:
            # Find the most important feature to change
            if "key_factors" in explanation and explanation["key_factors"]:
                # Use the top factor from our explanation
                top_feature = explanation["key_factors"][0]["feature"]
                direction = explanation["key_factors"][0]["direction"]
                
                # Create counterfactual by modifying the value
                counterfactual = feature_values.copy()
                
                if direction == "increases":
                    # Decrease the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 0.5
                else:
                    # Increase the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 1.5
                
                explanation["counterfactual"] = {
                    "feature_changed": top_feature,
                    "original_value": feature_values[top_feature],
                    "new_value": counterfactual[top_feature],
                    "note": f"Changing {top_feature} might lead to a different prediction"
                }

    except Exception as e:
        logger.error(f"Error creating human readable explanation: {str(e)}")
        explanation["error"] = f"Failed to create explanation: {str(e)}"
    
    return explanation

def generate_structured_report(results, model=None, function=None, output_file="explainability_report.json"):
    """
    Generate a structured report with all explainability results.
    
    :param results: Dictionary of explainability assessment results.
    :param model: The model that was analyzed (optional).
    :param function: The function that was analyzed (optional).
    :param output_file: Path to save the JSON report.
    :return: Dictionary with the full report.
    """
    # Create report structure
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "assessment_type": "model" if model else "function",
        "summary": {},
        "details": results
    }
    
    # Add model or function information
    if model:
        model_info = {
            "type": type(model).__name__,
            "module": model.__module__
        }
        
        # Add additional info based on model type
        if is_tree_based_model(model):
            model_info["category"] = "tree_based"
            
            if hasattr(model, "estimators_"):
                model_info["n_estimators"] = len(model.estimators_)
            
            if hasattr(model, "get_depth"):
                model_info["max_depth"] = model.get_depth()
            elif hasattr(model, "max_depth"):
                model_info["max_depth"] = model.max_depth
            
        elif is_linear_model(model):
            model_info["category"] = "linear"
            
        elif is_neural_network(model):
            model_info["category"] = "neural_network"
        
        report["model_info"] = model_info
    
    if function:
        function_info = {
            "name": function.__name__,
            "module": function.__module__
        }
        
        if "function_complexity" in results:
            function_info.update({
                "complexity_score": results["function_complexity"]["complexity_score"],
                "interpretability": results["function_complexity"]["interpretability"]
            })
        
        report["function_info"] = function_info
    
    # Create summary of key metrics
    summary = {}
    
    # Extract key metrics from results
    if "docstring_quality" in results:
        summary["docstring_quality_score"] = results["docstring_quality"]["quality_score"]
    
    if "function_complexity" in results:
        summary["complexity_score"] = results["function_complexity"]["complexity_score"]
        summary["interpretability"] = results["function_complexity"]["interpretability"]
    
    if "model_structure" in results:
        summary["interpretability_score"] = results["model_structure"]["interpretability_score"]
        summary["interpretability_category"] = results["model_structure"]["interpretability_category"]
    
    if "explanation_quality" in results:
        summary["explanation_quality"] = results["explanation_quality"]["overall_quality"]
    
    if "feature_importance" in results:
        summary["feature_importance_available"] = bool(results["feature_importance"]["methods_available"])
    
    if "decision_rules" in results:
        summary["rule_count"] = results["decision_rules"].get("rule_count", 0)
    
    if "visualizations" in results:
        summary["visualization_count"] = results["visualizations"]["count"]
    
    # Calculate overall explainability score
    explainability_score = 0
    score_components = 0
    
    if "docstring_quality_score" in summary:
        explainability_score += summary["docstring_quality_score"] * 0.1
        score_components += 0.1
    
    if "interpretability" in summary and summary["interpretability"] == "High":
        explainability_score += 100 * 0.2
        score_components += 0.2
    elif "interpretability" in summary and summary["interpretability"] == "Medium":
        explainability_score += 50 * 0.2
        score_components += 0.2
    elif "interpretability" in summary:
        explainability_score += 20 * 0.2
        score_components += 0.2
    
    if "interpretability_score" in summary:
        explainability_score += summary["interpretability_score"] * 0.3
        score_components += 0.3
    
    if "explanation_quality" in summary:
        explainability_score += summary["explanation_quality"] * 0.2
        score_components += 0.2
    
    if "feature_importance_available" in summary and summary["feature_importance_available"]:
        explainability_score += 100 * 0.1
        score_components += 0.1
    
    if "rule_count" in summary and summary["rule_count"] > 0:
        explainability_score += min(100, summary["rule_count"] * 10) * 0.1
        score_components += 0.1
    
    # Normalize score if we have components
    if score_components > 0:
        explainability_score = explainability_score / score_components
    
    # Determine explainability category
    if explainability_score >= 80:
        explainability_category = "High"
    elif explainability_score >= 50:
        explainability_category = "Medium"
    else:
        explainability_category = "Low"
    
    # Add overall score to summary
    summary["explainability_score"] = explainability_score
    summary["explainability_category"] = explainability_category
    
    # Add recommendations based on results
    recommendations = []
    
    if "docstring_quality" in results and results["docstring_quality"]["suggestions"]:
        recommendations.extend(results["docstring_quality"]["suggestions"])
    
    if "function_complexity" in results and results["function_complexity"]["complexity_score"] > 20:
        recommendations.append("Simplify function logic for better interpretability")
    
    if "model_structure" in results and results["model_structure"]["interpretability_category"] == "Low":
        recommendations.append("Consider using a more interpretable model type")
    
    if not summary.get("feature_importance_available", False):
        recommendations.append("Implement feature importance mechanisms")
    
    if "decision_rules" in results and results["decision_rules"].get("rule_count", 0) == 0:
        recommendations.append("Consider models that can provide decision rules")
    
    summary["recommendations"] = recommendations
    
    # Add summary to report
    report["summary"] = summary
    
    # Save report to file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return report

def generate_html_report(results, output_file="explainability_report.html"):
    """
    Generate an HTML report for easier readability.
    
    :param results: Explainability results dictionary.
    :param output_file: Path to save HTML report.
    :return: True if successful, False otherwise.
    """
    try:
        # Define HTML template (simplified version)
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Explainability & Interpretability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2c3e50; }
                .summary { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .metric { margin: 10px 0; }
                .high { color: #27ae60; }
                .medium { color: #f39c12; }
                .low { color: #e74c3c; }
                .recommendations { background-color: #eaf5fb; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .recommendation { margin: 5px 0; }
                .explanation { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .code { font-family: monospace; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Explainability & Interpretability Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                {summary_html}
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
            
            {details_html}
            
        </body>
        </html>
        """
        
        # Format summary section
        summary_html = ""
        if "summary" in results:
            summary = results["summary"]
            
            # Overall score
            if "explainability_score" in summary:
                score = summary["explainability_score"]
                category = summary["explainability_category"]
                summary_html += f"<div class='metric'><strong>Overall Explainability:</strong> "
                summary_html += f"<span class='{category.lower()}'>{score:.1f}/100 ({category})</span></div>"
            
            # Other key metrics
            metrics_to_show = [
                ("docstring_quality_score", "Docstring Quality"),
                ("complexity_score", "Function Complexity"),
                ("interpretability_score", "Model Interpretability"),
                ("explanation_quality", "Explanation Quality")
            ]
            
            for key, label in metrics_to_show:
                if key in summary:
                    summary_html += f"<div class='metric'><strong>{label}:</strong> {summary[key]:.1f}/100</div>"
        
        # Format recommendations section
        recommendations_html = "<ul>"
        if "summary" in results and "recommendations" in results["summary"]:
            for rec in results["summary"]["recommendations"]:
                recommendations_html += f"<li class='recommendation'>{rec}</li>"
        else:
            recommendations_html += "<li>No specific recommendations available.</li>"
        recommendations_html += "</ul>"
        
        # Format details section
        details_html = ""
        
        # Docstring information
        if "docstring_quality" in results:
            details_html += "<h2>Docstring Analysis</h2>"
            doc_quality = results["docstring_quality"]
            
            details_html += "<div class='explanation'>"
            if doc_quality["has_docstring"]:
                details_html += f"<p>Docstring quality score: {doc_quality['quality_score']:.1f}/100</p>"
                details_html += "<ul>"
                details_html += f"<li>Length: {doc_quality['length']} characters</li>"
                details_html += f"<li>Parameters documented: {'Yes' if doc_quality['has_parameters'] else 'No'}</li>"
                details_html += f"<li>Return values documented: {'Yes' if doc_quality['has_return_info'] else 'No'}</li>"
                details_html += f"<li>Examples included: {'Yes' if doc_quality['has_examples'] else 'No'}</li>"
                details_html += "</ul>"
            else:
                details_html += "<p>No docstring found.</p>"
            details_html += "</div>"
        
        # Function complexity
        if "function_complexity" in results:
            details_html += "<h2>Function Complexity Analysis</h2>"
            complexity = results["function_complexity"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Complexity score: {complexity['complexity_score']:.1f}</p>"
            details_html += f"<p>Interpretability: <span class='{complexity['interpretability'].lower()}'>{complexity['interpretability']}</span></p>"
            details_html += "<ul>"
            details_html += f"<li>Lines of code: {complexity['lines_of_code']}</li>"
            details_html += f"<li>Nested depth: {complexity['nested_depth']}</li>"
            details_html += f"<li>Branching factor: {complexity['branching_factor']:.2f}</li>"
            details_html += f"<li>Conditional statements: {complexity.get('conditional_counts', 'N/A')}</li>"
            details_html += f"<li>Loops: {complexity.get('loop_counts', 'N/A')}</li>"
            details_html += "</ul>"
            details_html += "</div>"
        
        # Model structure
        if "model_structure" in results:
            details_html += "<h2>Model Structure Analysis</h2>"
            structure = results["model_structure"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Model type: {structure['model_type']}</p>"
            details_html += f"<p>Interpretability score: {structure['interpretability_score']:.1f}/100</p>"
            details_html += f"<p>Interpretability category: <span class='{structure['interpretability_category'].lower()}'>{structure['interpretability_category']}</span></p>"
            
            # Add structure details
            if "structure_info" in structure and structure["structure_info"]:
                details_html += "<h3>Structure Details</h3>"
                details_html += "<table>"
                details_html += "<tr><th>Property</th><th>Value</th></tr>"
                
                for key, value in structure["structure_info"].items():
                    # Skip complex nested structures
                    if isinstance(value, (dict, list)):
                        continue
                    details_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
                details_html += "</table>"
            
            details_html += "</div>"
        
        # Feature importance
        if "feature_importance" in results:
            details_html += "<h2>Feature Importance Analysis</h2>"
            importance = results["feature_importance"]
            
            details_html += "<div class='explanation'>"
            if importance["methods_available"]:
                details_html += f"<p>Available methods: {', '.join(importance['methods_available'])}</p>"
                
                # Show feature importance for each method
                for method, values in importance["feature_importance"].items():
                    details_html += f"<h3>Method: {method}</h3>"
                    
                    if isinstance(values, dict):
                        # Create a table for feature importance values
                        details_html += "<table>"
                        details_html += "<tr><th>Feature</th><th>Importance</th></tr>"
                        
                        # Sort by importance value
                        sorted_features = sorted(values.items(), key=lambda x: x[1], reverse=True)
                        
                        for feature, value in sorted_features[:10]:  # Show top 10
                            details_html += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>"
                        
                        details_html += "</table>"
                    else:
                        details_html += "<p>Feature importance values available but feature names not provided.</p>"
            else:
                details_html += "<p>No feature importance methods available for this model.</p>"
            
            if "errors" in importance and importance["errors"]:
                details_html += "<h3>Errors</h3>"
                details_html += "<ul>"
                for error in importance["errors"]:
                    details_html += f"<li>{error}</li>"
                details_html += "</ul>"
            
            details_html += "</div>"
        
        # Decision rules
        if "decision_rules" in results:
            details_html += "<h2>Decision Rules</h2>"
            rules = results["decision_rules"]
            
            details_html += "<div class='explanation'>"
            if "rules" in rules and rules["rules"]:
                details_html += f"<p>Number of rules: {rules.get('rule_count', len(rules['rules']))}</p>"
                details_html += "<ol>"
                for rule in rules["rules"]:
                    details_html += f"<li><code>{rule}</code></li>"
                details_html += "</ol>"
            else:
                details_html += "<p>No decision rules available or applicable for this model.</p>"
                if "error" in rules:
                    details_html += f"<p>Error: {rules['error']}</p>"
            details_html += "</div>"
        
        # Human-readable explanation
        if "human_explanation" in results:
            details_html += "<h2>Human-Readable Explanation</h2>"
            explanation = results["human_explanation"]
            
            details_html += "<div class='explanation'>"
            if "error" not in explanation:
                if "natural_language" in explanation:
                    details_html += f"<p>{explanation['natural_language'].replace('\n', '<br>')}</p>"
                
                if "key_factors" in explanation:
                    details_html += "<h3>Key Factors</h3>"
                    details_html += "<table>"
                    details_html += "<tr><th>Feature</th><th>Contribution</th><th>Direction</th></tr>"
                    
                    for factor in explanation["key_factors"]:
                        details_html += (f"<tr><td>{factor['feature']}</td>"
                                        f"<td>{factor['contribution']:.4f}</td>"
                                        f"<td>{factor['direction']}</td></tr>")
                    
                    details_html += "</table>"
                
                if "counterfactual" in explanation:
                    details_html += "<h3>Counterfactual Example</h3>"
                    cf = explanation["counterfactual"]
                    details_html += f"<p>If <strong>{cf['feature_changed']}</strong> were "
                    details_html += f"changed from {cf['original_value']} to {cf['new_value']}, "
                    details_html += "the prediction might change.</p>"
            else:
                details_html += f"<p>Error generating explanation: {explanation['error']}</p>"
            details_html += "</div>"
        
        # Visualizations
        if "visualizations" in results:
            details_html += "<h2>Visualizations</h2>"
            visualizations = results["visualizations"]
            
            if "visualizations" in visualizations and visualizations["visualizations"]:
                for viz in visualizations["visualizations"]:
                    details_html += "<div class='explanation'>"
                    details_html += f"<h3>{viz['type'].replace('_', ' ').title()}</h3>"
                    
                    if "path" in viz:
                        # Check if the path exists and is accessible
                        if os.path.exists(viz["path"]):
                            # For HTML report, use relative paths
                            rel_path = os.path.relpath(viz["path"], os.path.dirname(output_file))
                            details_html += f"<img src='{rel_path}' alt='{viz['type']}' />"
                        else:
                            details_html += f"<p>Visualization file not found: {viz['path']}</p>"
                    
                    if "feature" in viz:
                        details_html += f"<p>Feature: {viz['feature']}</p>"
                    
                    if "note" in viz:
                        details_html += f"<p>Note: {viz['note']}</p>"
                    
                    details_html += "</div>"
            else:
                details_html += "<p>No visualizations available.</p>"
        
        # Fill in the template
        html_content = html_template.format(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_html=summary_html,
            recommendations_html=recommendations_html,
            details_html=details_html
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return False

def evaluate_model_explainability(model, X=None, y=None, feature_names=None, output_dir="explainability_results"):
    """
    Comprehensive assessment of model explainability.
    
    :param model: The model to evaluate.
    :param X: Input data (optional, enables advanced assessments).
    :param y: Target data (optional, enables advanced assessments).
    :param feature_names: List of feature names (optional).
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "model_type": type(model).__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Analyze model structure
    logger.info("Analyzing model structure...")
    results["model_structure"] = analyze_model_structure(model)
    
    # Extract feature importance
    logger.info("Extracting feature importance...")
    results["feature_importance"] = extract_feature_importance(model, feature_names, X, y)
    
    # Generate decision rules if possible
    logger.info("Generating decision rules...")
    results["decision_rules"] = generate_decision_rules(model, feature_names)
    
    # If we have input data, perform more advanced evaluations
    if X is not None:
        # Generate visualizations
        logger.info("Generating visualizations...")
        results["visualizations"] = generate_visualizations(
            model, X, y, feature_names, 
            output_dir=os.path.join(output_dir, "visualizations")
        )
        
        # Generate example-specific explanations
        if len(X) > 0:
            logger.info("Generating instance explanation...")
            # Use the first instance as an example
            results["human_explanation"] = create_human_readable_explanation(
                model, X, y, feature_names, instance_idx=0
            )
            
            # Generate explanation quality assessment
            if LIME_AVAILABLE:
                lime_explanation = generate_lime_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(lime_explanation)
            elif SHAP_AVAILABLE:
                shap_explanation = generate_shap_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(shap_explanation)
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, model=model, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def evaluate_function_explainability(function, output_dir="explainability_results"):
    """
    Comprehensive assessment of function explainability.
    
    :param function: The function to evaluate.
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "function_name": function.__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Check docstring quality
    logger.info("Checking docstring quality...")
    results["docstring_quality"] = check_docstring_quality(function)
    
    # Analyze function complexity
    logger.info("Analyzing function complexity...")
    results["function_complexity"] = analyze_function_complexity(function)
    
    # Extract function source code
    try:
        results["function_source"] = inspect.getsource(function)
    except:
        results["function_source"] = "Source code not available"
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, function=function, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Explainability & Interpretability Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--function",
        help="Function name to analyze"
    )
    
    parser.add_argument(
        "--module",
        help="Module containing the function or model"
    )
    
    parser.add_argument(
        "--model",
        help="Name of model object to analyze (must be in the specified module)"
    )
    
    parser.add_argument(
        "--data",
        help="Path to data file for model analysis (CSV, pickle, etc.)"
    )
    
    parser.add_argument(
        "--target-column",
        help="Name of target column in the data file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="explainability_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# Example function to be analyzed
def example_function(x):
    """
    Simple function that squares a number.
    
    This function takes an input number and returns its square.
    
    :param x: The number to be squared.
    :return: The square of the input number.
    
    Example:
        >>> example_function(4)
        16
    """
    return x ** 2

# Run explainability evaluation
if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Handle function evaluation
    if args.function:
        try:
            if args.module:
                # Import the module containing the function
                module = importlib.import_module(args.module)
                function = getattr(module, args.function)
            else:
                # Use the example function if no module specified
                function = example_function
            
            print(f"Evaluating explainability of function '{function.__name__}'...")
            evaluate_function_explainability(function, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating function: {e}")
            sys.exit(1)
    
    # Handle model evaluation
    elif args.model:
        try:
            if not args.module:
                logger.error("Module must be specified when analyzing a model")
                sys.exit(1)
            
            # Import the module containing the model
            module = importlib.import_module(args.module)
            model = getattr(module, args.model)
            
            # Load data if provided
            X = None
            y = None
            feature_names = None
            
            if args.data:
                print(f"Loading data from {args.data}...")
                
                if args.data.endswith('.csv'):
                    if not PANDAS_AVAILABLE:
                        logger.error("Pandas is required to load CSV files")
                        sys.exit(1)
                    
                    data = pd.read_csv(args.data)
                    
                    if args.target_column and args.target_column in data.columns:
                        y = data[args.target_column]
                        X = data.drop(columns=[args.target_column])
                    else:
                        # Assume all columns are features
                        X = data
                    
                    feature_names = X.columns.tolist()
                
                elif args.data.endswith('.pkl') or args.data.endswith('.pickle'):
                    with open(args.data, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        X = data.get('X') or data.get('data')
                        y = data.get('y') or data.get('target')
                        feature_names = data.get('feature_names')
                    else:
                        # Assume it's just the features
                        X = data
            
            print(f"Evaluating explainability of model '{args.model}'...")
            evaluate_model_explainability(model, X, y, feature_names, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            sys.exit(1)
    
    else:
        # No specific function or model specified, run on example function
        print("No function or model specified, using example function...")
        evaluate_function_explainability(example_function, args.output_dir)
    
    print(f"Explainability assessment complete. Results saved to {args.output_dir}")

"""
TODO:
- Add support for more advanced model types (e.g., ensemble methods, deep learning frameworks)
- Implement benchmarking against "gold standard" human interpretations
- Add support for explainability of embeddings and representations
- Include more advanced LIME and SHAP visualization options
- Implement counterfactual explanation generation with optimization
- Add evaluation of explanation faithfulness and stability
- Support for multi-modal explanations (text + visualization)
- Add natural language explanation generation with templates
- Integrate with popular model registries and governance frameworks
- Extend report generation with more detailed legal and compliance reporting
"""#!/usr/bin/env python3
"""
EXPLAINABILITY & INTERPRETABILITY EVALUATION FRAMEWORK
======================================================

This framework evaluates and enhances the explainability and interpretability of machine learning
models and algorithmic functions. It provides:

1. Comprehensive explainability assessment for different model types
2. Integration with state-of-the-art XAI techniques (SHAP, LIME, etc.)
3. Visualization of feature importance and decision boundaries
4. Quantitative metrics for measuring interpretability
5. Structured reporting for regulatory and legal compliance
6. Human-friendliness evaluation of model explanations
7. Output format optimization for stakeholder understanding

The framework supports various model types including:
- Tree-based models (Random Forests, XGBoost, etc.)
- Linear models (Linear/Logistic Regression, etc.)
- Neural Networks (with activation visualization)
- Black-box models (through post-hoc explanation techniques)
"""

import logging
import json
import inspect
import os
import re
import sys
import argparse
import datetime
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import wraps
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging with both file and console handlers
def setup_logging(log_file='explainability_interpretability.log', console_level=logging.INFO):
    """Setup logging to file and console with different levels."""
    logger = logging.getLogger('explainability')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Utility function to check if a model is a specific type
def is_tree_based_model(model):
    """Check if the model is tree-based."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['tree', 'forest', 'boost', 'xgb', 'lgbm', 'catboost'])

def is_linear_model(model):
    """Check if the model is a linear model."""
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['linear', 'logistic', 'regression', 'lasso', 'ridge', 'elasticnet'])

def is_neural_network(model):
    """Check if the model is a neural network."""
    # Check for TensorFlow/Keras models
    if TF_AVAILABLE and isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
        return True
    
    # Check for PyTorch models
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        return True
    
    # Check by class name
    model_module = model.__module__
    model_name = model.__class__.__name__
    return any(name in model_module.lower() or name in model_name.lower() 
              for name in ['neural', 'deep', 'mlp', 'keras', 'tensorflow', 'torch', 'nn.'])

# Core explainability assessment functions
def check_docstring_quality(function):
    """
    Evaluate the quality of a function's docstring.
    
    :param function: The function to be checked.
    :return: Dictionary with docstring quality metrics.
    """
    if not function.__doc__:
        return {
            "has_docstring": False,
            "quality_score": 0,
            "length": 0,
            "has_parameters": False,
            "has_return_info": False,
            "has_examples": False,
            "suggestions": ["Add a docstring to explain the function's purpose and usage."]
        }
    
    docstring = function.__doc__.strip()
    
    # Basic metrics
    has_parameters = ":param" in docstring or "Parameters:" in docstring
    has_return_info = ":return" in docstring or "Returns:" in docstring
    has_examples = "Example" in docstring
    length = len(docstring)
    
    # Calculate quality score (0-100)
    quality_score = 0
    if length > 0:
        quality_score += min(length / 5, 30)  # Up to 30 points for length
    if has_parameters:
        quality_score += 25  # 25 points for parameter documentation
    if has_return_info:
        quality_score += 25  # 25 points for return value documentation
    if has_examples:
        quality_score += 20  # 20 points for examples
    
    # Generate suggestions
    suggestions = []
    if not has_parameters:
        suggestions.append("Add parameter descriptions.")
    if not has_return_info:
        suggestions.append("Document return values.")
    if not has_examples:
        suggestions.append("Include usage examples.")
    if length < 50:
        suggestions.append("Expand the description to be more detailed.")
        
    return {
        "has_docstring": True,
        "quality_score": quality_score,
        "length": length,
        "has_parameters": has_parameters,
        "has_return_info": has_return_info,
        "has_examples": has_examples,
        "suggestions": suggestions
    }

def analyze_function_complexity(function):
    """
    Analyze function complexity and structure for explainability.
    
    :param function: The function to be analyzed.
    :return: Dictionary with complexity metrics.
    """
    try:
        source_code = inspect.getsource(function)
    except (TypeError, OSError) as e:
        logger.warning(f"Could not retrieve source code: {e}")
        return {
            "complexity_score": None,
            "lines_of_code": None,
            "nested_depth": None,
            "branching_factor": None,
            "error": str(e)
        }
    
    # Count lines of code
    lines = source_code.strip().split('\n')
    lines_of_code = len(lines)
    
    # Count conditional statements (if, elif, else)
    conditional_counts = len(re.findall(r'\bif\b|\belif\b|\belse\b', source_code))
    
    # Count loops (for, while)
    loop_counts = len(re.findall(r'\bfor\b|\bwhile\b', source_code))
    
    # Estimate nesting depth
    indent_levels = [len(line) - len(line.lstrip()) for line in lines]
    max_indent = max(indent_levels) if indent_levels else 0
    nested_depth = max_indent // 4  # Assuming 4 spaces per indentation level
    
    # Calculate complexity score (higher means more complex, less interpretable)
    complexity_score = (
        conditional_counts * 2 + 
        loop_counts * 3 + 
        nested_depth * 5 + 
        lines_of_code / 10
    )
    
    # Calculate branching factor
    branching_factor = conditional_counts / lines_of_code if lines_of_code > 0 else 0
    
    # Interpretability assessment
    if complexity_score < 10:
        interpretability = "High"
    elif complexity_score < 25:
        interpretability = "Medium"
    else:
        interpretability = "Low"
        
    return {
        "complexity_score": complexity_score,
        "interpretability": interpretability,
        "lines_of_code": lines_of_code,
        "nested_depth": nested_depth,
        "branching_factor": branching_factor,
        "conditional_counts": conditional_counts,
        "loop_counts": loop_counts
    }

def extract_feature_importance(model, feature_names=None, X=None, y=None):
    """
    Extract feature importance from a model using appropriate method.
    
    :param model: The trained model to analyze.
    :param feature_names: List of feature names.
    :param X: Input data for permutation importance (optional).
    :param y: Target data for permutation importance (optional).
    :return: Dictionary with feature importance information.
    """
    importance_methods = []
    importance_values = {}
    error_messages = []
    
    # Try built-in feature importance
    if hasattr(model, "feature_importances_"):
        importance_methods.append("built_in")
        if feature_names and len(model.feature_importances_) == len(feature_names):
            importance_values["built_in"] = dict(zip(feature_names, model.feature_importances_.tolist()))
        else:
            importance_values["built_in"] = model.feature_importances_.tolist()
    
    # Try coefficients for linear models
    if hasattr(model, "coef_"):
        importance_methods.append("coefficients")
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = np.mean(np.abs(coefs), axis=0)  # For multi-class models
            
        if feature_names and len(coefs) == len(feature_names):
            importance_values["coefficients"] = dict(zip(feature_names, coefs.tolist()))
        else:
            importance_values["coefficients"] = coefs.tolist()
    
    # Try permutation importance if sklearn is available and data is provided
    if SKLEARN_AVAILABLE and X is not None and y is not None:
        try:
            importance_methods.append("permutation")
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            
            if feature_names and len(perm_importance.importances_mean) == len(feature_names):
                importance_values["permutation"] = dict(zip(feature_names, perm_importance.importances_mean.tolist()))
            else:
                importance_values["permutation"] = perm_importance.importances_mean.tolist()
        except Exception as e:
            error_messages.append(f"Permutation importance failed: {str(e)}")
    
    # Try SHAP if available and appropriate
    if SHAP_AVAILABLE and X is not None:
        try:
            sample_X = X[:100] if hasattr(X, "__len__") and len(X) > 100 else X
            
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
                importance_methods.append("shap_tree")
            else:
                explainer = shap.Explainer(model)
                importance_methods.append("shap")
                
            shap_values = explainer.shap_values(sample_X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # For multi-class models, take mean absolute value across classes
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            
            # Calculate feature importance as mean absolute SHAP value
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            if feature_names and len(mean_abs_shap) == len(feature_names):
                importance_values["shap"] = dict(zip(feature_names, mean_abs_shap.tolist()))
            else:
                importance_values["shap"] = mean_abs_shap.tolist()
                
        except Exception as e:
            error_messages.append(f"SHAP analysis failed: {str(e)}")
    
    return {
        "methods_available": importance_methods,
        "feature_importance": importance_values,
        "errors": error_messages
    }

def analyze_model_structure(model):
    """
    Analyze the structure of a model to assess interpretability.
    
    :param model: The model to analyze.
    :return: Dictionary with model structure information.
    """
    model_type = "unknown"
    interpretability_score = 0
    structure_info = {}
    
    # Determine model type
    if is_tree_based_model(model):
        model_type = "tree_based"
        
        # Extract tree structure information
        if hasattr(model, "tree_") or hasattr(model, "estimators_"):
            n_nodes = 0
            max_depth = 0
            
            if hasattr(model, "tree_"):
                # Single tree model
                n_nodes = model.tree_.node_count
                max_depth = model.get_depth() if hasattr(model, "get_depth") else model.tree_.max_depth
                structure_info["n_nodes"] = n_nodes
                structure_info["max_depth"] = max_depth
            
            elif hasattr(model, "estimators_"):
                # Ensemble of trees
                n_estimators = len(model.estimators_)
                structure_info["n_estimators"] = n_estimators
                
                depths = []
                nodes = []
                
                for estimator in model.estimators_:
                    if hasattr(estimator, "tree_"):
                        nodes.append(estimator.tree_.node_count)
                        if hasattr(estimator, "get_depth"):
                            depths.append(estimator.get_depth())
                        else:
                            depths.append(estimator.tree_.max_depth)
                
                if depths:
                    max_depth = max(depths)
                    structure_info["max_depth"] = max_depth
                    structure_info["mean_depth"] = sum(depths) / len(depths)
                
                if nodes:
                    n_nodes = sum(nodes)
                    structure_info["total_nodes"] = n_nodes
                    structure_info["mean_nodes_per_tree"] = n_nodes / n_estimators
            
            # Calculate interpretability score for tree models (higher is more interpretable)
            if max_depth > 0:
                depth_penalty = max(0, (max_depth - 3) * 10)  # Deeper trees are less interpretable
                interpretability_score = max(0, 100 - depth_penalty)
                
                if "n_estimators" in structure_info:
                    # Ensemble models are less interpretable
                    ensemble_penalty = min(80, structure_info["n_estimators"] * 2)
                    interpretability_score = max(0, interpretability_score - ensemble_penalty)
        
    elif is_linear_model(model):
        model_type = "linear"
        
        # Linear models are generally interpretable
        interpretability_score = 80
        
        # Check for sparsity in linear models
        if hasattr(model, "coef_"):
            coefs = model.coef_
            n_features = coefs.size if len(coefs.shape) == 1 else coefs.shape[1]
            n_nonzero = np.count_nonzero(coefs)
            sparsity = 1.0 - (n_nonzero / n_features) if n_features > 0 else 0
            
            structure_info["n_features"] = n_features
            structure_info["n_nonzero_coefficients"] = n_nonzero
            structure_info["sparsity"] = sparsity
            
            # Sparse models are more interpretable
            if sparsity > 0.5:
                interpretability_score += 15
            elif sparsity < 0.2:
                interpretability_score -= 10
    
    elif is_neural_network(model):
        model_type = "neural_network"
        
        # Neural networks are generally less interpretable
        interpretability_score = 30
        
        # Analyze network architecture
        if TF_AVAILABLE and isinstance(model, tf.keras.Model):
            # For TensorFlow/Keras models
            structure_info["layers"] = []
            total_params = 0
            
            for i, layer in enumerate(model.layers):
                layer_info = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape)
                }
                
                if hasattr(layer, "count_params"):
                    params = layer.count_params()
                    layer_info["parameters"] = params
                    total_params += params
                
                structure_info["layers"].append(layer_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(model.layers) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
        
        elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            # For PyTorch models
            structure_info["modules"] = []
            total_params = 0
            
            for name, module in model.named_modules():
                if list(module.children()):  # Skip container modules
                    continue
                    
                module_info = {
                    "name": name,
                    "type": module.__class__.__name__
                }
                
                params = sum(p.numel() for p in module.parameters())
                module_info["parameters"] = params
                total_params += params
                
                structure_info["modules"].append(module_info)
            
            structure_info["total_parameters"] = total_params
            
            # Very simple networks can be somewhat interpretable
            if len(structure_info["modules"]) <= 3 and total_params < 1000:
                interpretability_score += 20
            elif total_params > 1000000:  # Large networks are less interpretable
                interpretability_score -= 20
    
    else:
        # Unknown model type
        model_type = "unknown"
        interpretability_score = 40  # Default score for unknown models
        
        # Try to extract some generic information
        model_attributes = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
        for attr in model_attributes:
            try:
                val = getattr(model, attr)
                if isinstance(val, (int, float, str, bool)):
                    structure_info[attr] = val
            except:
                pass
    
    # Determine interpretability category
    if interpretability_score >= 70:
        interpretability_category = "High"
    elif interpretability_score >= 40:
        interpretability_category = "Medium"
    else:
        interpretability_category = "Low"
        
    return {
        "model_type": model_type,
        "interpretability_score": interpretability_score,
        "interpretability_category": interpretability_category,
        "structure_info": structure_info
    }

def generate_lime_explanation(model, X, feature_names=None, class_names=None, instance_idx=0):
    """
    Generate a LIME explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param class_names: List of class names for classification.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with LIME explanation results.
    """
    if not LIME_AVAILABLE:
        return {"error": "LIME package is not available. Install with 'pip install lime'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx].values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx]
        
        # Determine if we're doing classification or regression
        # This is a simplification - in practice, you might need to check model type
        mode = "classification"
        if class_names is None:
            # Try to guess if it's binary classification
            mode = "classification"
            class_names = [0, 1]
        
        # Create the LIME explainer
        explainer = LimeTabularExplainer(
            X if isinstance(X, np.ndarray) else X.values,
            feature_names=feature_names,
            class_names=class_names,
            mode=mode
        )
        
        # Define prediction function based on expected input
        if mode == "classification":
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:
                # Fallback to non-probabilistic prediction
                def predict_fn(x):
                    preds = model.predict(x)
                    # Convert to pseudo-probabilities for binary classification
                    if len(preds.shape) == 1 and len(class_names) == 2:
                        return np.vstack([(1-preds), preds]).T
                    return preds
        else:
            predict_fn = model.predict
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance, 
            predict_fn,
            num_features=min(10, len(feature_names) if feature_names else 10)
        )
        
        # Extract explanation data
        if mode == "classification":
            # For classification, get explanation for top predicted class
            try:
                prediction = model.predict([instance])[0]
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.argmax()
                exp_data = explanation.as_list(label=prediction)
            except:
                # Fallback to first class
                exp_data = explanation.as_list(label=0)
        else:
            # For regression
            exp_data = explanation.as_list()
        
        # Format the explanation data
        lime_features = []
        for feature, weight in exp_data:
            lime_features.append({
                "feature": feature,
                "weight": weight
            })
        
        return {
            "explanation_type": "lime",
            "mode": mode,
            "features": lime_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        return {"error": f"Failed to generate LIME explanation: {str(e)}"}

def generate_shap_explanation(model, X, feature_names=None, instance_idx=0):
    """
    Generate a SHAP explanation for a specific instance.
    
    :param model: The model to explain.
    :param X: The input data.
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with SHAP explanation results.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP package is not available. Install with 'pip install shap'."}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
        
        # Choose the right explainer based on model type
        if is_tree_based_model(model):
            explainer = shap.TreeExplainer(model)
        elif is_neural_network(model) and hasattr(model, 'predict'):
            # For neural networks, use KernelExplainer with a small background
            sample_X = X[:50] if hasattr(X, "__len__") and len(X) > 50 else X
            explainer = shap.KernelExplainer(model.predict, sample_X)
        else:
            explainer = shap.Explainer(model)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class, use values for predicted class
            prediction = model.predict(instance)[0]
            if isinstance(prediction, np.ndarray):
                prediction = prediction.argmax()
            values = shap_values[prediction]
        else:
            values = shap_values
        
        # Format the explanation
        if len(values.shape) > 1:
            values = values[0]  # Get values for first (only) instance
        
        # Pair with feature names
        if feature_names and len(feature_names) == len(values):
            shap_features = [{"feature": name, "value": float(val)} 
                            for name, val in zip(feature_names, values)]
        else:
            shap_features = [{"feature": f"Feature {i}", "value": float(val)} 
                            for i, val in enumerate(values)]
        
        # Sort by absolute value for importance
        shap_features.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "explanation_type": "shap",
            "features": shap_features,
            "instance_idx": instance_idx
        }
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        return {"error": f"Failed to generate SHAP explanation: {str(e)}"}

def evaluate_explanation_quality(explanation_data):
    """
    Evaluate the quality of model explanations.
    
    :param explanation_data: Dictionary with explanation results.
    :return: Dictionary with explanation quality metrics.
    """
    metrics = {
        "completeness": 0,
        "consistency": 0,
        "compactness": 0,
        "clarity": 0,
        "overall_quality": 0
    }
    
    explanation_type = explanation_data.get("explanation_type", "unknown")
    
    if "error" in explanation_data:
        # Failed explanation
        metrics["error"] = explanation_data["error"]
        metrics["overall_quality"] = 0
        return metrics
    
    # Evaluate LIME explanations
    if explanation_type == "lime":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are weights consistent?
        if features:
            weights = [abs(f["weight"]) for f in features]
            weight_std = np.std(weights)
            weight_mean = np.mean(weights)
            if weight_mean > 0:
                cv = weight_std / weight_mean
                metrics["consistency"] = max(0, 100 - cv * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Evaluate SHAP explanations
    elif explanation_type == "shap":
        features = explanation_data.get("features", [])
        
        # Completeness: are important features present?
        metrics["completeness"] = min(1.0, len(features) / 10) * 100
        
        # Compactness: is the explanation concise?
        metrics["compactness"] = max(0, 100 - (len(features) - 5) * 10) if len(features) > 5 else 100
        
        # Consistency: are values distributed reasonably?
        if features:
            values = [abs(f["value"]) for f in features]
            # Check if there's a good distribution of importance
            sorted_values = sorted(values, reverse=True)
            if sorted_values[0] > 0:
                # Measure how quickly importance drops off
                importance_ratio = sum(sorted_values[1:]) / sorted_values[0]
                metrics["consistency"] = min(100, importance_ratio * 100)
            
        # Clarity: are features well-labeled?
        has_clear_features = all(not f["feature"].startswith("Feature ") for f in features) if features else False
        metrics["clarity"] = 80 if has_clear_features else 40
    
    # Calculate overall quality
    weights = {
        "completeness": 0.3,
        "consistency": 0.3,
        "compactness": 0.2,
        "clarity": 0.2
    }
    
    metrics["overall_quality"] = sum(metrics[k] * weights[k] for k in weights)
    
    return metrics

def generate_decision_rules(model, feature_names=None):
    """
    Extract interpretable decision rules from a model.
    
    :param model: The model to extract rules from.
    :param feature_names: List of feature names.
    :return: Dictionary with decision rules.
    """
    rules = []
    
    # Only tree-based models and some linear models can generate rules
    if not is_tree_based_model(model) and not is_linear_model(model):
        return {
            "rules": [],
            "error": "Rule extraction is only supported for tree-based and linear models"
        }
    
    try:
        if is_tree_based_model(model):
            # Try to extract rules from tree-based model
            if hasattr(model, "estimators_") and SKLEARN_AVAILABLE:
                # For random forests, extract from first few trees
                from sklearn.tree import _tree
                
                max_trees = min(3, len(model.estimators_))
                for i in range(max_trees):
                    tree = model.estimators_[i]
                    tree_rules = _extract_rules_from_tree(tree, feature_names)
                    rules.extend([f"Tree {i+1}: {rule}" for rule in tree_rules[:5]])  # Limit rules per tree
            
            elif hasattr(model, "tree_") and SKLEARN_AVAILABLE:
                # For single decision tree
                from sklearn.tree import _tree
                rules = _extract_rules_from_tree(model, feature_names)
        
        elif is_linear_model(model):
            # For linear models, create rules based on coefficients
            if hasattr(model, "coef_") and feature_names:
                coefs = model.coef_
                intercept = model.intercept_ if hasattr(model, "intercept_") else 0
                
                # For binary classification or regression
                if len(coefs.shape) == 1:
                    # Sort features by importance (coefficient magnitude)
                    sorted_idx = np.argsort(np.abs(coefs))[::-1]
                    
                    # Create rules for top features
                    for idx in sorted_idx[:5]:  # Limit to 5 most important features
                        feature = feature_names[idx]
                        coef = coefs[idx]
                        if coef > 0:
                            rules.append(f"Higher values of '{feature}' increase the prediction")
                        else:
                            rules.append(f"Higher values of '{feature}' decrease the prediction")
                    
                    # Add intercept information
                    if intercept != 0:
                        rules.append(f"Base value (intercept) is {intercept:.4f}")
                
                # For multi-class classification
                elif len(coefs.shape) == 2:
                    n_classes = coefs.shape[0]
                    for class_idx in range(min(n_classes, 3)):  # Limit to first 3 classes
                        class_rules = []
                        
                        # Sort features by importance for this class
                        sorted_idx = np.argsort(np.abs(coefs[class_idx]))[::-1]
                        
                        # Create rules for top features
                        for idx in sorted_idx[:3]:  # Limit to 3 most important features per class
                            feature = feature_names[idx]
                            coef = coefs[class_idx, idx]
                            if coef > 0:
                                class_rules.append(f"Higher '{feature}' increases probability")
                            else:
                                class_rules.append(f"Higher '{feature}' decreases probability")
                        
                        # Add class-specific intercept
                        if isinstance(intercept, np.ndarray) and len(intercept) > class_idx:
                            class_rules.append(f"Base value: {intercept[class_idx]:.4f}")
                        
                        rules.append(f"Class {class_idx}: " + ", ".join(class_rules))
    
    except Exception as e:
        logger.warning(f"Error extracting decision rules: {str(e)}")
        return {"rules": [], "error": f"Rule extraction failed: {str(e)}"}
    
    return {
        "rules": rules[:10],  # Limit total rules
        "rule_count": len(rules)
    }

def _extract_rules_from_tree(tree_model, feature_names):
    """Helper function to extract rules from a decision tree."""
    if not SKLEARN_AVAILABLE:
        return []
        
    from sklearn.tree import _tree
    
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if feature_names and i != _tree.TREE_UNDEFINED else f"feature {i}"
        for i in tree_.feature
    ]
    
    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left path: feature <= threshold
            left_path = path.copy()
            left_path.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], depth + 1, left_path)
            
            # Right path: feature > threshold
            right_path = path.copy()
            right_path.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], depth + 1, right_path)
        else:
            # Leaf node
            if tree_.n_outputs == 1:
                value = tree_.value[node][0][0]
                rule = " AND ".join(path) + f" → {value:.2f}"
                rules.append(rule)
            else:
                # Multi-output: get class with highest probability
                class_idx = np.argmax(tree_.value[node])
                value = tree_.value[node][0][class_idx]
                rule = " AND ".join(path) + f" → Class {class_idx} (prob: {value:.2f})"
                rules.append(rule)
    
    rules = []
    recurse(0, 1, [])
    
    # Sort rules by complexity (number of conditions)
    rules.sort(key=lambda x: x.count("AND"))
    
    return rules

def generate_visualizations(model, X, y=None, feature_names=None, output_dir="explainability_visualizations"):
    """
    Generate visualizations to aid in model understanding.
    
    :param model: The model to visualize.
    :param X: The input data.
    :param y: The target data (optional).
    :param feature_names: List of feature names.
    :param output_dir: Directory to save visualizations.
    :return: Dictionary with visualization information.
    """
    os.makedirs(output_dir, exist_ok=True)
    visualizations = []
    
    # Convert X to numpy if it's pandas
    if hasattr(X, 'values'):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
    
    # Generate feature importance visualization if available
    importance_data = extract_feature_importance(model, feature_names, X, y)
    
    if importance_data["methods_available"]:
        # Use the first available method
        method = importance_data["methods_available"][0]
        importances = importance_data["feature_importance"].get(method, [])
        
        if importances and isinstance(importances, dict):
            # Create feature importance bar chart
            features = list(importances.keys())
            values = list(importances.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            sorted_features = [features[i] for i in sorted_idx[-15:]]  # Top 15 features
            sorted_values = [values[i] for i in sorted_idx[-15:]]
            
            plt.figure(figsize=(10, 8))
            plt.barh(sorted_features, sorted_values)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance ({method})')
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(output_dir, f"feature_importance_{method}.png")
            plt.savefig(viz_path)
            plt.close()
            
            visualizations.append({
                "type": "feature_importance",
                "method": method,
                "path": viz_path
            })
    
    # Generate SHAP visualizations if available
    if SHAP_AVAILABLE and len(X_values) > 0:
        try:
            # Use a sample of the data for SHAP
            sample_size = min(100, len(X_values))
            sample_indices = np.random.choice(len(X_values), sample_size, replace=False)
            X_sample = X_values[sample_indices]
            
            # Create the explainer
            if is_tree_based_model(model):
                explainer = shap.TreeExplainer(model)
            else:
                # For other models, use Kernel explainer with background data
                background = shap.kmeans(X_values, 5)  # Use k-means for background data
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For multi-class, use class 0
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot, 
                X_sample, 
                feature_names=feature_names,
                show=False
            )
            
            viz_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(viz_path, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                "type": "shap_summary",
                "path": viz_path
            })
            
            # Dependence plot for most important feature
            if feature_names and len(feature_names) > 0:
                # Find most important feature by mean absolute SHAP value
                mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
                top_feature_idx = mean_abs_shap.argmax()
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    top_feature_idx, 
                    shap_values_plot, 
                    X_sample, 
                    feature_names=feature_names,
                    show=False
                )
                
                viz_path = os.path.join(output_dir, "shap_dependence.png")
                plt.savefig(viz_path, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    "type": "shap_dependence",
                    "feature": feature_names[top_feature_idx],
                    "path": viz_path
                })
        
        except Exception as e:
            logger.warning(f"Error generating SHAP visualizations: {str(e)}")
    
    # Generate decision tree visualization if applicable
    if is_tree_based_model(model) and SKLEARN_AVAILABLE:
        try:
            from sklearn.tree import export_graphviz
            import subprocess
            
            # If it's a random forest, visualize the first tree
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                tree_to_viz = model.estimators_[0]
            else:
                tree_to_viz = model
            
            # Export tree as dot file
            dot_path = os.path.join(output_dir, "decision_tree.dot")
            export_graphviz(
                tree_to_viz,
                out_file=dot_path,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                max_depth=3  # Limit depth for readability
            )
            
            # Try to convert to PNG if graphviz is installed
            png_path = os.path.join(output_dir, "decision_tree.png")
            try:
                subprocess.run(
                    ["dot", "-Tpng", dot_path, "-o", png_path],
                    check=True,
                    stderr=subprocess.PIPE
                )
                
                visualizations.append({
                    "type": "decision_tree",
                    "path": png_path
                })
            except (subprocess.SubprocessError, FileNotFoundError):
                # Graphviz not installed or error occurred
                visualizations.append({
                    "type": "decision_tree_dot",
                    "path": dot_path,
                    "note": "Install Graphviz to convert to PNG"
                })
        
        except Exception as e:
            logger.warning(f"Error generating decision tree visualization: {str(e)}")
    
    # Generate partial dependence plots if sklearn is available
    if SKLEARN_AVAILABLE and feature_names and len(feature_names) > 0:
        try:
            from sklearn.inspection import plot_partial_dependence
            
            # Use feature importance to find the top features
            importance_data = extract_feature_importance(model, feature_names, X, y)
            
            if importance_data["methods_available"]:
                method = importance_data["methods_available"][0]
                importances = importance_data["feature_importance"].get(method, [])
                
                if importances:
                    # Get top 2 features
                    if isinstance(importances, dict):
                        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        top_features = [feature_names.index(f[0]) for f in sorted_features[:2] 
                                       if f[0] in feature_names]
                    else:
                        top_features = np.argsort(importances)[-2:]
                    
                    # Generate partial dependence plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_partial_dependence(
                        model, X_values, top_features,
                        feature_names=feature_names,
                        ax=ax
                    )
                    
                    viz_path = os.path.join(output_dir, "partial_dependence.png")
                    plt.savefig(viz_path)
                    plt.close()
                    
                    visualizations.append({
                        "type": "partial_dependence",
                        "features": [feature_names[i] for i in top_features],
                        "path": viz_path
                    })
        
        except Exception as e:
            logger.warning(f"Error generating partial dependence plots: {str(e)}")
    
    return {
        "visualizations": visualizations,
        "count": len(visualizations),
        "output_dir": output_dir
    }

def create_human_readable_explanation(model, X, y=None, feature_names=None, instance_idx=0):
    """
    Generate a human-readable explanation of model prediction.
    
    :param model: The model to explain.
    :param X: Input data.
    :param y: Target data (optional).
    :param feature_names: List of feature names.
    :param instance_idx: Index of the instance to explain.
    :return: Dictionary with human-readable explanation.
    """
    explanation = {}
    
    try:
        # Get the instance to explain
        if hasattr(X, "iloc"):
            # Pandas DataFrame
            instance = X.iloc[instance_idx:instance_idx+1]
            if feature_names is None:
                feature_names = X.columns.tolist()
            
            # Get feature values as a dict
            feature_values = instance.iloc[0].to_dict()
        else:
            # Numpy array
            instance = X[instance_idx:instance_idx+1]
            
            # Create feature value dict
            if feature_names:
                feature_values = {name: instance[0, i] for i, name in enumerate(feature_names)}
            else:
                feature_values = {f"Feature {i}": val for i, val in enumerate(instance[0])}
        
        # Get the model's prediction
        prediction = model.predict(instance)[0]
        
        # Try to get probability if it's a classifier
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(instance)[0]
                probability = np.max(probs)
            except:
                pass
        
        # Add basic prediction info
        explanation["prediction"] = {
            "value": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
            "probability": float(probability) if probability is not None else None
        }
        
        # Add instance values
        explanation["instance"] = {
            "index": instance_idx,
            "features": feature_values
        }
        
        # Try to generate LIME explanation
        if LIME_AVAILABLE:
            lime_explanation = generate_lime_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in lime_explanation:
                top_features = lime_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add LIME insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["weight"],
                        "direction": "increases" if feature["weight"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If LIME failed or is not available, try SHAP
        if "key_factors" not in explanation and SHAP_AVAILABLE:
            shap_explanation = generate_shap_explanation(
                model, X, feature_names, instance_idx=instance_idx
            )
            
            if "error" not in shap_explanation:
                top_features = shap_explanation.get("features", [])[:5]  # Top 5 features
                
                # Add SHAP insights
                explanation["key_factors"] = [
                    {
                        "feature": feature["feature"],
                        "contribution": feature["value"],
                        "direction": "increases" if feature["value"] > 0 else "decreases"
                    }
                    for feature in top_features
                ]
        
        # If we have key factors, create a natural language explanation
        if "key_factors" in explanation:
            factors = explanation["key_factors"]
            
            # Create natural language summary
            if is_tree_based_model(model):
                model_type = "decision tree" if not hasattr(model, "estimators_") else "random forest"
            elif is_linear_model(model):
                model_type = "linear model"
            elif is_neural_network(model):
                model_type = "neural network"
            else:
                model_type = "model"
                
            # Start with prediction statement
            if "probability" in explanation["prediction"] and explanation["prediction"]["probability"] is not None:
                prob = explanation["prediction"]["probability"]
                confidence = "high" if prob > 0.8 else "moderate" if prob > 0.6 else "low"
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']} with {confidence} confidence ({prob:.2f})."
            else:
                nl_explanation = f"The {model_type} predicts {explanation['prediction']['value']}."
            
            # Add key factors
            nl_explanation += " This prediction is based on the following factors:\n"
            
            for i, factor in enumerate(factors):
                feature = factor["feature"]
                contrib = abs(factor["contribution"])
                direction = factor["direction"]
                
                # Get the feature value if available
                feature_value = feature_values.get(feature, None)
                value_str = f" (value: {feature_value:.2f})" if feature_value is not None else ""
                
                nl_explanation += f"\n{i+1}. {feature}{value_str} {direction} the prediction"
                
                # Add magnitude description
                if i == 0:  # First factor
                    nl_explanation += " (primary factor)"
                elif contrib < factors[0]["contribution"] * 0.2:  # Small contribution
                    nl_explanation += " (minor factor)"
            
            explanation["natural_language"] = nl_explanation
        
        # Add counterfactual example if we have feature values
        if feature_values:
            # Find the most important feature to change
            if "key_factors" in explanation and explanation["key_factors"]:
                # Use the top factor from our explanation
                top_feature = explanation["key_factors"][0]["feature"]
                direction = explanation["key_factors"][0]["direction"]
                
                # Create counterfactual by modifying the value
                counterfactual = feature_values.copy()
                
                if direction == "increases":
                    # Decrease the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 0.5
                else:
                    # Increase the value to potentially change the prediction
                    counterfactual[top_feature] = counterfactual[top_feature] * 1.5
                
                explanation["counterfactual"] = {
                    "feature_changed": top_feature,
                    "original_value": feature_values[top_feature],
                    "new_value": counterfactual[top_feature],
                    "note": f"Changing {top_feature} might lead to a different prediction"
                }

    except Exception as e:
        logger.error(f"Error creating human readable explanation: {str(e)}")
        explanation["error"] = f"Failed to create explanation: {str(e)}"
    
    return explanation

def generate_structured_report(results, model=None, function=None, output_file="explainability_report.json"):
    """
    Generate a structured report with all explainability results.
    
    :param results: Dictionary of explainability assessment results.
    :param model: The model that was analyzed (optional).
    :param function: The function that was analyzed (optional).
    :param output_file: Path to save the JSON report.
    :return: Dictionary with the full report.
    """
    # Create report structure
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "assessment_type": "model" if model else "function",
        "summary": {},
        "details": results
    }
    
    # Add model or function information
    if model:
        model_info = {
            "type": type(model).__name__,
            "module": model.__module__
        }
        
        # Add additional info based on model type
        if is_tree_based_model(model):
            model_info["category"] = "tree_based"
            
            if hasattr(model, "estimators_"):
                model_info["n_estimators"] = len(model.estimators_)
            
            if hasattr(model, "get_depth"):
                model_info["max_depth"] = model.get_depth()
            elif hasattr(model, "max_depth"):
                model_info["max_depth"] = model.max_depth
            
        elif is_linear_model(model):
            model_info["category"] = "linear"
            
        elif is_neural_network(model):
            model_info["category"] = "neural_network"
        
        report["model_info"] = model_info
    
    if function:
        function_info = {
            "name": function.__name__,
            "module": function.__module__
        }
        
        if "function_complexity" in results:
            function_info.update({
                "complexity_score": results["function_complexity"]["complexity_score"],
                "interpretability": results["function_complexity"]["interpretability"]
            })
        
        report["function_info"] = function_info
    
    # Create summary of key metrics
    summary = {}
    
    # Extract key metrics from results
    if "docstring_quality" in results:
        summary["docstring_quality_score"] = results["docstring_quality"]["quality_score"]
    
    if "function_complexity" in results:
        summary["complexity_score"] = results["function_complexity"]["complexity_score"]
        summary["interpretability"] = results["function_complexity"]["interpretability"]
    
    if "model_structure" in results:
        summary["interpretability_score"] = results["model_structure"]["interpretability_score"]
        summary["interpretability_category"] = results["model_structure"]["interpretability_category"]
    
    if "explanation_quality" in results:
        summary["explanation_quality"] = results["explanation_quality"]["overall_quality"]
    
    if "feature_importance" in results:
        summary["feature_importance_available"] = bool(results["feature_importance"]["methods_available"])
    
    if "decision_rules" in results:
        summary["rule_count"] = results["decision_rules"].get("rule_count", 0)
    
    if "visualizations" in results:
        summary["visualization_count"] = results["visualizations"]["count"]
    
    # Calculate overall explainability score
    explainability_score = 0
    score_components = 0
    
    if "docstring_quality_score" in summary:
        explainability_score += summary["docstring_quality_score"] * 0.1
        score_components += 0.1
    
    if "interpretability" in summary and summary["interpretability"] == "High":
        explainability_score += 100 * 0.2
        score_components += 0.2
    elif "interpretability" in summary and summary["interpretability"] == "Medium":
        explainability_score += 50 * 0.2
        score_components += 0.2
    elif "interpretability" in summary:
        explainability_score += 20 * 0.2
        score_components += 0.2
    
    if "interpretability_score" in summary:
        explainability_score += summary["interpretability_score"] * 0.3
        score_components += 0.3
    
    if "explanation_quality" in summary:
        explainability_score += summary["explanation_quality"] * 0.2
        score_components += 0.2
    
    if "feature_importance_available" in summary and summary["feature_importance_available"]:
        explainability_score += 100 * 0.1
        score_components += 0.1
    
    if "rule_count" in summary and summary["rule_count"] > 0:
        explainability_score += min(100, summary["rule_count"] * 10) * 0.1
        score_components += 0.1
    
    # Normalize score if we have components
    if score_components > 0:
        explainability_score = explainability_score / score_components
    
    # Determine explainability category
    if explainability_score >= 80:
        explainability_category = "High"
    elif explainability_score >= 50:
        explainability_category = "Medium"
    else:
        explainability_category = "Low"
    
    # Add overall score to summary
    summary["explainability_score"] = explainability_score
    summary["explainability_category"] = explainability_category
    
    # Add recommendations based on results
    recommendations = []
    
    if "docstring_quality" in results and results["docstring_quality"]["suggestions"]:
        recommendations.extend(results["docstring_quality"]["suggestions"])
    
    if "function_complexity" in results and results["function_complexity"]["complexity_score"] > 20:
        recommendations.append("Simplify function logic for better interpretability")
    
    if "model_structure" in results and results["model_structure"]["interpretability_category"] == "Low":
        recommendations.append("Consider using a more interpretable model type")
    
    if not summary.get("feature_importance_available", False):
        recommendations.append("Implement feature importance mechanisms")
    
    if "decision_rules" in results and results["decision_rules"].get("rule_count", 0) == 0:
        recommendations.append("Consider models that can provide decision rules")
    
    summary["recommendations"] = recommendations
    
    # Add summary to report
    report["summary"] = summary
    
    # Save report to file
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    return report

def generate_html_report(results, output_file="explainability_report.html"):
    """
    Generate an HTML report for easier readability.
    
    :param results: Explainability results dictionary.
    :param output_file: Path to save HTML report.
    :return: True if successful, False otherwise.
    """
    try:
        # Define HTML template (simplified version)
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Explainability & Interpretability Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2c3e50; }
                .summary { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .metric { margin: 10px 0; }
                .high { color: #27ae60; }
                .medium { color: #f39c12; }
                .low { color: #e74c3c; }
                .recommendations { background-color: #eaf5fb; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .recommendation { margin: 5px 0; }
                .explanation { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .code { font-family: monospace; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Explainability & Interpretability Report</h1>
            <p>Generated: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                {summary_html}
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                {recommendations_html}
            </div>
            
            {details_html}
            
        </body>
        </html>
        """
        
        # Format summary section
        summary_html = ""
        if "summary" in results:
            summary = results["summary"]
            
            # Overall score
            if "explainability_score" in summary:
                score = summary["explainability_score"]
                category = summary["explainability_category"]
                summary_html += f"<div class='metric'><strong>Overall Explainability:</strong> "
                summary_html += f"<span class='{category.lower()}'>{score:.1f}/100 ({category})</span></div>"
            
            # Other key metrics
            metrics_to_show = [
                ("docstring_quality_score", "Docstring Quality"),
                ("complexity_score", "Function Complexity"),
                ("interpretability_score", "Model Interpretability"),
                ("explanation_quality", "Explanation Quality")
            ]
            
            for key, label in metrics_to_show:
                if key in summary:
                    summary_html += f"<div class='metric'><strong>{label}:</strong> {summary[key]:.1f}/100</div>"
        
        # Format recommendations section
        recommendations_html = "<ul>"
        if "summary" in results and "recommendations" in results["summary"]:
            for rec in results["summary"]["recommendations"]:
                recommendations_html += f"<li class='recommendation'>{rec}</li>"
        else:
            recommendations_html += "<li>No specific recommendations available.</li>"
        recommendations_html += "</ul>"
        
        # Format details section
        details_html = ""
        
        # Docstring information
        if "docstring_quality" in results:
            details_html += "<h2>Docstring Analysis</h2>"
            doc_quality = results["docstring_quality"]
            
            details_html += "<div class='explanation'>"
            if doc_quality["has_docstring"]:
                details_html += f"<p>Docstring quality score: {doc_quality['quality_score']:.1f}/100</p>"
                details_html += "<ul>"
                details_html += f"<li>Length: {doc_quality['length']} characters</li>"
                details_html += f"<li>Parameters documented: {'Yes' if doc_quality['has_parameters'] else 'No'}</li>"
                details_html += f"<li>Return values documented: {'Yes' if doc_quality['has_return_info'] else 'No'}</li>"
                details_html += f"<li>Examples included: {'Yes' if doc_quality['has_examples'] else 'No'}</li>"
                details_html += "</ul>"
            else:
                details_html += "<p>No docstring found.</p>"
            details_html += "</div>"
        
        # Function complexity
        if "function_complexity" in results:
            details_html += "<h2>Function Complexity Analysis</h2>"
            complexity = results["function_complexity"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Complexity score: {complexity['complexity_score']:.1f}</p>"
            details_html += f"<p>Interpretability: <span class='{complexity['interpretability'].lower()}'>{complexity['interpretability']}</span></p>"
            details_html += "<ul>"
            details_html += f"<li>Lines of code: {complexity['lines_of_code']}</li>"
            details_html += f"<li>Nested depth: {complexity['nested_depth']}</li>"
            details_html += f"<li>Branching factor: {complexity['branching_factor']:.2f}</li>"
            details_html += f"<li>Conditional statements: {complexity.get('conditional_counts', 'N/A')}</li>"
            details_html += f"<li>Loops: {complexity.get('loop_counts', 'N/A')}</li>"
            details_html += "</ul>"
            details_html += "</div>"
        
        # Model structure
        if "model_structure" in results:
            details_html += "<h2>Model Structure Analysis</h2>"
            structure = results["model_structure"]
            
            details_html += "<div class='explanation'>"
            details_html += f"<p>Model type: {structure['model_type']}</p>"
            details_html += f"<p>Interpretability score: {structure['interpretability_score']:.1f}/100</p>"
            details_html += f"<p>Interpretability category: <span class='{structure['interpretability_category'].lower()}'>{structure['interpretability_category']}</span></p>"
            
            # Add structure details
            if "structure_info" in structure and structure["structure_info"]:
                details_html += "<h3>Structure Details</h3>"
                details_html += "<table>"
                details_html += "<tr><th>Property</th><th>Value</th></tr>"
                
                for key, value in structure["structure_info"].items():
                    # Skip complex nested structures
                    if isinstance(value, (dict, list)):
                        continue
                    details_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
                details_html += "</table>"
            
            details_html += "</div>"
        
        # Feature importance
        if "feature_importance" in results:
            details_html += "<h2>Feature Importance Analysis</h2>"
            importance = results["feature_importance"]
            
            details_html += "<div class='explanation'>"
            if importance["methods_available"]:
                details_html += f"<p>Available methods: {', '.join(importance['methods_available'])}</p>"
                
                # Show feature importance for each method
                for method, values in importance["feature_importance"].items():
                    details_html += f"<h3>Method: {method}</h3>"
                    
                    if isinstance(values, dict):
                        # Create a table for feature importance values
                        details_html += "<table>"
                        details_html += "<tr><th>Feature</th><th>Importance</th></tr>"
                        
                        # Sort by importance value
                        sorted_features = sorted(values.items(), key=lambda x: x[1], reverse=True)
                        
                        for feature, value in sorted_features[:10]:  # Show top 10
                            details_html += f"<tr><td>{feature}</td><td>{value:.4f}</td></tr>"
                        
                        details_html += "</table>"
                    else:
                        details_html += "<p>Feature importance values available but feature names not provided.</p>"
            else:
                details_html += "<p>No feature importance methods available for this model.</p>"
            
            if "errors" in importance and importance["errors"]:
                details_html += "<h3>Errors</h3>"
                details_html += "<ul>"
                for error in importance["errors"]:
                    details_html += f"<li>{error}</li>"
                details_html += "</ul>"
            
            details_html += "</div>"
        
        # Decision rules
        if "decision_rules" in results:
            details_html += "<h2>Decision Rules</h2>"
            rules = results["decision_rules"]
            
            details_html += "<div class='explanation'>"
            if "rules" in rules and rules["rules"]:
                details_html += f"<p>Number of rules: {rules.get('rule_count', len(rules['rules']))}</p>"
                details_html += "<ol>"
                for rule in rules["rules"]:
                    details_html += f"<li><code>{rule}</code></li>"
                details_html += "</ol>"
            else:
                details_html += "<p>No decision rules available or applicable for this model.</p>"
                if "error" in rules:
                    details_html += f"<p>Error: {rules['error']}</p>"
            details_html += "</div>"
        
        # Human-readable explanation
        if "human_explanation" in results:
            details_html += "<h2>Human-Readable Explanation</h2>"
            explanation = results["human_explanation"]
            
            details_html += "<div class='explanation'>"
            if "error" not in explanation:
                if "natural_language" in explanation:
                    details_html += f"<p>{explanation['natural_language'].replace('\n', '<br>')}</p>"
                
                if "key_factors" in explanation:
                    details_html += "<h3>Key Factors</h3>"
                    details_html += "<table>"
                    details_html += "<tr><th>Feature</th><th>Contribution</th><th>Direction</th></tr>"
                    
                    for factor in explanation["key_factors"]:
                        details_html += (f"<tr><td>{factor['feature']}</td>"
                                        f"<td>{factor['contribution']:.4f}</td>"
                                        f"<td>{factor['direction']}</td></tr>")
                    
                    details_html += "</table>"
                
                if "counterfactual" in explanation:
                    details_html += "<h3>Counterfactual Example</h3>"
                    cf = explanation["counterfactual"]
                    details_html += f"<p>If <strong>{cf['feature_changed']}</strong> were "
                    details_html += f"changed from {cf['original_value']} to {cf['new_value']}, "
                    details_html += "the prediction might change.</p>"
            else:
                details_html += f"<p>Error generating explanation: {explanation['error']}</p>"
            details_html += "</div>"
        
        # Visualizations
        if "visualizations" in results:
            details_html += "<h2>Visualizations</h2>"
            visualizations = results["visualizations"]
            
            if "visualizations" in visualizations and visualizations["visualizations"]:
                for viz in visualizations["visualizations"]:
                    details_html += "<div class='explanation'>"
                    details_html += f"<h3>{viz['type'].replace('_', ' ').title()}</h3>"
                    
                    if "path" in viz:
                        # Check if the path exists and is accessible
                        if os.path.exists(viz["path"]):
                            # For HTML report, use relative paths
                            rel_path = os.path.relpath(viz["path"], os.path.dirname(output_file))
                            details_html += f"<img src='{rel_path}' alt='{viz['type']}' />"
                        else:
                            details_html += f"<p>Visualization file not found: {viz['path']}</p>"
                    
                    if "feature" in viz:
                        details_html += f"<p>Feature: {viz['feature']}</p>"
                    
                    if "note" in viz:
                        details_html += f"<p>Note: {viz['note']}</p>"
                    
                    details_html += "</div>"
            else:
                details_html += "<p>No visualizations available.</p>"
        
        # Fill in the template
        html_content = html_template.format(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_html=summary_html,
            recommendations_html=recommendations_html,
            details_html=details_html
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return False

def evaluate_model_explainability(model, X=None, y=None, feature_names=None, output_dir="explainability_results"):
    """
    Comprehensive assessment of model explainability.
    
    :param model: The model to evaluate.
    :param X: Input data (optional, enables advanced assessments).
    :param y: Target data (optional, enables advanced assessments).
    :param feature_names: List of feature names (optional).
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "model_type": type(model).__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Analyze model structure
    logger.info("Analyzing model structure...")
    results["model_structure"] = analyze_model_structure(model)
    
    # Extract feature importance
    logger.info("Extracting feature importance...")
    results["feature_importance"] = extract_feature_importance(model, feature_names, X, y)
    
    # Generate decision rules if possible
    logger.info("Generating decision rules...")
    results["decision_rules"] = generate_decision_rules(model, feature_names)
    
    # If we have input data, perform more advanced evaluations
    if X is not None:
        # Generate visualizations
        logger.info("Generating visualizations...")
        results["visualizations"] = generate_visualizations(
            model, X, y, feature_names, 
            output_dir=os.path.join(output_dir, "visualizations")
        )
        
        # Generate example-specific explanations
        if len(X) > 0:
            logger.info("Generating instance explanation...")
            # Use the first instance as an example
            results["human_explanation"] = create_human_readable_explanation(
                model, X, y, feature_names, instance_idx=0
            )
            
            # Generate explanation quality assessment
            if LIME_AVAILABLE:
                lime_explanation = generate_lime_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(lime_explanation)
            elif SHAP_AVAILABLE:
                shap_explanation = generate_shap_explanation(
                    model, X, feature_names, instance_idx=0
                )
                results["explanation_quality"] = evaluate_explanation_quality(shap_explanation)
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, model=model, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def evaluate_function_explainability(function, output_dir="explainability_results"):
    """
    Comprehensive assessment of function explainability.
    
    :param function: The function to evaluate.
    :param output_dir: Directory to save results.
    :return: Dictionary with all assessment results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dict
    results = {
        "function_name": function.__name__,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Check docstring quality
    logger.info("Checking docstring quality...")
    results["docstring_quality"] = check_docstring_quality(function)
    
    # Analyze function complexity
    logger.info("Analyzing function complexity...")
    results["function_complexity"] = analyze_function_complexity(function)
    
    # Extract function source code
    try:
        results["function_source"] = inspect.getsource(function)
    except:
        results["function_source"] = "Source code not available"
    
    # Generate reports
    logger.info("Generating reports...")
    generate_structured_report(
        results, function=function, 
        output_file=os.path.join(output_dir, "explainability_report.json")
    )
    
    generate_html_report(
        results,
        output_file=os.path.join(output_dir, "explainability_report.html")
    )
    
    logger.info(f"Explainability assessment complete. Results saved to {output_dir}")
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Explainability & Interpretability Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--function",
        help="Function name to analyze"
    )
    
    parser.add_argument(
        "--module",
        help="Module containing the function or model"
    )
    
    parser.add_argument(
        "--model",
        help="Name of model object to analyze (must be in the specified module)"
    )
    
    parser.add_argument(
        "--data",
        help="Path to data file for model analysis (CSV, pickle, etc.)"
    )
    
    parser.add_argument(
        "--target-column",
        help="Name of target column in the data file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="explainability_results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "html", "both"],
        default="both",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

# Example function to be analyzed
def example_function(x):
    """
    Simple function that squares a number.
    
    This function takes an input number and returns its square.
    
    :param x: The number to be squared.
    :return: The square of the input number.
    
    Example:
        >>> example_function(4)
        16
    """
    return x ** 2

# Run explainability evaluation
if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)
    
    # Handle function evaluation
    if args.function:
        try:
            if args.module:
                # Import the module containing the function
                module = importlib.import_module(args.module)
                function = getattr(module, args.function)
            else:
                # Use the example function if no module specified
                function = example_function
            
            print(f"Evaluating explainability of function '{function.__name__}'...")
            evaluate_function_explainability(function, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating function: {e}")
            sys.exit(1)
    
    # Handle model evaluation
    elif args.model:
        try:
            if not args.module:
                logger.error("Module must be specified when analyzing a model")
                sys.exit(1)
            
            # Import the module containing the model
            module = importlib.import_module(args.module)
            model = getattr(module, args.model)
            
            # Load data if provided
            X = None
            y = None
            feature_names = None
            
            if args.data:
                print(f"Loading data from {args.data}...")
                
                if args.data.endswith('.csv'):
                    if not PANDAS_AVAILABLE:
                        logger.error("Pandas is required to load CSV files")
                        sys.exit(1)
                    
                    data = pd.read_csv(args.data)
                    
                    if args.target_column and args.target_column in data.columns:
                        y = data[args.target_column]
                        X = data.drop(columns=[args.target_column])
                    else:
                        # Assume all columns are features
                        X = data
                    
                    feature_names = X.columns.tolist()
                
                elif args.data.endswith('.pkl') or args.data.endswith('.pickle'):
                    with open(args.data, 'rb') as f:
                        data = pickle.load(f)
                    
                    if isinstance(data, dict):
                        X = data.get('X') or data.get('data')
                        y = data.get('y') or data.get('target')
                        feature_names = data.get('feature_names')
                    else:
                        # Assume it's just the features
                        X = data
            
            print(f"Evaluating explainability of model '{args.model}'...")
            evaluate_model_explainability(model, X, y, feature_names, args.output_dir)
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            sys.exit(1)
    
    else:
        # No specific function or model specified, run on example function
        print("No function or model specified, using example function...")
        evaluate_function_explainability(example_function, args.output_dir)
    
    print(f"Explainability assessment complete. Results saved to {args.output_dir}")

"""
TODO:
- Add support for more advanced model types (e.g., ensemble methods, deep learning frameworks)
- Implement benchmarking against "gold standard" human interpretations
- Add support for explainability of embeddings and representations
- Include more advanced LIME and SHAP visualization options
- Implement counterfactual explanation generation with optimization
- Add evaluation of explanation faithfulness and stability
- Support for multi-modal explanations (text + visualization)
- Add natural language explanation generation with templates
- Integrate with popular model registries and governance frameworks
- Extend report generation with more detailed legal and compliance reporting
"""
