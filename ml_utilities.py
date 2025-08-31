"""
Machine Learning Utilities

A collection of utility functions and helper classes for common machine learning tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve
import pandas as pd

class MLEvaluator:
    """
    Comprehensive machine learning model evaluation toolkit.
    Provides methods for visualization, performance analysis, and model comparison.
    """
    
    def __init__(self):
        self.results = {}
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """
        Plot an enhanced confusion matrix with percentages.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Class labels for display
        title : str
            Title for the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{title}\nAccuracy: {np.trace(cm)/np.sum(cm):.3f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Also show percentage matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{title} (Percentages)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        -----------
        y_true : array-like
            True binary labels
        y_prob : array-like
            Predicted probabilities for positive class
        title : str
            Title for the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return roc_auc
    
    def plot_learning_curve(self, estimator, X, y, cv=5, title="Learning Curve"):
        """
        Plot learning curve to analyze training vs validation performance.
        
        Parameters:
        -----------
        estimator : sklearn estimator
            The machine learning model
        X : array-like
            Training features
        y : array-like
            Training labels
        cv : int
            Number of cross-validation folds
        title : str
            Title for the plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                         alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                         alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_models(self, models_dict, X_train, X_test, y_train, y_test):
        """
        Compare multiple models and return performance metrics.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary with model names as keys and fitted models as values
        X_train, X_test, y_train, y_test : array-like
            Training and testing data
        
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        results = []
        
        for name, model in models_dict.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_acc = (train_pred == y_train).mean()
            test_acc = (test_pred == y_test).mean()
            
            results.append({
                'Model': name,
                'Training Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Overfitting': train_acc - test_acc
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Test Accuracy', ascending=False)
        
        return df_results
    
    def feature_importance_plot(self, model, feature_names, title="Feature Importance"):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        model : sklearn model
            Fitted model with feature_importances_ attribute
        feature_names : list
            Names of the features
        title : str
            Title for the plot
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
                   rotation=45, ha='right')
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()


class DataPreprocessor:
    """
    Advanced data preprocessing utilities for machine learning pipelines.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values with multiple strategies.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            'auto', 'mean', 'median', 'mode', 'drop', 'forward_fill'
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with handled missing values
        """
        df_processed = df.copy()
        
        if strategy == 'auto':
            for col in df_processed.columns:
                if df_processed[col].dtype in ['int64', 'float64']:
                    # Numerical columns: use median
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                else:
                    # Categorical columns: use mode
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        elif strategy == 'mean':
            df_processed = df_processed.fillna(df_processed.mean())
        elif strategy == 'median':
            df_processed = df_processed.fillna(df_processed.median())
        elif strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'forward_fill':
            df_processed = df_processed.fillna(method='ffill')
        
        return df_processed
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        method : str
            'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
        
        Returns:
        --------
        pd.DataFrame
            Boolean dataframe indicating outliers
        """
        outliers = pd.DataFrame(index=df.index, columns=df.columns, data=False)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = z_scores > threshold
        
        return outliers
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features for better model performance.
        
        Parameters:
        -----------
        X : array-like
            Input features
        degree : int
            Polynomial degree
        
        Returns:
        --------
        array-like
            Polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        return X_poly, poly


def generate_sample_datasets():
    """
    Generate sample datasets for testing machine learning algorithms.
    
    Returns:
    --------
    dict
        Dictionary containing various sample datasets
    """
    from sklearn.datasets import make_classification, make_regression, make_blobs
    
    datasets = {}
    
    # Binary classification dataset
    X_bin, y_bin = make_classification(
        n_samples=1000, n_features=10, n_informative=5, 
        n_redundant=2, n_classes=2, random_state=42
    )
    datasets['binary_classification'] = (X_bin, y_bin)
    
    # Multi-class classification dataset
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=3, random_state=42
    )
    datasets['multiclass_classification'] = (X_multi, y_multi)
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    datasets['regression'] = (X_reg, y_reg)
    
    # Clustering dataset
    X_cluster, y_cluster = make_blobs(
        n_samples=300, centers=4, n_features=2,
        random_state=42, cluster_std=1.0
    )
    datasets['clustering'] = (X_cluster, y_cluster)
    
    return datasets


# Example usage and demonstration
if __name__ == "__main__":
    # This section provides examples of how to use the utilities
    print("Machine Learning Utilities - Example Usage")
    print("==========================================")
    
    # Generate sample data
    datasets = generate_sample_datasets()
    X, y = datasets['binary_classification']
    
    # Initialize evaluator
    evaluator = MLEvaluator()
    
    # Example model comparison
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    comparison_results = evaluator.compare_models(models, X_train, X_test, y_train, y_test)
    print("\nModel Comparison Results:")
    print(comparison_results)
