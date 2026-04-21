"""
Differential privacy mechanisms for behavioural biometric features.

Implements the Laplace mechanism for adding calibrated noise to feature
vectors before classifier training. This tests the privacy-accuracy
trade-off across multiple epsilon values.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import prepare_binary_classification, compute_metrics, FEATURE_COLUMNS


def apply_laplace_noise(X, epsilon, sensitivity=1.0, random_state=42):
    """
    Add Laplace noise to a feature matrix for differential privacy.
    
    Noise is drawn from a Laplace distribution with scale = sensitivity / epsilon.
    Smaller epsilon produces larger noise and stronger privacy guarantees.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already standardised).
    epsilon : float
        Privacy budget. Smaller values = stronger privacy.
    sensitivity : float, default 1.0
        Maximum change any single record can induce on the output.
        For z-score normalised features, 1.0 is a reasonable bound.
    random_state : int
        Seed for reproducibility.
    
    Returns
    -------
    np.ndarray
        Feature matrix with noise added.
    """
    rng = np.random.default_rng(random_state)
    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=X.shape)
    return X + noise


def train_with_privacy(df, classifier_name, epsilon, max_users=None):
    """
    Train classifier on noisy features for all users in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.
    classifier_name : str
        One of 'RandomForest', 'KNN', 'GradientBoosting'.
    epsilon : float or None
        Privacy budget. If None, no noise is added (baseline).
    max_users : int or None
        If set, only evaluate this many users.
    
    Returns
    -------
    pd.DataFrame
        Per-user metrics.
    """
    users = df['user_id'].unique()
    if max_users is not None:
        users = users[:max_users]
    
    epsilon_label = f"eps={epsilon}" if epsilon is not None else "baseline"
    print(f"  {classifier_name} at {epsilon_label} - {len(users)} users...")
    
    results = []
    for user in users:
        if (df['user_id'] == user).sum() < 5:
            continue
        
        X, y = prepare_binary_classification(df, user)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply differential privacy noise (only to training data)
        if epsilon is not None:
            X_train = apply_laplace_noise(X_train, epsilon)
        
        if classifier_name == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif classifier_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        elif classifier_name == 'GradientBoosting':
            clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred)
        metrics['user_id'] = user
        metrics['classifier'] = classifier_name
        metrics['epsilon'] = epsilon if epsilon is not None else float('inf')
        results.append(metrics)
    
    return pd.DataFrame(results)


def run_privacy_evaluation(df, dataset_name, epsilons=(0.5, 1.0, 2.0, 5.0)):
    """
    Run all classifiers across all epsilon values plus baseline.
    
    Returns
    -------
    pd.DataFrame
        Combined per-user metrics.
    """
    all_results = []
    
    # Include None (baseline) plus the epsilon values
    eps_values = [None] + list(epsilons)
    
    print(f"Privacy evaluation on {dataset_name}:")
    for classifier in ['RandomForest', 'KNN', 'GradientBoosting']:
        for eps in eps_values:
            results = train_with_privacy(df, classifier, eps)
            results['dataset'] = dataset_name
            all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)