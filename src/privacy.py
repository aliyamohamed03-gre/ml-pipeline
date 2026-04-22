import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import prepare_binary_classification, compute_metrics, FEATURE_COLUMNS


def apply_laplace_noise(X, epsilon, sensitivity=1.0, random_state=42):
    
    rng = np.random.default_rng(random_state)
    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=X.shape)
    return X + noise


def train_with_privacy(df, classifier_name, epsilon, max_users=None):
   
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
        
        #apply differential privacy noise but only to training data
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
   
    all_results = []
    
    #include none (baseline) plus the epsilon values
    eps_values = [None] + list(epsilons)
    
    print(f"Privacy evaluation on {dataset_name}:")
    for classifier in ['RandomForest', 'KNN', 'GradientBoosting']:
        for eps in eps_values:
            results = train_with_privacy(df, classifier, eps)
            results['dataset'] = dataset_name
            all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)