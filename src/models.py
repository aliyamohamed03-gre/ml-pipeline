import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


FEATURE_COLUMNS = [
    'mean_hold_time',
    'mean_inter_key_interval',
    'std_inter_key_interval',
    'total_duration',
]


def prepare_binary_classification(df, target_user, imposter_sample_size=None):
    
    legitimate = df[df['user_id'] == target_user]
    imposters = df[df['user_id'] != target_user]
    
    #sample imposters to match legitimate count (roughly balanced classes)
    if imposter_sample_size is None:
        imposter_sample_size = min(len(imposters), len(legitimate) * 3)
    
    imposters_sampled = imposters.sample(
        n=min(imposter_sample_size, len(imposters)),
        random_state=42
    )
    
    #combine all and label
    legitimate = legitimate.copy()
    legitimate['label'] = 1
    imposters_sampled = imposters_sampled.copy()
    imposters_sampled['label'] = 0
    
    combined = pd.concat([legitimate, imposters_sampled], ignore_index=True)
    
    X = combined[FEATURE_COLUMNS].values
    y = combined['label'].values
    
    return X, y


def compute_metrics(y_true, y_pred):
   
    acc = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer = (far + frr) / 2
    
    return {'accuracy': acc, 'FAR': far, 'FRR': frr, 'EER': eer}


def train_and_evaluate(df, classifier_name='RandomForest', max_users=None):
   
    users = df['user_id'].unique()
    if max_users is not None:
        users = users[:max_users]
    
    print(f"Training {classifier_name} for {len(users)} users...")
    
    results = []
    for i, user in enumerate(users):
        #need at least a few sessions of this user
        if (df['user_id'] == user).sum() < 5:
            continue
        
        X, y = prepare_binary_classification(df, user)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
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
        results.append(metrics)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{len(users)} users")
    
    print(f"  Done. Evaluated {len(results)} users.")
    return pd.DataFrame(results)


def run_full_evaluation(df, dataset_name):
   
    all_results = []
    for classifier in ['RandomForest', 'KNN', 'GradientBoosting']:
        per_user_results = train_and_evaluate(df, classifier)
        per_user_results['dataset'] = dataset_name
        all_results.append(per_user_results)
    
    return pd.concat(all_results, ignore_index=True)