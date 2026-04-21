"""
Attack simulation and security evaluation for behavioural biometric authentication.

Implements three attack scenarios:
1. Zero-effort imposter attack: random users attempt to authenticate as the target
2. Graduated knowledge attack: varying numbers of imposter samples available
3. Cross-dataset generalisation: models trained on one dataset tested on another

These analyses reframe the baseline classification results through a security lens
and produce figures suitable for the evaluation chapter of the dissertation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from models import FEATURE_COLUMNS


def zero_effort_attack_per_user(df, max_users=None):
    """
    Measure each user's vulnerability to zero-effort imposter attacks.
    
    For each target user, every other user attempts to authenticate as them.
    Returns per-user detection rates showing which users are most vulnerable.
    
    This is the standard zero-effort attack model: the imposter makes no attempt
    to mimic the target's behaviour and simply uses their own natural typing.
    """
    users = df['user_id'].unique()
    if max_users is not None:
        users = users[:max_users]
    
    print(f"Zero-effort attack analysis for {len(users)} users...")
    
    results = []
    for i, target_user in enumerate(users):
        target_sessions = df[df['user_id'] == target_user]
        
        if len(target_sessions) < 10:
            continue
        
        # Split target user's data: 70% for training, 30% for testing
        n_train = int(len(target_sessions) * 0.7)
        target_train = target_sessions.iloc[:n_train]
        target_test = target_sessions.iloc[n_train:]
        
        # Get imposter sessions from ALL other users
        imposter_sessions = df[df['user_id'] != target_user]
        
        # Sample imposters for training (balanced with target)
        imposter_train = imposter_sessions.sample(
            n=min(len(imposter_sessions), len(target_train) * 3),
            random_state=42
        )
        
        # Build training set
        X_train = pd.concat([
            target_train[FEATURE_COLUMNS],
            imposter_train[FEATURE_COLUMNS]
        ])
        y_train = np.array([1] * len(target_train) + [0] * len(imposter_train))
        
        # Standardise
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest (best performer from baseline)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        # Test 1: Can the system correctly accept the legitimate user?
        X_legit_test = scaler.transform(target_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit_test)
        true_accept_rate = np.mean(legit_preds == 1)
        
        # Test 2: Can the system correctly reject each imposter?
        other_users = [u for u in users if u != target_user]
        imposter_reject_rates = []
        
        for imposter_user in other_users:
            imposter_data = df[df['user_id'] == imposter_user]
            if len(imposter_data) < 3:
                continue
            
            X_imposter = scaler.transform(imposter_data[FEATURE_COLUMNS])
            imposter_preds = clf.predict(X_imposter)
            reject_rate = np.mean(imposter_preds == 0)
            imposter_reject_rates.append(reject_rate)
        
        mean_reject_rate = np.mean(imposter_reject_rates) if imposter_reject_rates else 0.0
        min_reject_rate = np.min(imposter_reject_rates) if imposter_reject_rates else 0.0
        
        results.append({
            'user_id': target_user,
            'true_accept_rate': true_accept_rate,
            'mean_imposter_reject_rate': mean_reject_rate,
            'min_imposter_reject_rate': min_reject_rate,
            'num_imposters_tested': len(imposter_reject_rates),
            'target_sessions': len(target_sessions)
        })
        
        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{len(users)} users")
    
    print(f"  Done. Evaluated {len(results)} users.")
    return pd.DataFrame(results)


def graduated_knowledge_attack(df, target_user=None, sample_sizes=(5, 10, 20, 50, 100)):
    """
    Test how detection rate changes as the imposter has more sample data.
    
    Simulates an attacker who has observed varying amounts of the target user's
    typing behaviour. With more observations, the attacker could potentially
    build a better model of the target's behaviour.
    
    In this zero-effort variant, the attacker doesn't actually USE the knowledge
    to mimic behaviour — they just type naturally. But the classifier's training
    set size varies, showing how model robustness scales with enrollment depth.
    """
    users = df['user_id'].unique()
    
    if target_user is None:
        # Pick user with most sessions for best demonstration
        session_counts = df.groupby('user_id').size()
        target_user = session_counts.idxmax()
    
    print(f"Graduated knowledge attack on user {target_user}...")
    
    target_sessions = df[df['user_id'] == target_user]
    imposter_sessions = df[df['user_id'] != target_user]
    
    results = []
    for n_train_samples in sample_sizes:
        if n_train_samples > len(target_sessions) - 5:
            continue
        
        # Use first n samples for training, rest for testing
        target_train = target_sessions.iloc[:n_train_samples]
        target_test = target_sessions.iloc[n_train_samples:]
        
        # Fixed imposter training set
        imposter_train = imposter_sessions.sample(
            n=min(len(imposter_sessions), n_train_samples * 3),
            random_state=42
        )
        
        # Build and train
        X_train = pd.concat([
            target_train[FEATURE_COLUMNS],
            imposter_train[FEATURE_COLUMNS]
        ])
        y_train = np.array([1] * len(target_train) + [0] * len(imposter_train))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        # Test legitimate acceptance
        X_legit_test = scaler.transform(target_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit_test)
        true_accept_rate = np.mean(legit_preds == 1)
        
        # Test imposter rejection (sample of 50 random imposters)
        imposter_test_users = [u for u in users if u != target_user]
        np.random.seed(42)
        imposter_test_sample = np.random.choice(
            imposter_test_users,
            size=min(50, len(imposter_test_users)),
            replace=False
        )
        
        reject_rates = []
        for imp_user in imposter_test_sample:
            imp_data = df[df['user_id'] == imp_user]
            if len(imp_data) < 3:
                continue
            X_imp = scaler.transform(imp_data[FEATURE_COLUMNS])
            preds = clf.predict(X_imp)
            reject_rates.append(np.mean(preds == 0))
        
        mean_reject = np.mean(reject_rates) if reject_rates else 0.0
        
        results.append({
            'training_samples': n_train_samples,
            'true_accept_rate': true_accept_rate,
            'mean_imposter_reject_rate': mean_reject,
            'test_sessions': len(target_test)
        })
        
        print(f"  {n_train_samples} training samples: "
              f"accept={true_accept_rate:.3f}, reject={mean_reject:.3f}")
    
    return pd.DataFrame(results)


def cross_dataset_attack(train_df, test_df, train_name, test_name, max_users=None):
    """
    Train a model on one dataset and test on another.
    
    This tests whether behavioural patterns generalise across different
    typing contexts (fixed password vs free text, desktop vs mobile).
    Poor cross-dataset performance indicates context-dependence of
    behavioural biometrics, which has implications for deployment.
    """
    train_users = train_df['user_id'].unique()
    test_users = test_df['user_id'].unique()
    
    if max_users is not None:
        test_users = test_users[:max_users]
    
    print(f"Cross-dataset attack: train on {train_name}, test on {test_name}...")
    print(f"  Training users: {len(train_users)}, Test users: {len(test_users)}")
    
    # Build a general "legitimate typing" model from the training dataset
    # This represents a generic behavioural profile, not user-specific
    
    # Strategy: for each test user, train on ALL training dataset users as imposters
    # and the test user's own data as legitimate
    # Then test whether the model can distinguish the test user from training users
    
    results = []
    for i, test_user in enumerate(test_users):
        test_user_sessions = test_df[test_df['user_id'] == test_user]
        
        if len(test_user_sessions) < 5:
            continue
        
        # Split test user data
        n_train = int(len(test_user_sessions) * 0.7)
        user_train = test_user_sessions.iloc[:n_train]
        user_test = test_user_sessions.iloc[n_train:]
        
        # Use random sample from training dataset as imposters
        imposter_sample = train_df.sample(
            n=min(len(train_df), len(user_train) * 3),
            random_state=42
        )
        
        X_train = pd.concat([
            user_train[FEATURE_COLUMNS],
            imposter_sample[FEATURE_COLUMNS]
        ])
        y_train = np.array([1] * len(user_train) + [0] * len(imposter_sample))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        # Test legitimate acceptance
        X_legit = scaler.transform(user_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit)
        true_accept = np.mean(legit_preds == 1)
        
        # Test rejection of training dataset users
        cross_imposters = train_df.sample(
            n=min(500, len(train_df)),
            random_state=42
        )
        X_cross = scaler.transform(cross_imposters[FEATURE_COLUMNS])
        cross_preds = clf.predict(X_cross)
        cross_reject = np.mean(cross_preds == 0)
        
        results.append({
            'test_user': test_user,
            'true_accept_rate': true_accept,
            'cross_reject_rate': cross_reject
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{len(test_users)} users")
    
    result_df = pd.DataFrame(results)
    result_df['train_dataset'] = train_name
    result_df['test_dataset'] = test_name
    
    print(f"  Done. Mean accept: {result_df['true_accept_rate'].mean():.3f}, "
          f"Mean cross-reject: {result_df['cross_reject_rate'].mean():.3f}")
    
    return result_df