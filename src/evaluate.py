import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from models import FEATURE_COLUMNS


def zero_effort_attack_per_user(df, max_users=None):
  
    users = df['user_id'].unique()
    if max_users is not None:
        users = users[:max_users]
    
    print(f"Zero-effort attack analysis for {len(users)} users...")
    
    results = []
    for i, target_user in enumerate(users):
        target_sessions = df[df['user_id'] == target_user]
        
        if len(target_sessions) < 10:
            continue
        
        #split the target user's data=70% for training- 30% for testing
        n_train = int(len(target_sessions) * 0.7)
        target_train = target_sessions.iloc[:n_train]
        target_test = target_sessions.iloc[n_train:]
        
        #get imposter sessions from all other users
        imposter_sessions = df[df['user_id'] != target_user]
        
        #sample imposters for training so it;s balanced with target
        imposter_train = imposter_sessions.sample(
            n=min(len(imposter_sessions), len(target_train) * 3),
            random_state=42
        )
        
        #build the training set
        X_train = pd.concat([
            target_train[FEATURE_COLUMNS],
            imposter_train[FEATURE_COLUMNS]
        ])
        y_train = np.array([1] * len(target_train) + [0] * len(imposter_train))
        
       
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        #start training Random Forest as it was the best performer from baseline
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        #first test: Can the system correctly accept the legitimate user?
        X_legit_test = scaler.transform(target_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit_test)
        true_accept_rate = np.mean(legit_preds == 1)
        
        #secondd test: Can the system correctly reject each imposter?
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
    
    users = df['user_id'].unique()
    
    if target_user is None:
        #pick user with most sessions for best demonstration
        session_counts = df.groupby('user_id').size()
        target_user = session_counts.idxmax()
    
    print(f"Graduated knowledge attack on user {target_user}...")
    
    target_sessions = df[df['user_id'] == target_user]
    imposter_sessions = df[df['user_id'] != target_user]
    
    results = []
    for n_train_samples in sample_sizes:
        if n_train_samples > len(target_sessions) - 5:
            continue
        
        #use first n samples for training, rest for testing
        target_train = target_sessions.iloc[:n_train_samples]
        target_test = target_sessions.iloc[n_train_samples:]
        
        #fixed imposter training set
        imposter_train = imposter_sessions.sample(
            n=min(len(imposter_sessions), n_train_samples * 3),
            random_state=42
        )
        
        #build and train
        X_train = pd.concat([
            target_train[FEATURE_COLUMNS],
            imposter_train[FEATURE_COLUMNS]
        ])
        y_train = np.array([1] * len(target_train) + [0] * len(imposter_train))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        
        #test legitimate acceptance
        X_legit_test = scaler.transform(target_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit_test)
        true_accept_rate = np.mean(legit_preds == 1)
        
        #test imposter rejection (sample of 50 random imposters)
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
    
    train_users = train_df['user_id'].unique()
    test_users = test_df['user_id'].unique()
    
    if max_users is not None:
        test_users = test_users[:max_users]
    
    print(f"Cross-dataset attack: train on {train_name}, test on {test_name}...")
    print(f"  Training users: {len(train_users)}, Test users: {len(test_users)}")
    
    #build a general model from the training dataset-- not user specific
   
    
    #the strategy is for each test user, train on all training dataset users as imposters
    #and the test user's own data as legitimate then test whether the model can distinguish the test user from training users
    
    results = []
    for i, test_user in enumerate(test_users):
        test_user_sessions = test_df[test_df['user_id'] == test_user]
        
        if len(test_user_sessions) < 5:
            continue
        
       
        n_train = int(len(test_user_sessions) * 0.7)
        user_train = test_user_sessions.iloc[:n_train]
        user_test = test_user_sessions.iloc[n_train:]
        
        #use random sample from training dataset as imposters
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
        
        #test legitimate acceptance
        X_legit = scaler.transform(user_test[FEATURE_COLUMNS])
        legit_preds = clf.predict(X_legit)
        true_accept = np.mean(legit_preds == 1)
        
        #test rejection of training dataset users
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