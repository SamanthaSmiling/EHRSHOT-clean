import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
import traceback

def load_features():
    with open("features/clmbr_features.pkl", "rb") as f:
        clmbr_features = pickle.load(f)
    
    X_matrix = np.array(clmbr_features["data_matrix"])
    patient_ids = np.array(clmbr_features["patient_ids"])
    
    features_df = pd.DataFrame(X_matrix, index=patient_ids)
    features_df.index = features_df.index.astype(int)
    return features_df

def process_task(task_name, features_df):
    # task data
    with open(f"tasks/{task_name}/all_shots_data.json", "r") as f:
        all_data = json.load(f)
    
    # Only process SHOT="-1"
    shot = "-1"
    fold_data = all_data[task_name][shot]["0"]  # Only use fold 0
    
    train_idxs = fold_data["train_idxs"]
    val_idxs = fold_data["val_idxs"]
    
    X_train = features_df.values[train_idxs]
    y_train = np.array(fold_data["label_values_train_k"])
    X_val = features_df.values[val_idxs]
    y_val = np.array(fold_data["label_values_val_k"])
    
    model = LogisticRegression(max_iter=100000) # logit model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print_confusion_matrix(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    class_labels = [str(label) for label in report.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    result = {
        'task_name': task_name,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'roc_auc': round(roc_auc, 4),
        'accuracy': round(report['accuracy'], 4)
    }
    
    # metrics for each class
    for label in class_labels:
        result.update({
            f'precision_{label}': round(report[label]['precision'], 4),
            f'recall_{label}': round(report[label]['recall'], 4),
            f'f1_{label}': round(report[label]['f1-score'], 4)
        })
    
    return result

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted  0    1")
    print(f"Actual 0: {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"Actual 1: {cm[1][0]:<5} {cm[1][1]:<5}")
    
    # additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive (recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive (precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive 
    
    print("\nDetailed Metrics:")
    print(f"TP: {sensitivity:.4f}")
    print(f"TN: {specificity:.4f}")
    print(f"PP:   {precision:.4f}")
    print(f"NP:  {npv:.4f}")

def main():
    # all tasks
    task_dirs = [d for d in os.listdir("tasks") if os.path.isdir(os.path.join("tasks", d))]
    
    print("Loading...")
    features_df = load_features()
    
    # each task
    all_results = []
    for task_name in task_dirs:
        print(f"\nTask name : {task_name}")
        try:
            result = process_task(task_name, features_df)
            all_results.append(result)
            print(f"Train samples: {result['n_train']}, Val samples: {result['n_val']}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"Error processing {task_name}: {str(e)}")
            print("Full error traceback:")
            print(traceback.format_exc())
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("logistic_regression_results.csv", index=False)
    print("\nResults saved to logistic_regression_results.csv")

if __name__ == "__main__":
    main() 