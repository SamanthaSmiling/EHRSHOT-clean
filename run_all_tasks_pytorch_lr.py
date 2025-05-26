import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score
import os
from pathlib import Path

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def load_features():
    with open("features/clmbr_features.pkl", "rb") as f:
        clmbr_features = pickle.load(f)
    
    X_matrix = np.array(clmbr_features["data_matrix"])
    patient_ids = np.array(clmbr_features["patient_ids"])
    
    features_df = pd.DataFrame(X_matrix, index=patient_ids)
    features_df.index = features_df.index.astype(int)
    return features_df

def train_model(model, train_loader, val_loader, device, n_epochs=100):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # training
    for epoch in range(n_epochs):
        # training
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        
        val_loss /= len(val_loader)
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def process_task(task_name, features_df, device):
    # load task data
    with open(f"tasks/{task_name}/all_shots_data.json", "r") as f:
        all_data = json.load(f)
    
    # only process SHOT="-1"
    shot = "-1"
    fold_data = all_data[task_name][shot]["0"]  # only use fold 0
    
    # get train and validation data
    train_idxs = fold_data["train_idxs"]
    val_idxs = fold_data["val_idxs"]
    
    X_train = features_df.values[train_idxs]
    y_train = np.array(fold_data["label_values_train_k"])
    X_val = features_df.values[val_idxs]
    y_val = np.array(fold_data["label_values_val_k"])
    
    # convert to pytorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # initialize and train model
    model = LogisticRegression(input_size=X_train.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device)
    
    # get predictions
    model.eval()
    with torch.no_grad():
        y_proba = model(X_val_tensor.to(device)).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)
    
    # calculate metrics
    roc_auc = roc_auc_score(y_val, y_proba)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # get the actual class labels from the report
    class_labels = [str(label) for label in report.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # store results
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get all task 
    task_dirs = [d for d in os.listdir("tasks") if os.path.isdir(os.path.join("tasks", d))]

    print(f"Tasks: {task_dirs}")
    
    # load features once
    print("Loading...")
    features_df = load_features()
    
    # process each task
    all_results = []
    for task_name in task_dirs:
        print(f"\nTask name: {task_name}")
        try:
            result = process_task(task_name, features_df, device)
            all_results.append(result)
            print(f"Train samples: {result['n_train']}, Val samples: {result['n_val']}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"Error processing {task_name}: {str(e)}")
    
    # save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("pytorch_lr_results.csv", index=False)
    print("\nResults saved to pytorch_lr_results.csv")

if __name__ == "__main__":
    main() 