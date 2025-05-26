import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import os
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

class SimpleNN(nn.Module): #input_size is 786 as given by CLMBR
    def __init__(self, input_size, hidden_size=256):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        return x

def load_features():
    with open("features/clmbr_features.pkl", "rb") as f:
        clmbr_features = pickle.load(f)
    
    X_matrix = np.array(clmbr_features["data_matrix"])
    patient_ids = np.array(clmbr_features["patient_ids"])
    
    features_df = pd.DataFrame(X_matrix, index=patient_ids)
    features_df.index = features_df.index.astype(int)
    return features_df

def train_model(model, train_loader, val_loader, device, patience=3, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    total_loss = 0
    batch_count = 0
    epoch = 0
    
    while True:  # loop until early stopping
        epoch += 1
        print(f"\nEpoch {epoch}")
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            
            # batch loss for every batch
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        #  epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Epoch Loss: {avg_epoch_loss:.4f}")
        
        # validation after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                outputs = model(val_X)
                val_loss += criterion(outputs, val_y.unsqueeze(1)).item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("New best model saved!")
        else:
            patience_counter += 1
            print(f"no improvement for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"early stopping triggered after {epoch} epochs")
                break
        
        model.train()
    
    # best model during training/eval
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from training")
    
    return model

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

def process_task(task_name, features_df, device):
    # Load data
    with open(f"tasks/{task_name}/all_shots_data.json", "r") as f:
        all_data = json.load(f)
    
    # Only SHOT="-1"
    shot = "-1"
    fold_data = all_data[task_name][shot]["0"]  # Only use fold 0
    
    train_idxs = fold_data["train_idxs"]
    val_idxs = fold_data["val_idxs"]
    
    X_train = features_df.values[train_idxs]
    y_train = np.array(fold_data["label_values_train_k"])
    X_val = features_df.values[val_idxs]
    y_val = np.array(fold_data["label_values_val_k"])

    # data distribution
    print("\nData Distribution:")
    print(f"Train labels: {np.bincount(y_train.astype(int))}")
    print(f"Val labels: {np.bincount(y_val.astype(int))}")
    
    #  normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    #  tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    #  data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # init and train 
    model = SimpleNN(input_size=X_train.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, device, lr=0.001)
    
    # predict
    model.eval()
    with torch.no_grad():
        y_proba = model(X_val_tensor.to(device)).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)
    
    # confusion matrix
    print_confusion_matrix(y_val, y_pred)
    
    # metrics
    roc_auc = roc_auc_score(y_val, y_proba)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # actual class labels
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_dirs = [d for d in os.listdir("tasks") if os.path.isdir(os.path.join("tasks", d))]
    
    print("Loading...")
    features_df = load_features()
    all_results = []
    
    tuning_mode = False  # tuning or not
    if tuning_mode:
        task_dirs = ["new_lupus"]
        print("\n=== TUNING MODE: Only new_lupus task ===")
    
    for task_name in task_dirs:
        print(f"\nTask name: {task_name}")
        try:
            result = process_task(task_name, features_df, device)
            all_results.append(result)
            print(f"Train samples: {result['n_train']}, Val samples: {result['n_val']}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}, Accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"Error processing {task_name}: {str(e)}")
    
    results_df = pd.DataFrame(all_results)
    # results_df.to_csv("neural_network_results_tuning.csv", index=False)
    # print("\nResults saved to neural_network_results_tuning.csv")
    results_df.to_csv("neural_network_results.csv", index=False)
    print("\nResults saved to neural_network_results.csv")

if __name__ == "__main__":
    main() 