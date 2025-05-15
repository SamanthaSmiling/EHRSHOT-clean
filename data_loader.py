import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class EHRSHOTDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_features(feature_path="features/clmbr_features.pkl"):
    
    with open(feature_path, "rb") as f:
        clmbr_features = pickle.load(f)
    
    X_matrix = np.array(clmbr_features["data_matrix"])
    patient_ids = np.array(clmbr_features["patient_ids"])
    
    features_df = pd.DataFrame(X_matrix, index=patient_ids)
    features_df.index = features_df.index.astype(int)
    
    return features_df

def load_task_data(task_name="new_lupus", shot="8", fold=None):
    
    with open(f"{task_name}/all_shots_data.json", "r") as f:
        all_data = json.load(f)
    
    if fold is not None:
        # Load single fold
        fold_data = all_data[task_name][shot][fold]
        return fold_data
    else:
        # Load all folds
        return all_data[task_name][shot]

def prepare_single_fold_data(features_df, fold_data):
    
    train_idxs = fold_data["train_idxs"]
    val_idxs = fold_data["val_idxs"]
    
    X_train = features_df.values[train_idxs]
    X_val = features_df.values[val_idxs]
    
    y_train = np.array(fold_data["label_values_train_k"])
    y_val = np.array(fold_data["label_values_val_k"])
    
    return X_train, X_val, y_train, y_val

def prepare_all_folds_data(features_df, task_data, shot):
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    
    if shot == "-1":
        # For shot="-1", there is only one fold
        fold_data = task_data["0"]
        X_train, X_val, y_train, y_val = prepare_single_fold_data(features_df, fold_data)
        return X_train, X_val, y_train, y_val
    else:
        # For other shots, process 5 folds
        for fold in ["0", "1", "2", "3", "4"]:
            fold_data = task_data[fold]
            X_train, X_val, y_train, y_val = prepare_single_fold_data(features_df, fold_data)
            
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_val_list.append(X_val)
            y_val_list.append(y_val)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        
        return X_train, X_val, y_train, y_val

def get_data_loaders(X_train, X_val, y_train, y_val, batch_size=16):
    
    train_dataset = EHRSHOTDataset(X_train, y_train)
    val_dataset = EHRSHOTDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2,  # Larger batch size for validation
        shuffle=False
    )
    
    return train_loader, val_loader

def load_and_prepare_data(task_name="new_lupus", shot="8", fold=None, batch_size=16):
    
    # Load features
    features_df = load_features()
    
    # Load task data
    task_data = load_task_data(task_name, shot, fold)
    
    if fold is not None:
        # Prepare single fold data
        X_train, X_val, y_train, y_val = prepare_single_fold_data(features_df, task_data)
    else:
        # Prepare all folds data
        X_train, X_val, y_train, y_val = prepare_all_folds_data(features_df, task_data, shot)
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        X_train, X_val, y_train, y_val, 
        batch_size=batch_size
    )
    
    return train_loader, val_loader 