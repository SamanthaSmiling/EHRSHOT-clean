import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

class EHRSHOTDNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(EHRSHOTDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.BCELoss() # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_preds = np.array(val_preds).squeeze()
        val_labels = np.array(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Training Loss: {train_loss/len(train_loader):.4f}, Validation AUC: {val_auc:.4f}')
        
        # Save model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    val_preds = np.array(val_preds).squeeze()
    val_labels = np.array(val_labels)
    
    # Calculate metrics
    val_auc = roc_auc_score(val_labels, val_preds)
    val_preds_binary = (val_preds > 0.5).astype(int)
    print(f'ROC-AUC: {val_auc:.4f}')
    print('Classification Report:')
    print(classification_report(val_labels, val_preds_binary))
    
    return val_auc, val_preds_binary 