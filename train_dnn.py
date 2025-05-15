import torch
from dnn_model import EHRSHOTDNN, train_model, evaluate_model
from data_loader import load_and_prepare_data

def main():
    # Configuration
    # TASK_NAME = "lab_anemia"
    TASK_NAME = "new_lupus"
    SHOT = "16" 
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Load and prepare data
    train_loader, val_loader = load_and_prepare_data(
        task_name=TASK_NAME,
        shot=SHOT,
        batch_size=BATCH_SIZE
    )
    # **early stop** necessary. 1 hidden layer, 100  .AUC concerned.  -1 shot only. 
    
    # Initialize model
    model = EHRSHOTDNN(
        input_dim=768,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3
    )
    
    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Evaluate the model
    print("\nFinal Evaluation:")
    evaluate_model(trained_model, val_loader)
    
    # Save the model
    torch.save(trained_model.state_dict(), f"trained_models/{TASK_NAME}_dnn_{SHOT}shot.pth")

if __name__ == "__main__":
    main() 