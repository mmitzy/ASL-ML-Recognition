import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from Evaluate import Evaluate
import matplotlib.pyplot as plt
import time
import sys

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

    def restore_checkpoint(self, model):
        model.load_state_dict(self.best_weights)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Print a progress bar to the terminal
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def print_training_dashboard(epoch, total_epochs, train_loss, train_acc, val_loss=None, val_acc=None, time_elapsed=0):
    """
    Print a training dashboard with current metrics
    """
    print("\n" + "="*80)
    print(f"{'NEURAL NETWORK TRAINING DASHBOARD':^80}")
    print("="*80)
    print(f"Epoch: {epoch}/{total_epochs} ({(epoch/total_epochs)*100:.1f}% complete)")
    print(f"Time Elapsed: {time_elapsed:.2f}s")
    print("-"*80)
    print(f"Training Loss:     {train_loss:.6f}")
    print(f"Training Accuracy: {train_acc:.2f}%")
    
    if val_loss is not None and val_acc is not None:
        print(f"Validation Loss:   {val_loss:.6f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")
    
    print("="*80)


def clear_lines(n):
    """Clear n lines in the terminal"""
    for _ in range(n):
        sys.stdout.write('\x1b[1A')  # Move cursor up
        sys.stdout.write('\x1b[2K')  # Clear line


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, learning_rate=0.001):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, num_epochs, device, validation_loader=None, early_stopping_patience=7):
        self.to(device)  
        self.train() 
        
        training_losses = []
        training_accuracies = []
        validation_losses = []
        validation_accuracies = []
        
        early_stopping = EarlyStopping(patience=early_stopping_patience) if validation_loader else None
        
        # Initialize timing
        total_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"{'STARTING NEURAL NETWORK TRAINING':^60}")
        print(f"{'='*60}")
        print(f"Total Epochs: {num_epochs}")
        print(f"Batches per Epoch: {len(train_loader)}")
        print(f"Early Stopping: {'Enabled' if early_stopping else 'Disabled'}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            running_loss = 0.0
            correct = 0
            total = 0
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                batch_start_time = time.time()
                
                images, labels = images.to(device), labels.to(device)
                
                self.optimizer.zero_grad()
                
                outputs = self(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Real-time batch progress
                current_acc = 100 * correct / total if total > 0 else 0
                batch_time = time.time() - batch_start_time
                
                # Update progress bar
                progress_suffix = f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}% | Batch Time: {batch_time:.3f}s"
                print_progress_bar(batch_idx + 1, len(train_loader), 
                                 prefix=f"Training Batch", 
                                 suffix=progress_suffix, 
                                 length=40)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            training_losses.append(epoch_loss)
            training_accuracies.append(epoch_accuracy)

            # Validation phase
            val_loss, val_accuracy = 0, 0
            if validation_loader:
                print(f"\nValidating...")
                val_start_time = time.time()
                val_loss, val_accuracy = self._validate(validation_loader, device)
                val_time = time.time() - val_start_time
                
                validation_losses.append(val_loss)
                validation_accuracies.append(val_accuracy)
                
                print(f"Validation completed in {val_time:.2f}s")
            
            # Calculate epoch timing
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - total_start_time
            
            # Display comprehensive dashboard
            print_training_dashboard(
                epoch + 1, num_epochs, 
                epoch_loss, epoch_accuracy,
                val_loss if validation_loader else None,
                val_accuracy if validation_loader else None,
                total_elapsed
            )
            
            # Detailed metrics output
            if validation_loader:
                print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
                print(f"  Train Loss: {epoch_loss:.6f} | Train Acc: {epoch_accuracy:.2f}%")
                print(f"  Val Loss:   {val_loss:.6f} | Val Acc:   {val_accuracy:.2f}%")
                print(f"  Epoch Time: {epoch_time:.2f}s | Total Time: {total_elapsed:.2f}s")
                
                # Early stopping check
                if early_stopping and early_stopping(val_loss, self):
                    print(f"\n{'='*60}")
                    print(f"{'EARLY STOPPING TRIGGERED':^60}")
                    print(f"{'='*60}")
                    print(f"Training stopped at epoch {epoch+1}")
                    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
                    break
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
                print(f"  Loss: {epoch_loss:.6f} | Accuracy: {epoch_accuracy:.2f}%")
                print(f"  Epoch Time: {epoch_time:.2f}s | Total Time: {total_elapsed:.2f}s")
        
        # Training completion summary
        final_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"{'TRAINING COMPLETED':^60}")
        print(f"{'='*60}")
        print(f"Total Training Time: {final_time:.2f}s")
        print(f"Average Time per Epoch: {final_time/len(training_losses):.2f}s")
        print(f"Final Training Loss: {training_losses[-1]:.6f}")
        print(f"Final Training Accuracy: {training_accuracies[-1]:.2f}%")
        if validation_loader:
            print(f"Final Validation Loss: {validation_losses[-1]:.6f}")
            print(f"Final Validation Accuracy: {validation_accuracies[-1]:.2f}%")
        print(f"{'='*60}\n")
        
        # Plot training metrics
        if validation_loader:
            self.plot_training_metrics_with_validation(training_losses, training_accuracies, validation_losses, validation_accuracies)
        else:
            self.plot_training_metrics(training_losses, training_accuracies)

    def _validate(self, validation_loader, device):
        """Helper method for validation during training with progress visualization"""
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(validation_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Real-time validation progress
                current_acc = 100 * correct / total if total > 0 else 0
                progress_suffix = f"Val Loss: {loss.item():.4f} | Val Acc: {current_acc:.2f}%"
                print_progress_bar(batch_idx + 1, len(validation_loader), 
                                 prefix="Validation", 
                                 suffix=progress_suffix, 
                                 length=40)
        
        self.train()  # Set back to training mode
        return val_loss / len(validation_loader), 100 * correct / total

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path, device):
        self.load_state_dict(torch.load(file_path))
        self.to(device)
        print(f"Model loaded from {file_path}")

    def plot_training_metrics(self, training_losses, training_accuracies):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(training_losses, color='tab:blue', label='Training Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:orange')
        ax2.plot(training_accuracies, color='tab:orange', label='Training Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title('Training Loss and Accuracy')
        plt.tight_layout()
        plt.savefig('training_metrics_nn.png', dpi=300)
        print("Training metrics saved as 'training_metrics_nn.png'")
        plt.show()

    def plot_training_metrics_with_validation(self, training_losses, training_accuracies, validation_losses, validation_accuracies):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Loss plot
        ax1.plot(training_losses, color='tab:blue', label='Training Loss')
        ax1.plot(validation_losses, color='tab:red', label='Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(training_accuracies, color='tab:orange', label='Training Accuracy')
        ax2.plot(validation_accuracies, color='tab:green', label='Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_metrics_nn_with_validation.png', dpi=300)
        print("Training metrics with validation saved as 'training_metrics_nn_with_validation.png'")
        plt.show()

# -------------------------------
# Example Usage

# Hyperparameters
input_dim = 64 * 64 * 3  
hidden_dim = 512  
num_classes = 29  
learning_rate = 0.001
num_epochs = 10
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

train_data_path = "archive/asl_alphabet_train"
test_data_path = "archive/asl_alphabet_test"
model_save_path = "training_files/simple_nn_model.pth"

if not os.path.exists(train_data_path):
    raise ValueError(f"Training data path {train_data_path} does not exist!")
if not os.path.exists(test_data_path):
    raise ValueError(f"Test data path {test_data_path} does not exist!")

train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

# Create validation split
from torch.utils.data import random_split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = SimpleNeuralNetwork(input_dim, hidden_dim, num_classes, learning_rate)

if os.path.exists(model_save_path):
    print("Model already exists, skipping training.")
    model.load_model(model_save_path, device)
else:
    print("Training the model with validation and early stopping...")
    model.train_model(train_loader, num_epochs, device, validation_loader=val_loader, early_stopping_patience=5)
    model.save_model(model_save_path)

print("Evaluating the model...")
evaluator = Evaluate() 
accuracy = evaluator.evaluate_model(model, test_loader, device)
