import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import numpy as np

class Evaluate:
    def evaluate_model(self, model, test_loader, device):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        class_names = test_loader.dataset.classes

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

        cm = confusion_matrix(all_labels, all_predictions, labels=range(len(class_names)))

        plt.figure(figsize=(25, 20)) 

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=0, ax=plt.gca())
        plt.title('Confusion Matrix (Results)', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300)
        print("Evaluation plots saved as 'model_evaluation.png'")
        plt.show()

        tp = cm.diagonal()  
        fp = cm.sum(axis=0) - tp  
        fn = cm.sum(axis=1) - tp  

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)

        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        print("\nOverall Metrics:")
        print(f"Precision: {overall_precision * 100:.2f}")
        print(f"Recall: {overall_recall * 100:.2f}")
        print(f"F1-Score: {overall_f1 * 100:.2f}")

        return accuracy, overall_precision, overall_recall, overall_f1

    def cross_validate_sklearn_model(self, model, X, y, cv_folds=5, random_state=42):
        """
        Perform stratified k-fold cross-validation on sklearn models
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Clone and train model
            from sklearn.base import clone
            model_fold = clone(model)
            model_fold.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = model_fold.predict(X_val_fold)
            accuracy = (y_pred == y_val_fold).mean()
            cv_scores.append(accuracy)
            
            print(f"Fold {fold + 1} accuracy: {accuracy * 100:.2f}%")
        
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)
        
        print(f"\nCross-Validation Results:")
        print(f"Mean Accuracy: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
        print(f"Individual fold scores: {[f'{score * 100:.2f}%' for score in cv_scores]}")
        
        return cv_scores, mean_accuracy, std_accuracy

    def evaluate_model_with_confidence(self, model, test_loader, device, confidence_threshold=0.8):
        """
        Evaluate model with confidence-based predictions
        """
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        high_confidence_correct = 0
        high_confidence_total = 0
        all_labels = []
        all_predictions = []
        all_confidences = []

        class_names = test_loader.dataset.classes

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Get probabilities and confidence
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # High confidence predictions
                high_conf_mask = max_probs >= confidence_threshold
                high_confidence_total += high_conf_mask.sum().item()
                high_confidence_correct += ((predicted == labels) & high_conf_mask).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_confidences.extend(max_probs.cpu().numpy())

        accuracy = 100 * correct / total
        high_conf_accuracy = 100 * high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0
        high_conf_coverage = 100 * high_confidence_total / total
        
        print(f'Overall Accuracy: {accuracy:.2f}%')
        print(f'High Confidence Accuracy (≥{confidence_threshold:.1f}): {high_conf_accuracy:.2f}%')
        print(f'High Confidence Coverage: {high_conf_coverage:.2f}%')
        print(f'Average Confidence: {np.mean(all_confidences):.3f}')

        return accuracy, high_conf_accuracy, high_conf_coverage, np.mean(all_confidences)