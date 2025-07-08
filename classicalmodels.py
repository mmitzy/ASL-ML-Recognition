import os
import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
from tqdm import tqdm
from collections import Counter
from Evaluate import Evaluate

# Constants
DATA_PATH = "archive/asl_alphabet_train"
TEST_PATH = "archive/asl_alphabet_test"
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)  # Increased from 32x32 for better feature representation
N_PCA_COMPONENTS = 200  # Increased from 100 to retain more information
TOTAL_SUBSET_SIZE = 14500  # Increased to use ~500 per class

MODEL_DIR = "training_files"
KNN_PATH = os.path.join(MODEL_DIR, "knn_model.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca_transformer.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom dataset loader
class CustomImageFolder(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        classes = sorted(os.listdir(root_dir))
        self.classes = classes  # Add this for compatibility with Evaluate class
        print(f"Loading dataset from {root_dir}...")
        for idx, class_name in enumerate(tqdm(classes, desc="Reading classes")):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            class_folder = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_folder, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        image_np = np.array(image).transpose((2, 0, 1))  # HWC to CHW
        image_tensor = torch.tensor(image_np).float() / 255.0
        return image_tensor, label


# Load datasets
train_dataset = CustomImageFolder(DATA_PATH)
test_dataset = CustomImageFolder(TEST_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_dataset.class_to_idx)


# Helper to flatten dataset
def flatten_dataset(loader, desc="Flattening dataset"):
    X, y = [], []
    total = len(loader.dataset)
    with tqdm(total=total, desc=desc) as pbar:
        for images, labels in loader:
            for img, label in zip(images, labels):
                X.append(img.view(-1).numpy())
                y.append(label.item())
                pbar.update(1)
    return np.array(X), np.array(y)


print("Flattening training and test datasets...")
X_train_full, y_train_full = flatten_dataset(train_loader, desc="Flattening Train")
X_test, y_test = flatten_dataset(test_loader, desc="Flattening Test")

# Stratified sampling: pick 2800 samples balanced across all classes
print("Sampling a balanced training subset of 2800 samples...")
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=TOTAL_SUBSET_SIZE,
    stratify=y_train_full,
    random_state=42
)
print("Subset label distribution:", Counter(y_train))

# Ensure training_files directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature scaling before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Incremental PCA
print("Applying Incremental PCA for dimensionality reduction...")
if not os.path.exists(PCA_PATH):
    print("Fitting Incremental PCA...")
    pca = IncrementalPCA(n_components=N_PCA_COMPONENTS, batch_size=500)
    pca.fit(X_train_scaled)
    joblib.dump(pca, PCA_PATH)
    print("PCA model saved.")
else:
    print("Loading existing PCA model...")
    pca = joblib.load(PCA_PATH)


# PCA transform helper with progress
def pca_transform_with_progress(pca_model, data, desc="Transforming with PCA"):
    chunk_size = 5000
    transformed = []
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    with tqdm(total=total_chunks, desc=desc) as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            transformed_chunk = pca_model.transform(chunk)
            transformed.append(transformed_chunk)
            pbar.update(1)
    return np.vstack(transformed)


X_train_pca = pca_transform_with_progress(pca, X_train_scaled, desc="Transforming Train with PCA")
X_test_pca = pca_transform_with_progress(pca, X_test_scaled, desc="Transforming Test with PCA")

# Train and save models with cross-validation
print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

if not os.path.exists(KNN_PATH):
    print("Training KNN model with optimized parameters...")
    # Grid search for better hyperparameters (single-threaded for compatibility)
    from sklearn.model_selection import GridSearchCV
    knn_params = {
        'n_neighbors': [5, 7, 9],
        'weights': ['distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn_base = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn_base, knn_params, cv=3, scoring='accuracy', n_jobs=1)  # Changed from -1 to 1
    knn_grid.fit(X_train_pca, y_train)
    knn = knn_grid.best_estimator_
    print(f"Best KNN parameters: {knn_grid.best_params_}")
    print(f"Best CV score: {knn_grid.best_score_:.4f}")
    joblib.dump(knn, KNN_PATH)
    print("KNN model saved.")
else:
    knn = joblib.load(KNN_PATH)
    print("KNN model loaded.")

if not os.path.exists(SVM_PATH):
    print("Training SVM model with optimized parameters...")
    # Optimized SVM parameters (single-threaded)
    svm_params = {
        'C': [0.5, 1.0, 2.0],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }
    svm_base = SVC(random_state=42, probability=True)
    svm_grid = GridSearchCV(svm_base, svm_params, cv=3, scoring='accuracy', n_jobs=1)  # Changed from -1 to 1
    svm_grid.fit(X_train_pca, y_train)
    svm = svm_grid.best_estimator_
    print(f"Best SVM parameters: {svm_grid.best_params_}")
    print(f"Best CV score: {svm_grid.best_score_:.4f}")
    joblib.dump(svm, SVM_PATH)
    print("SVM model saved.")
else:
    svm = joblib.load(SVM_PATH)
    print("SVM model loaded.")

if not os.path.exists(RF_PATH):
    print("Training Random Forest model with optimized parameters...")
    # Optimized Random Forest parameters (single-threaded)
    rf_params = {
        'n_estimators': [150, 200, 250],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5]
    }
    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='accuracy', n_jobs=1)  # Changed from -1 to 1
    rf_grid.fit(X_train_pca, y_train)
    rf = rf_grid.best_estimator_
    print(f"Best RF parameters: {rf_grid.best_params_}")
    print(f"Best CV score: {rf_grid.best_score_:.4f}")
    joblib.dump(rf, RF_PATH)
    print("Random Forest model saved.")
else:
    rf = joblib.load(RF_PATH)
    print("Random Forest model loaded.")


# Evaluation wrapper
class SklearnWrapper(torch.nn.Module):
    def __init__(self, model, pca, scaler, num_classes):
        super().__init__()
        self.model = model
        self.pca = pca
        self.scaler = scaler
        self.num_classes = num_classes

    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, x):
        x_np = x.view(x.size(0), -1).cpu().numpy()
        x_np_scaled = self.scaler.transform(x_np)
        x_np_pca = self.pca.transform(x_np_scaled)
        preds = self.model.predict(x_np_pca)
        preds_tensor = torch.tensor(preds, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(preds_tensor, num_classes=self.num_classes).float()
        return one_hot


# Evaluate models with enhanced metrics
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

models = {
    'KNN': knn,
    'SVM': svm, 
    'Random Forest': rf
}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    # Standard evaluation
    evaluator = Evaluate()
    wrapped_model = SklearnWrapper(model, pca, scaler, NUM_CLASSES)
    accuracy, precision, recall, f1 = evaluator.evaluate_model(wrapped_model, test_loader, DEVICE)
    
    # Cross-validation on training data - DISABLED
    # print(f"\nCross-validation for {model_name}:")
    # cv_scores, mean_cv, std_cv = evaluator.cross_validate_sklearn_model(model, X_train_pca, y_train)
    # print(f"CV Accuracy: {mean_cv*100:.2f}% ± {std_cv*100:.2f}%")
    
    print(f"\n{model_name} Summary:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision*100:.2f}%")
    print(f"Test Recall: {recall*100:.2f}%")
    print(f"Test F1-Score: {f1*100:.2f}%")
    print("-" * 40)