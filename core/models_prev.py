import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from tqdm import tqdm

benchmark_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Gaussian NB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis()
}


class BinaryBasicNN:
    def __init__(
            self,
            X=None,
            y=None,
            sample_size=8192,
            epochs=4,
            batch_size=512,
            learning_rate=0.01,
            chromosome=None,
            random_seed=True,
            track_accuracy=False,
            use_gpu=True
    ):
        self.X = X
        self.y = y
        self.sample_size = sample_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.chromosome = chromosome if chromosome is not None else np.random.randint(0, 2, X.shape[1])
        self.random_seed = random_seed
        self.track_accuracy = track_accuracy

        # GPU setup
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")

    def evaluate(self):
        # Select features based on binary chromosome
        selected_indices = np.where(self.chromosome == 1)[0]

        if len(selected_indices) == 0:
            return {
                "chromosome": self.chromosome,
                "acc": 0.0,
                "auc": 0.5,
                "f1": 0.0,
                "logloss_score": 0.0,
                "epoch_acc": None
            }

        X_selected = self.X.iloc[:, selected_indices].to_numpy(dtype=np.float32)
        y_array = self.y.to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_array, test_size=0.3)

        seed = None if self.random_seed else 42
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=min(self.sample_size, len(X_train)), replace=False)

        X_train, y_train = X_train[idx], y_train[idx]

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).unsqueeze(1).to(self.device)
        X_test_t = torch.tensor(X_test).to(self.device)

        train_dl = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)

        model = nn.Sequential(
            nn.Linear(len(selected_indices), 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        epoch_acc = []
        for _ in tqdm(range(self.epochs), desc='Training NN', leave=False, position=0):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            if self.track_accuracy:
                with torch.no_grad():
                    train_probs = torch.sigmoid(model(X_train_t)).cpu().numpy().flatten()
                    train_preds = (train_probs > 0.5).astype(int)
                    epoch_acc.append(np.mean(train_preds == y_train))

        model.eval()

        with torch.no_grad():
            probs = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        acc = np.mean(preds == y_test)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        ll = log_loss(y_test, probs)
        logloss_score = 1 / (1 + ll)

        return {
            "chromosome": self.chromosome,
            "acc": acc,
            "auc": auc,
            "f1": f1,
            "logloss_score": logloss_score,
            "epoch_acc": epoch_acc if self.track_accuracy else None
        }


class BasicNN:
    pass


__all__ = ['benchmark_models', 'BinaryBasicNN', 'BasicNN']