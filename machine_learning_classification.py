# pip install numpy pandas scikit-learn keras tensorflow matplotlib seaborn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from collections import defaultdict

# Simulate binary classification dataset
np.random.seed(0)
n_samples = 200
n_features = 100
X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'gene{i}' for i in range(n_features)])
y = np.random.randint(0, 2, n_samples)

# Feature selection
def select_top_features(X, y, k=30):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]

# Model definition
models = {
    "RandomForest": (RandomForestClassifier(), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }),
    "ElasticNet": (LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000), {
        'C': [0.1, 1.0],
        'l1_ratio': [0.2, 0.5, 0.8]
    }),
    "NaiveBayes": (GaussianNB(), {}),
    "SVM": (SVC(probability=True), {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    })
}

# feature selection
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = defaultdict(list)
selected_features_freq = defaultdict(int)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection
    top_features = select_top_features(X_train, y_train, k=30)
    for f in top_features:
        selected_features_freq[f] += 1

    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    for name, (model, param_grid) in models.items():
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_sel, y_train)
        preds = grid.predict(X_test_sel)
        acc = accuracy_score(y_test, preds)
        results[name].append(acc)

# CNN establishment on reshaped data
from sklearn.preprocessing import StandardScaler

cnn_acc = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Feature selection
    top_features = select_top_features(X_train, y_train, k=30)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[top_features])
    X_test_scaled = scaler.transform(X_test[top_features])

    # Reshape for CNN
    X_train_cnn = X_train_scaled.reshape(-1, 30, 1)
    X_test_cnn = X_test_scaled.reshape(-1, 30, 1)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, 1)),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_cnn, y_train_cat, epochs=20, batch_size=16, verbose=0)
    _, acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    cnn_acc.append(acc)

# result summary
print("Model Accuracy (mean ± std):")
for model_name, accs in results.items():
    print(f"{model_name}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"CNN: {np.mean(cnn_acc):.3f} ± {np.std(cnn_acc):.3f}")

# feature selection (final)
top_selected = sorted(selected_features_freq.items(), key=lambda x: x[1], reverse=True)
print("\nTop selected features across CV folds:")
for gene, freq in top_selected[:10]:
    print(f"{gene}: {freq} times")