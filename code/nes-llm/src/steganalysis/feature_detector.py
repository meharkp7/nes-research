import pickle
import numpy as np

from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def extract_features(x):

    x = np.asarray(x).flatten()

    hist, _ = np.histogram(x, bins=50, density=True)

    return [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),

        np.median(x),

        np.percentile(x, 25),
        np.percentile(x, 75),

        np.mean(x ** 2),

        np.mean(x > 0),

        skew(x),
        kurtosis(x),

        entropy(hist + 1e-12),
    ]


def main():
    
    with open("real_detector_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    X = np.array([
        x.numpy()
        for x, label in dataset
    ])

    y = np.array([
        label
        for x, label in dataset
    ])

    print("Samples:", len(X))

    feats = np.array([
        extract_features(x)
        for x in X
    ])

    print("Feature Shape:", feats.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        feats,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining RF...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\n====================")
    print("FEATURE DETECTOR")
    print("====================")
    print("Accuracy:", round(acc, 4))

    importances = clf.feature_importances_

    names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "p25",
        "p75",
        "energy",
        "sign_ratio",
        "skew",
        "kurtosis",
        "entropy",
    ]

    print("\nTop Features")

    ranking = sorted(
        zip(names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for name, score in ranking:
        print(name, round(score, 4))
    cm = confusion_matrix(y_test, preds)
    print(cm)
    print("Accuracy:", acc)
    print("Flipped:", 1 - acc)


if __name__ == "__main__":
   
    main()
    