"""
IR CLA-2 | Task 2 : Text Classification Battle
Compare Decision Tree, KNN, SVM and Naive Bayes on technical newsgroup posts.

Student : Harini K K (RA2411026050041), III CSE AIML A
Dataset : 20 Newsgroups (Lang, 1995) - six computing / science groups
Usage   : python text_classification.py
"""

import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "tech_newsgroups.csv"
OUT = BASE / "outputs"
PLOTS = BASE / "plots"
OUT.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

RANDOM_STATE = 7
TEST_FRACTION = 0.25

CLASSES = {
    "comp.graphics": "Graphics",
    "comp.os.ms-windows.misc": "Windows OS",
    "comp.sys.ibm.pc.hardware": "PC Hardware",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.space": "Space",
}

porter = PorterStemmer()
STOPWORDS = ENGLISH_STOP_WORDS


# --------------------------------------------------------------------------
# 1. Cleaning and preprocessing
# --------------------------------------------------------------------------
def strip_newsgroup_noise(post):
    """Remove parts of a Usenet post that would reveal the group without
    reading the content: the header block (From, Organization, e-mail
    addresses ...), quoted replies and the signature. Only the Subject line
    is kept from the header because it is written by the author."""
    head, _, body = post.partition("\n\n")
    subject = ""
    for line in head.splitlines():
        if line.lower().startswith("subject:"):
            subject = line[8:].replace("Re:", "").strip()
    lines = []
    for line in body.splitlines():
        if line.strip().startswith((">", "|", ":")):
            continue  # quoted text from an earlier message
        if re.search(r"(writes|wrote|says):\s*$", line):
            continue  # "In article <...> someone writes:"
        if line.strip() in ("--", "-- "):
            break  # signature starts here
        lines.append(line)
    return subject + "\n" + "\n".join(lines)


def preprocess(text):
    """Same steps as Task 1: lowercase, alphabetic tokens, stop-word
    removal, Porter stemming."""
    text = re.sub(r"\S+@\S+", " ", text.lower())  # e-mail ids
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # urls
    words = re.findall(r"[a-z]+", text)
    return " ".join(porter.stem(w) for w in words if len(w) > 2 and w not in STOPWORDS)


def load_dataset():
    df = pd.read_csv(DATA)
    df["label"] = df["newsgroup"].map(CLASSES)
    df["body"] = df["text"].apply(strip_newsgroup_noise)
    df["tokens"] = df["body"].apply(preprocess)
    before = len(df)
    df = df[df["tokens"].str.split().str.len() >= 5]  # drop near-empty posts
    df = df.drop_duplicates(subset="tokens").reset_index(drop=True)
    return df, before - len(df)


# --------------------------------------------------------------------------
# 2. Feature extraction and classifiers
# --------------------------------------------------------------------------
def tfidf(ngrams=(1, 1)):
    return TfidfVectorizer(ngram_range=ngrams, min_df=3, max_df=0.9, sublinear_tf=True)


def classifiers():
    return {
        "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance"),
        "SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(alpha=0.1),
    }


def score(name, clf, Xtr, ytr, Xte, yte):
    start = time.perf_counter()
    clf.fit(Xtr, ytr)
    fit_s = time.perf_counter() - start
    start = time.perf_counter()
    pred = clf.predict(Xte)
    pred_s = time.perf_counter() - start
    return pred, {
        "Classifier": name,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, average="macro"),
        "Recall": recall_score(yte, pred, average="macro"),
        "F1-score": f1_score(yte, pred, average="macro"),
        "Train accuracy": accuracy_score(ytr, clf.predict(Xtr)),
        "Fit time (s)": round(fit_s, 4),
        "Predict time (s)": round(pred_s, 4),
    }


# --------------------------------------------------------------------------
# 3. Plots
# --------------------------------------------------------------------------
PALETTE = ["#2E86AB", "#F18F01", "#3B8B5A", "#A23B72"]


def plot_metrics(table):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    ax = table.set_index("Classifier")[metrics].plot(kind="barh", figsize=(10, 5.5), color=PALETTE, width=0.8)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Score (macro average)")
    ax.set_title("Classifier comparison on six technical newsgroups")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "metrics_comparison.png", dpi=150)
    plt.close()


def plot_confusions(cms, labels):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (name, cm) in zip(axes.flat, cms.items()):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}  (accuracy {np.trace(cm) / cm.sum():.3f})")
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_whatif(df):
    pivot = df.pivot(index="Classifier", columns="Representation", values="F1-score")
    pivot = pivot[["Bag-of-words counts", "TF-IDF unigrams", "TF-IDF uni+bigrams"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#9E9E9E", "#2E86AB", "#F18F01"], width=0.75)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Macro F1-score")
    ax.set_xlabel("")
    ax.set_title("What-if: changing the feature representation")
    plt.xticks(rotation=0)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS / "whatif_features.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 4. Main experiment
# --------------------------------------------------------------------------
def main():
    report = []

    def out(line=""):
        print(line)
        report.append(str(line))

    out("TEXT CLASSIFICATION BATTLE - 20 Newsgroups (6 technical groups)")
    out("-" * 64)

    df, dropped = load_dataset()
    out(f"Posts after cleaning          : {len(df)}  (removed {dropped} empty / duplicate posts)")
    counts = df["label"].value_counts()
    out("Posts per class:")
    for lab, n in counts.items():
        out(f"   {lab:<13} {n:>4}  ({100 * n / len(df):.1f}%)")
    out(f"Largest / smallest class ratio: {counts.max() / counts.min():.2f}")
    counts.rename("posts").to_csv(OUT / "class_counts.csv", header=True)

    ex = df.iloc[5]
    (OUT / "cleaning_example.txt").write_text(
        f"CLASS: {ex['label']}\n\n--- RAW POST ---\n{ex['text'][:700]}\n\n"
        f"--- AFTER HEADER / QUOTE REMOVAL ---\n{ex['body'][:500]}\n\n"
        f"--- AFTER PREPROCESSING ---\n{ex['tokens'][:400]}\n")

    labels = sorted(df["label"].unique())
    tr_txt, te_txt, ytr, yte = train_test_split(
        df["tokens"], df["label"], test_size=TEST_FRACTION, stratify=df["label"], random_state=RANDOM_STATE)
    out(f"Training posts: {len(tr_txt)}   Test posts: {len(te_txt)}   (stratified 75/25)")

    vec = tfidf()
    Xtr = vec.fit_transform(tr_txt)
    Xte = vec.transform(te_txt)
    out(f"TF-IDF features: {Xtr.shape[1]}")
    out("")

    rows, cms, preds = [], {}, {}
    for name, clf in classifiers().items():
        pred, row = score(name, clf, Xtr, ytr, Xte, yte)
        rows.append(row)
        preds[name] = pred
        cms[name] = confusion_matrix(yte, pred, labels=labels)
        pd.DataFrame(cms[name], index=labels, columns=labels).to_csv(
            OUT / f"confusion_{name.lower().replace(' ', '_')}.csv")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison_table.csv", index=False)
    out("RESULTS ON TEST SET (macro averages)")
    out(table.round(4).to_string(index=False))

    per_class = pd.DataFrame(
        {n: f1_score(yte, p, average=None, labels=labels) for n, p in preds.items()}, index=labels)
    per_class.to_csv(OUT / "per_class_f1.csv")
    out("\nF1-score per class")
    out(per_class.round(3).to_string())

    best = table.loc[table["F1-score"].idxmax(), "Classifier"]
    out(f"\nBest classifier: {best}")

    # most frequent confusions of the best model
    cm = cms[best].copy()
    np.fill_diagonal(cm, 0)
    pairs = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels))
                    for j in range(len(labels)) if cm[i, j]), reverse=True)[:5]
    out(f"Top confusions for {best} (actual -> predicted):")
    for n, a, p in pairs:
        out(f"   {a} -> {p}: {n}")

    # interpretable terms
    feats = np.array(vec.get_feature_names_out())
    svm = LinearSVC(C=1.0, random_state=RANDOM_STATE).fit(Xtr, ytr)
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xtr, ytr)
    with open(OUT / "important_terms.txt", "w") as fh:
        fh.write("Strongest SVM terms per class\n")
        for i, c in enumerate(svm.classes_):
            fh.write(f"  {c:<13}: {', '.join(feats[np.argsort(-svm.coef_[i])[:12]])}\n")
        fh.write(f"\nDecision tree: depth {tree.get_depth()}, {tree.get_n_leaves()} leaves, "
                 f"{int((tree.feature_importances_ > 0).sum())} of {len(feats)} features used\n")
        fh.write(f"Root-level split terms: {', '.join(feats[np.argsort(-tree.feature_importances_)[:12]])}\n")

    plot_metrics(table)
    plot_confusions(cms, labels)

    # ------------------------------------------------------------------
    # 5. What-if experiment: change the feature representation
    # ------------------------------------------------------------------
    out("\nWHAT-IF: same split, same models, different feature representation")
    reps = {
        "Bag-of-words counts": CountVectorizer(min_df=3, max_df=0.9),
        "TF-IDF unigrams": tfidf((1, 1)),
        "TF-IDF uni+bigrams": tfidf((1, 2)),
    }
    wrows = []
    for rep, v in reps.items():
        A, B = v.fit_transform(tr_txt), v.transform(te_txt)
        for name, clf in classifiers().items():
            _, r = score(name, clf, A, ytr, B, yte)
            wrows.append({"Representation": rep, "Features": A.shape[1], "Classifier": name,
                          "Accuracy": r["Accuracy"], "F1-score": r["F1-score"]})
    whatif = pd.DataFrame(wrows)
    whatif.to_csv(OUT / "whatif_feature_representation.csv", index=False)
    out(whatif.pivot(index="Classifier", columns="Representation", values="F1-score").round(4).to_string())
    out("Feature counts: " + ", ".join(
        f"{r} = {n}" for r, n in whatif.groupby("Representation", sort=False)["Features"].first().items()))
    plot_whatif(whatif)

    (OUT / "summary.json").write_text(json.dumps(
        {"posts": int(len(df)), "train": int(len(tr_txt)), "test": int(len(te_txt)),
         "features": int(Xtr.shape[1]), "best": best}, indent=2))
    (OUT / "console_output.txt").write_text("\n".join(report) + "\n")
    out("\nFinished - see outputs/ and plots/")


if __name__ == "__main__":
    main()
