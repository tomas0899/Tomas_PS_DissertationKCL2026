import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#TOOLS EEG MODELS:
#=================================================================================
#=================================================================================
#=================================================================================
# Function #1

# -------------------------------------------------
# Helper: safe filename
# -------------------------------------------------
def sanitize_filename(text):
    """
    Convert text into a safe filename string.
    """
    text = str(text)
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# -------------------------------------------------
# Function: plot confusion matrix as percentages
# -------------------------------------------------
def plot_confusion_matrix_percent(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    patient_id=None,
    labels=None,
    save_pdf_path=None,
    show_plot=False
):
    """
    Plot a row-normalized confusion matrix in percentage format.
    Each row sums to 100%.

    Optionally:
    - includes patient ID in the title
    - saves the figure as PDF
    """

    # Force label order to match class_names
    if labels is None:
        labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Avoid division by zero if one class is absent in y_true
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    # Add patient ID to plot title if provided
    if patient_id is not None:
        title = f"Patient {patient_id} - {title}"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Add percentage text inside each cell
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            ax.text(
                j, i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    # Save figure as PDF
    if save_pdf_path is not None:
        save_pdf_path = Path(save_pdf_path)
        save_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_pdf_path, format="pdf", bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return cm, cm_percent


# -------------------------------------------------
# Function: evaluate model on a dataset
# -------------------------------------------------
def evaluate_and_plot_3_1(
    model,
    X_data,
    y_true,
    class_names,
    dataset_name="Validation",
    patient_id=None,
    output_dir=None,
    labels=None,
    show_plot=False,
    output_prefix=None
):
    """
    1. Predicts labels
    2. Prints classification table
    3. Prints global metrics
    4. Plots confusion matrix in percentages
    5. Saves confusion matrix PDF
    6. Saves confusion matrix as CSV
    7. Saves classification table as CSV
    8. Optionally adds patient ID to outputs
    """

    y_pred = model.predict(X_data)

    # Force label order to match class_names
    if labels is None:
        labels = list(range(len(class_names)))

    # -------------------------------------------------
    # Classification report as dataframe
    # -------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    # Add patient and dataset information to the classification table
    if patient_id is not None:
        report_df.insert(0, "patient_id", patient_id)

    report_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------
    print(f"\n{'='*40}")

    if patient_id is not None:
        print(f"PATIENT: {patient_id}")

    print(f"{dataset_name.upper()} SET")
    print(f"{'='*40}")

    print("\nClassification table:")
    print(report_df)

    print("\nGlobal metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    # -------------------------------------------------
    # Build output filenames
    # -------------------------------------------------
    save_pdf_path = None
    cm_counts_csv_path = None
    cm_percent_csv_path = None
    classification_csv_path = None
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        dataset_tag = sanitize_filename(dataset_name)
    
        if output_prefix is not None:
            base_stem = f"{sanitize_filename(output_prefix)}_{dataset_tag}"
        else:
            patient_tag = (
                f"PAT-{sanitize_filename(patient_id)}"
                if patient_id is not None
                else "PAT-unknown"
            )
            base_stem = f"{patient_tag}_{dataset_tag}"
    
        file_stem = f"{base_stem}_confusion_matrix"
    
        save_pdf_path = output_dir / f"{file_stem}.pdf"
        cm_counts_csv_path = output_dir / f"{file_stem}_counts.csv"
        cm_percent_csv_path = output_dir / f"{file_stem}_percent.csv"
    
        classification_csv_path = output_dir / f"{base_stem}_classification_table.csv"
    # -------------------------------------------------
    # Plot and optionally save confusion matrix PDF
    # -------------------------------------------------
    cm_counts, cm_percent = plot_confusion_matrix_percent(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"{dataset_name} Confusion Matrix (%)",
        patient_id=patient_id,
        labels=labels,
        save_pdf_path=save_pdf_path,
        show_plot=show_plot
    )

    # -------------------------------------------------
    # Return confusion matrices as dataframes
    # -------------------------------------------------
    cm_counts_df = pd.DataFrame(
        cm_counts,
        index=[f"True {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names]
    )

    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"True {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names]
    )

    # Add patient and dataset labels to confusion matrix tables
    if patient_id is not None:
        cm_counts_df.insert(0, "patient_id", patient_id)
        cm_percent_df.insert(0, "patient_id", patient_id)

    cm_counts_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    cm_percent_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------
    if output_dir is not None:

        cm_counts_df.to_csv(cm_counts_csv_path, index=True)
        cm_percent_df.to_csv(cm_percent_csv_path, index=True)

        # This saves the printed classification table as CSV
        report_df.to_csv(
            classification_csv_path,
            index=True,
            index_label="class_or_metric"
        )

        print("\nSaved outputs:")
        print(f"PDF: {save_pdf_path}")
        print(f"CSV counts: {cm_counts_csv_path}")
        print(f"CSV percent: {cm_percent_csv_path}")
        print(f"Classification table CSV: {classification_csv_path}")

    return {
        "patient_id": patient_id,
        "dataset": dataset_name,
        "y_pred": y_pred,
        "classification_table": report_df,
        "confusion_counts": cm_counts_df,
        "confusion_percent": cm_percent_df,
        "pdf_path": save_pdf_path,
        "confusion_counts_csv_path": cm_counts_csv_path,
        "confusion_percent_csv_path": cm_percent_csv_path,
        "classification_csv_path": classification_csv_path
    }
#=================================================================================
#=================================================================================
#=================================================================================
# Function #2
def find_best_temporal_split_3_2(
    y,
    ideal_train=0.70,
    ideal_val=0.15,
    ideal_test=0.15,
    train_search_range=(0.70, 0.90),
    val_search_range=(0.05, 0.20),
    ratio_weight=3
):
    """
    Find the best temporal train/validation/test split.

    The function preserves chronological order and searches for split boundaries
    that are close to the desired train/validation/test proportions while also
    keeping the class ratio similar across the three sets.

    Parameters
    ----------
    y : pd.Series
        Target variable ordered chronologically.

    ideal_train : float
        Desired proportion of the dataset for the training set.

    ideal_val : float
        Desired proportion of the dataset for the validation set.

    ideal_test : float
        Desired proportion of the dataset for the test set.

    train_search_range : tuple
        Range of possible train end positions as proportions of the dataset.
        Example: (0.70, 0.90) searches train_end between 70% and 90%.

    val_search_range : tuple
        Range of possible validation sizes as proportions of the dataset.
        Example: (0.05, 0.20) searches validation sizes between 5% and 20%.

    ratio_weight : float
        Weight applied to the class-ratio score.
        Higher values force the split to preserve class balance more strongly.

    Returns
    -------
    train_end : int
        Index where the training set ends.

    val_end : int
        Index where the validation set ends.

    best_score : float
        Score of the selected split. Lower is better.
    """

    # Number of samples in the cleaned dataset
    n = len(y)

    # Global proportion of class 1
    # In your binary setup: class 1 = seizure
    global_ratio = y.mean()

    # Candidate positions where the training set may end
    train_candidates = range(
        int(train_search_range[0] * n),
        int(train_search_range[1] * n),
        max(1, n // 1000)
    )

    # Candidate validation set sizes
    val_candidates = range(
        int(val_search_range[0] * n),
        int(val_search_range[1] * n),
        max(1, n // 1000)
    )

    best = None
    best_score = np.inf

    # Search for the best temporal split
    for train_end in train_candidates:
        for val_size in val_candidates:

            val_end = train_end + val_size

            # Skip invalid split where validation exceeds dataset length
            if val_end >= n:
                continue

            # Temporal split
            y_train_candidate = y.iloc[:train_end]
            y_val_candidate = y.iloc[train_end:val_end]
            y_test_candidate = y.iloc[val_end:]

            # Require both classes in train, validation, and test
            if (
                y_train_candidate.nunique() < 2
                or y_val_candidate.nunique() < 2
                or y_test_candidate.nunique() < 2
            ):
                continue

            # Measure how close the split sizes are to the ideal proportions
            train_frac = len(y_train_candidate) / n
            val_frac = len(y_val_candidate) / n
            test_frac = len(y_test_candidate) / n

            size_score = (
                abs(train_frac - ideal_train)
                + abs(val_frac - ideal_val)
                + abs(test_frac - ideal_test)
            )

            # Measure how close the class ratios are to the global class ratio
            ratio_score = (
                abs(y_train_candidate.mean() - global_ratio)
                + abs(y_val_candidate.mean() - global_ratio)
                + abs(y_test_candidate.mean() - global_ratio)
            )

            # Combined score
            score = size_score + ratio_score * ratio_weight

            # Keep the best split found so far
            if score < best_score:
                best_score = score
                best = (train_end, val_end)

    # Safety check in case no valid split was found
    if best is None:
        raise ValueError(
            "No valid temporal split found. "
            "Try changing candidate ranges or check class distribution over time."
        )

    train_end, val_end = best

    return train_end, val_end, best_score
#=================================================================================
#=================================================================================
#=================================================================================
# Function #3

def train_svm_gridsearch_3_3(
    X_train,
    y_train,
    n_splits=4,
    scoring="f1_macro",
    param_grid=None,
    n_jobs=-1,
    verbose=1
):
    """
    Train an SVM classifier using a sklearn Pipeline and temporal cross-validation.

    The pipeline includes:
    1. StandardScaler:
       Standardizes features using only the training fold during cross-validation.

    2. SVC with RBF kernel:
       Non-linear Support Vector Machine classifier.

    TimeSeriesSplit is used to preserve chronological order during hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used to rank models during GridSearchCV.

    param_grid : dict or None
        Hyperparameter grid for the SVM.
        If None, a default grid is used.

    n_jobs : int
        Number of CPU cores used by GridSearchCV.
        -1 means use all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    best_model : sklearn Pipeline
        Best trained pipeline selected by GridSearchCV.

    grid_search : GridSearchCV
        Full fitted GridSearchCV object containing results, scores, and best parameters.
    """

    # --------------------------------------------------------
    # 1. Build machine learning pipeline
    # --------------------------------------------------------
    # StandardScaler is inside the pipeline to avoid data leakage.
    # During cross-validation, it is fitted only on each training fold.

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced"))
    ])

    # --------------------------------------------------------
    # 2. Define temporal cross-validation
    # --------------------------------------------------------
    # TimeSeriesSplit preserves chronological order.
    # This is important for EEG windows because future data should not be used
    # to validate past data.

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --------------------------------------------------------
    # 3. Define default hyperparameter grid
    # --------------------------------------------------------
    # C controls the penalty for misclassification.
    # gamma controls the influence radius of each sample in the RBF kernel.

    if param_grid is None:
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
        }

    # --------------------------------------------------------
    # 4. Set up GridSearchCV
    # --------------------------------------------------------
    # GridSearchCV tests all combinations of C and gamma.
    # refit=True means that the best model is refitted on the full training set.

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # --------------------------------------------------------
    # 5. Fit grid search on training set only
    # --------------------------------------------------------
    # Validation and test sets are not used here.

    grid_search.fit(X_train, y_train)

    # --------------------------------------------------------
    # 6. Retrieve best model
    # --------------------------------------------------------

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nBest mean CV {scoring}:")
    print(grid_search.best_score_)

    return best_model, grid_search
#=================================================================================
#=================================================================================
#=================================================================================
# Function #4



def train_decision_tree_gridsearch_3_4(
    X_train,
    y_train,
    n_splits: int = 4,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """
    Trains a Decision Tree classifier using TimeSeriesSplit and GridSearchCV.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features. For your case, this can be the PCA-transformed features.

    y_train : array-like or Series
        Training labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used by GridSearchCV. Default is "f1_macro".

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores used by GridSearchCV. -1 uses all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    grid_dt : GridSearchCV object
        Full fitted GridSearchCV object.

    best_model_dt : Pipeline
        Best fitted Decision Tree pipeline.

    best_params_dt : dict
        Best hyperparameters found.

    best_score_dt : float
        Best mean cross-validation score.
    """

    # ----------------------------------------------------------
    # 1. Build Decision Tree pipeline
    # ----------------------------------------------------------
    pipeline_dt = Pipeline([
        ("tree", DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced"
        ))
    ])

    # ----------------------------------------------------------
    # 2. Define temporal cross-validation
    # ----------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ----------------------------------------------------------
    # 3. Define hyperparameter grid
    # ----------------------------------------------------------
    param_grid_dt = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [2, 3, 4, 5, 6, None],
        "tree__min_samples_split": [2, 5, 10, 20],
        "tree__min_samples_leaf": [1, 2, 5, 10]
    }

    # ----------------------------------------------------------
    # 4. Set up GridSearchCV
    # ----------------------------------------------------------
    grid_dt = GridSearchCV(
        estimator=pipeline_dt,
        param_grid=param_grid_dt,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # ----------------------------------------------------------
    # 5. Train model
    # ----------------------------------------------------------
    grid_dt.fit(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Extract best model and results
    # ----------------------------------------------------------
    best_model_dt = grid_dt.best_estimator_
    best_params_dt = grid_dt.best_params_
    best_score_dt = grid_dt.best_score_

    print("Best Decision Tree parameters:")
    print(best_params_dt)

    print("\nBest mean CV macro F1:")
    print(best_score_dt)

    return grid_dt, best_model_dt, best_params_dt, best_score_dt


#=================================================================================
#=================================================================================
#=================================================================================
# Function #5

def train_random_forest_gridsearch_3_5(
    X_train,
    y_train,
    n_splits: int = 4,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """
    Trains a Random Forest classifier using TimeSeriesSplit and GridSearchCV.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features. In your case, this can be the PCA-transformed features.

    y_train : array-like or Series
        Training labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used by GridSearchCV. Default is "f1_macro".

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores used by GridSearchCV. -1 uses all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    grid_rf : GridSearchCV object
        Full fitted GridSearchCV object.

    best_model_rf : Pipeline
        Best fitted Random Forest pipeline.

    best_params_rf : dict
        Best hyperparameters found.

    best_score_rf : float
        Best mean cross-validation score.
    """

    # ----------------------------------------------------------
    # 1. Build Random Forest pipeline
    # ----------------------------------------------------------
    pipeline_rf = Pipeline([
        ("forest", RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1
        ))
    ])

    # ----------------------------------------------------------
    # 2. Define temporal cross-validation
    # ----------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ----------------------------------------------------------
    # 3. Define hyperparameter grid
    # ----------------------------------------------------------
    param_grid_rf = {
        "forest__n_estimators": [100, 200, 500],
        "forest__max_depth": [3, 5, 10, None],
        "forest__min_samples_split": [2, 5, 10],
        "forest__min_samples_leaf": [1, 2, 5, 10],
        "forest__max_features": ["sqrt", "log2", None]
    }

    # ----------------------------------------------------------
    # 4. Set up GridSearchCV
    # ----------------------------------------------------------
    grid_rf = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid_rf,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # ----------------------------------------------------------
    # 5. Train model
    # ----------------------------------------------------------
    grid_rf.fit(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Extract best model and results
    # ----------------------------------------------------------
    best_model_rf = grid_rf.best_estimator_
    best_params_rf = grid_rf.best_params_
    best_score_rf = grid_rf.best_score_

    print("Best Random Forest parameters:")
    print(best_params_rf)

    print("\nBest mean CV macro F1:")
    print(best_score_rf)

    return grid_rf, best_model_rf, best_params_rf, best_score_rf