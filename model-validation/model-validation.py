import os
from pathlib import Path
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
    )
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import shap
from itertools import cycle

MODEL_PATH = './models/'
OUTPUT_PATH = './data/model_output/'
MODEL_FILENAME = 'lgbm_model_20250725_102140.pkl'

custom_colors = [
    "#b7eeee",
    "#c6e5ff",
    "#b4a0f7",
    "#fbccf6",
    "#f8bdbd",
    "#fef6a2",
    "#B0B0B0"
    ]
custom_cmap = ListedColormap(custom_colors)

# Load Model
def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(
        file_path: str
        ) -> pd.DataFrame:
    """Load data from a CSV file
    
    :param file_path: Path to the CSV file
    :return: DataFrame containing the loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def get_most_recent_datadir(
        dir: str = './data/preprocessed'
        ) -> str:
    """Get the most recent subdirectory in a given directory
    """
    all_subdirs = []
    for d in os.listdir(dir):
        _path = os.path.join(dir, d)
        if os.path.isdir(_path):
            all_subdirs.append(_path)
    latest = max(all_subdirs, key=os.path.getmtime)
    return '/'.join(Path(latest).parts)


# Make Predictions 
def predict(
        model,
        X_test: np.array
        ) -> np.array:
    return model.predict(X_test)


def calculate_accuracy(y_true: pd.Series, 
                       y_pred: pd.Series
                       ) -> float:
    """Calculate and return accuracy score.
    """
    return accuracy_score(y_true.values.ravel(), y_pred)


def generate_classification_report(
        y_test: pd.Series,
        y_pred: pd.Series
        ) -> str:
    """Generate classification report using sklearn.metrics.
    """
    return classification_report(y_test.values.ravel(), y_pred)


def save_text_output(
        file_path: str,
        train_accuracy: float,
        test_accuracy: float,
        output: str
        ) -> None:
    """ Save accuracy scores and classification report to text file.
    """
    try:
        with open(file_path, 'w') as f:
            f.write(f"Train Accuracy Score: {train_accuracy:.4f}\n\n")
            f.write(f"Test Accuracy Score: {test_accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(output)
        print(f"Output successfully saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving output: {e}")


def generate_confusion_matrix(
        y_test: pd.Series, 
        y_pred: pd.Series
        ) -> None:
    """Generates and returns a confusion matrix.
    """
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(
        cm: pd.DataFrame
        ) -> plt.Figure:
    """ Plot the confusion matrix using sklearn.metrics
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, 
                xticklabels=[f'Class {i}' for i in range(len(cm))],
                yticklabels=[f'Class {i}' for i in range(len(cm))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return plt.gcf()


def plot_roc_curve(model: object, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series, 
                   class_names: list
                   ) -> plt.Figure:
    """Plot Receiver Operating Characteristic (ROC) curves for multi-class classification.

    Parameters:
    model (sklearn model): sklearn model
    X_test (array-like): Test features
    y_test (array-like): True labels from test set
    class_names (list): List of class labels for the classes
    """

    # Predict probabilities from the trained model
    y_prob = model.predict_proba(X_test)

    # Bin the true labels
    y_test_bin = label_binarize(y_test, classes = class_names)
    n_classes = y_prob.shape[1]

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict() 
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])

    # Calculate micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc_per_class["micro"] = auc(fpr["micro"], tpr["micro"])

    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_per_class["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))

    # ROC curves for micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"], label = f"Micro-average ROC curve (AUC = {roc_auc_per_class['micro']:.2f})", 
             color = "pink", linestyle = ":", linewidth = 2)
    plt.plot(fpr["macro"], tpr["macro"], label = f"Macro-average ROC curve (AUC = {roc_auc_per_class['macro']:.2f})", 
             color = "navy", linestyle = ":", linewidth = 2)

    # Assign colors for each class
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "blue", "purple"])

    # Plot each class ROC curve
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color = color, lw = 2, 
                 label = f"{class_names[i]} ROC (AUC = {roc_auc_per_class[i]:.2f})")

    # Plot diagonal line (no discrimination line)
    plt.plot([0, 1], [0, 1], "k--", lw = 1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
        
    # Get the name of the model class
    model_name = model.__class__.__name__  
    plt.title(f"ROC Curves for {model_name} (OvR macro & micro avg)")
    plt.legend(loc = "lower right")

    return plt.gcf()


# Run SHAP analysis
def calculate_shap_values(model, 
                          X_train: pd.DataFrame
                          ) -> pd.DataFrame:
    """ Calculate SHAP values using the trained model and training data.
    """
    # Create SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    # Calculate SHAP values for test set (or subset for speed)
    shap_values = explainer.shap_values(X_train)
    return shap_values


def shap_plot_summary(
        shap_values: pd.DataFrame,
        X_test: pd.DataFrame
        ) -> plt.Figure:
    """ Plot a SHAP summary to visualize feature importance. 
    """
    plt.figure(figsize=(10, 6))
    shap.initjs()
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        class_names=[f'Obesity Level {i}' for i in range(7)],
        show=False
        )
    plt.title("SHAP Summary Bar Plot")
    plt.tight_layout()
    return plt.gcf()


def shap_sample_explanation(
        model,
        X_test: pd.DataFrame,
        sample_id: int = 0
        ) -> plt.Figure:
    """ Plot a bar plot for SHAP values to visualize feature impact. 
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    predicted_class = model.predict(X_test)[sample_id]
    prediction_proba = model.predict_proba(X_test)[sample_id]

    plt.figure(figsize=(10, 6))
    # shap.initjs()
    shap.plots.waterfall(
        shap_values[0, :, predicted_class],
        show=False
        )
    
    # Add in-plot text for prediction and probabilities
    plt.gcf().text(
        0.7, 0.25,
        f"Sample {sample_id} prediction: Class {predicted_class}\nClass probabilities: {np.round(prediction_proba, 2)}",
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )
    plt.title(f"SHAP Explanation for Sample {sample_id} - Predicted Class: {predicted_class} (Probability: {prediction_proba[predicted_class]:.2f})")
    plt.tight_layout()
    return plt.gcf()


# Save figure
def save_fig(
        fig: plt.Figure,
        path: str
        ) -> None:
    """Save a matplotlib Figure to a file

    :param fig: The matplotlib Figure object to save
    :param path: The file path where the figure will be saved
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def main():
    # Create a Unique Directory for Model Validation Output Files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print('Starting Model Validation...')
    MODEL_PATH = './models'
    OUTPUT_PATH = './data/model_validation_output/'
    output_dir = os.path.join(OUTPUT_PATH, f"{timestamp}")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory for output files: '{output_dir}/'")

    # Load model
    model = load_model(os.path.join(MODEL_PATH, MODEL_FILENAME))

    # Load test data
    data_dir = get_most_recent_datadir()
    X_train_path = os.path.join(data_dir, 'X_train_scaled.csv')
    y_train_path = os.path.join(data_dir, 'y_train_scaled.csv')
    X_test_path = os.path.join(data_dir, 'X_test_scaled.csv')
    y_test_path = os.path.join(data_dir, 'y_test_scaled.csv')
    X_train = load_data(X_train_path)
    y_train = load_data(y_train_path)
    X_test = load_data(X_test_path)
    y_test = load_data(y_test_path)

    # Make predictions
    y_pred_train = predict(model, X_train)
    y_pred_test = predict(model, X_test)

    # Calculate training and test accuracy
    train_accuracy = calculate_accuracy(y_train, y_pred_train)
    test_accuracy = calculate_accuracy(y_test, y_pred_test)

    # Generate classification report
    report = classification_report(y_test, y_pred_test)

    # Save accuracy scores and classification report to text file
    scoring_report_filename = 'model_output.txt'
    scoring_report_path = os.path.join(output_dir, scoring_report_filename)
    save_text_output(scoring_report_path, train_accuracy, test_accuracy, report)

    # Generate confusion matrix
    cm = generate_confusion_matrix(y_test, y_pred_test)

    # Display and save confusion matrix
    confusion_matrix_file_name = os.path.join(output_dir, 'confusion_matrix.png')
    confusion_matrix = plot_confusion_matrix(cm)
    save_fig(confusion_matrix, confusion_matrix_file_name)

    class_names = np.unique(y_test)

    # Plot and save ROC curve
    roc_curve_file_name = os.path.join(output_dir, 'roc_curve.png')
    roc_curve = plot_roc_curve(model, X_test, y_test, class_names)
    save_fig(roc_curve, roc_curve_file_name)
    
    # Run SHAP analysis
    shap_values = calculate_shap_values(model, X_test)

    shap_summary_file_name = os.path.join(output_dir, 'shap_summary.png')
    shap_summary = shap_plot_summary(shap_values, X_test)
    save_fig(shap_summary, shap_summary_file_name)

    shap_exp_file_name = os.path.join(output_dir, 'shap_explanation.png')
    shap_exp = shap_sample_explanation(model, X_test, sample_id=0)
    save_fig(shap_exp, shap_exp_file_name)

    print("Model validation completed successfully.")
    print(f"Results saved to: {output_dir}")

# Run the program
if __name__ == "__main__":
    main()