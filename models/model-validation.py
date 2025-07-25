import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

# Load Model
def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Make Predictions 
def predict(model, X_test):
    return model.predict(X_test)

def calculate_accuracy(y_true: pd.Series, 
                       y_pred: pd.Series
                       ) -> float:
    """Calculate and return accuracy score.
    """
    return accuracy_score(y_true.values.ravel(), y_pred)

def generate_classification_report(y_test: pd.Series, 
                                   y_pred: pd.Series
                                   ) -> str:
    """Generate classification report using sklearn.metrics.
    """
    return classification_report(y_test.values.ravel(), y_pred)

def save_text_output(train_accuracy: float,
                     test_accuracy: float,
                     output: str,
                     filename: str = 'model_output.txt'
                     ) -> None:

    """ Save accuracy scores and classification report to text file.
    """
    with open(filename, 'w') as f:
        f.write(f"Train Accuracy Score: {train_accuracy:.4f}\n\n")
        f.write(f"Test Accuracy Score: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(output)

def generate_confusion_matrix(y_test: pd.Series, 
                              y_pred: pd.Series
                              ) -> None:
    """Generates and returns a confusion matrix.
    """
    return confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(
        cm: pd.DataFrame, 
        labels: list,
        filename: str = 'confusion_matrix.png'
        ) -> plt.Figure:
    """ Plot the confusion matrix using sklearn.metrics
    """
    display = ConfusionMatrixDisplay(confusion_matrix = cm, 
                                     display_labels = labels)
    display.plot(cmap = 'coolwarm')
    plt.show()

def plot_roc_curve(model: object, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series, 
                   class_names: list,
                   filename: str = 'roc_curve.png'
                   ) -> plt.Figure:
    """
    Plot Receiver Operating Characteristic (ROC) curves for multi-class classification.

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

    plt.show()

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
    MODEL_PATH = './model-training/'
    OUTPUT_PATH = './data/model_output/'

    # Load model
    load_model(MODEL_PATH)

    # Make predictions
    y_pred = predict(model, X_test)

    # Create a Unique Directory for Output Files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_PATH, 'model_output', f"model_output_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory for output files: '{output_dir}/'")

    # Calculate training and test accuracy
    train_accuracy = calculate_accuracy(y_train, y_pred_train)
    test_accuracy = calculate_accuracy(y_test, y_pred_test)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Save accuracy scores and classification report to text file
    save_text_output(train_accuracy, test_accuracy, report, filename="model_results.txt")

    # Generate confusion matrix
    cm = generate_confusion_matrix(y_test, y_pred)

    # Display and save confusion matrix
    confusion_matrix = plot_confusion_matrix(cm, labels, filename = 'confusion_matrix.png')
    save_fig(confusion_matrix, OUTPUT_PATH)

    class_names = np.unique(y_test)

    # Plot and save ROC curve
    roc_curve = plot_roc_curve(model, X_test, y_test, class_names, filename = 'roc_curve.png')
    save_fig(roc_curve, OUTPUT_PATH)
    

# Run the program
if __name__ == "__main__":
    main()