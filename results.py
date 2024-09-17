import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import sys

# Function to load and evaluate predictions from CSV file
def evaluate_multiclass_classification(path):
    # Load the CSV file
    df = pd.read_csv(path+'results.csv')
    # Read labels from file to an array
    with open(path+'labels.txt', 'r') as file:
        labels = file.read().splitlines()

    # Extract predictions and ground truth
    y_pred = df['Predictions']
    y_true = df['Ground Truth']

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # Calculate balanced accuracy
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)

    # Calculate weighted precision, recall, and F1-score
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')

    # Calculate macro-averaged precision, recall, and F1-score
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    macro_f1_score = metrics.f1_score(y_true, y_pred, average='macro')

    # Calculate micro-averaged precision, recall, and F1-score
    micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
    micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
    micro_f1_score = metrics.f1_score(y_true, y_pred, average='micro')

    # Calculate confusion matrix
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
    print('')
    print(f'Precision (weighted): {precision:.4f}')
    print(f'Recall (weighted): {recall:.4f}')
    print(f'F1-score (weighted): {f1_score:.4f}')
    print('')
    print(f'Precision (macro): {macro_precision:.4f}')
    print(f'Recall (macro): {macro_recall:.4f}')
    print(f'F1-score (macro): {macro_f1_score:.4f}')
    print('')
    print(f'Precision (micro): {micro_precision:.4f}')
    print(f'Recall (micro): {micro_recall:.4f}')
    print(f'F1-score (micro): {micro_f1_score:.4f}')

    def custom_annot(val):
        return '' if val == 0 else val

    # Plot the confusion matrix
    if "family" in path:
        plt.figure(figsize=(8, 8))
    elif "major" in path:
        plt.figure(figsize=(11, 11))
    elif "minor" in path:
        plt.figure(figsize=(15, 10))
    else:
        raise ValueError("Invalid type of dataset")

    # Plot the second subplot: standardized confusion matrix
    plt.subplot(1, 1, 1)
    df_conf_matrix = pd.DataFrame(conf_matrix)
    sns.heatmap(df_conf_matrix, annot=df_conf_matrix.map(custom_annot), fmt='', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    # sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Standardized Confusion Matrix')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(path+'..\\confusion-matrix.png')

    def custom_annot(val):
        return '' if val < 0.001 else f'{val*100:.1f}'


    # Plot the confusion matrix
    if "family" in path:
        plt.figure(figsize=(8, 8))
    elif "major" in path:
        plt.figure(figsize=(11, 11))
    elif "minor" in path:
        plt.figure(figsize=(15, 10))
    else:
        raise ValueError("Invalid type of dataset")

    # Standardize the confusion matrix
    standardized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    # Plot the second subplot: standardized confusion matrix
    plt.subplot(1, 1, 1)
    df_std_conf_matrix = pd.DataFrame(standardized_conf_matrix)
    sns.heatmap(df_std_conf_matrix, annot=df_std_conf_matrix.map(custom_annot), fmt='', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    # sns.heatmap(standardized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Standardized Confusion Matrix')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(path+'..\\confusion-matrix-std.png')

# Example usage:
if __name__ == "__main__":
    path = '.\\nmap\\results\\results-nmap-tabt-minor\\results\\'  # Replace with your CSV file path
    sys.stdout = open(path+'..\\metrics.txt', 'w')
    evaluate_multiclass_classification(path)
    sys.stdout.close()
