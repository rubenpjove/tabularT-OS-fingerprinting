####################################################
##                   IMPORTS                      ##
####################################################

import warnings
import logging
import random
import sys
import absl
import absl.logging
import builtins
import time
import glob

import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

from icecream import ic

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)
####################################################
##                 DEFINITIONS                    ##
####################################################

seed = 42
random.seed(seed)
np.random.seed(seed)

def my_show_df_shape(df, target=None):
    print("Number of examples/rows: {:,}".format(df.shape[0]))
    print("Number of features: {:,}".format(df.shape[1]))
    if target is not None:
        print(f"Number of unique values in {target} column: {df[target].nunique()}")

def my_show_df_NaNs(df):
    nan_count = df.isna().sum()
    # nan_count = nan_count[nan_count > 0]
    nan_percentage = df.isna().mean() * 100
    # nan_percentage = nan_percentage[nan_percentage > 0]
    length = 0
    for i in nan_count.index: 
        if len(i) > length:
            length = len(i) 
    if nan_count.empty:
        print("No NaN values found in the dataset.")
    else:
        print("Number of examples/rows:", df.shape[0])
        print("Number of features:", df.shape[1])
        print("Features with NaNs:")
        for i in range(len(nan_count)):
            feature = nan_count.index[i]
            if nan_count[feature] > 0:
                print(f"{feature:<{length+5}} {str(df[feature].dtype):<8}     {nan_count[feature]:<7}({nan_percentage[feature]:<.4f} %)")
            else:
                print(f"{feature:<{length+5}} {str(df[feature].dtype):<8}     no NaNs")

####################################################
##                   ARGUMENTS                    ##
####################################################

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

with open(sys.argv[1], 'r') as file:
    config = yaml.safe_load(file)

# classes_detail = "family"
classes_detail = config['classes_detail']
assert classes_detail in ["family", "major", "minor"], "Invalid classes_detail. Must be 'family', 'major', or 'minor'."

# dataset_path = './data/lastovicka_2023_passiveOSRevisited.csv'
dataset_path = config['dataset_path']
assert dataset_path.endswith(".csv"), "Invalid dataset_path. Must be a .csv file."

####################################################
##                 DATASET LOAD                   ##
####################################################

filenames = glob.glob(dataset_path)

p = 1  # 100% of the lines - All rows
# p = 0.01  # 10% of the lines
# p = 0.0025  # 0,25% of the lines
dfs = []

if p < 1:
    for file in filenames:
        df = pd.read_csv(file, header=0, sep=',', skiprows=lambda i: i>0 and random.random() > p)
        dfs.append(df)
else:
    for file in filenames:
        df = pd.read_csv(file, 
                        header=0, 
                        sep=',')
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

####################################################
##                 PREPROCESSING                  ##
####################################################

target = 'Ground Truth OS'
# targets = ['UA OS family', 'UA OS major', 'UA OS minor']

# Drop rows with NaN values in target column
df.dropna(subset=[target], inplace=True)

if classes_detail == "family":
    # Undersampling to balance the classes
    df = df.groupby(target).apply(lambda x: x.sample(n=df[target].value_counts().min(), random_state=seed)).reset_index(drop=True)
elif classes_detail == "major":
    pass
elif classes_detail == "minor":
    pass


# Drop not needed columns
drop_columns = [
    # Not useful
    "Date flow start",
    "Date flow end",
    "Src IPv4",
    "Dst IPv4",
    "sPort",
    "dPort",
    "TLS SNI",
    "TLS SNI length",
    "Session ID"
]
df.drop(columns=drop_columns, inplace=True)
df.drop(columns=df.filter(regex="^SSH.*").columns, inplace=True)
df.drop(columns=df.filter(regex="^HTTP.*").columns, inplace=True)

df.dropna(inplace=True)

# Drop columns with constant values
numeric_columns = df.select_dtypes(include=['float', 'int'])
low_variance_columns = numeric_columns.columns[numeric_columns.var() <= 0]
df.drop(columns=low_variance_columns, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

####################################################
##                 DATA PREPARATION               ##
####################################################

# Label
LABEL = target

# Categorical features
CATEGORICAL_FEATURES = [
    "TLS Client Version", "Client Cipher Suites", 
    "TLS Extension Types", "TLS Extension Lengths",
    "TLS Elliptic Curves", "TLS EC Point Formats" 
]

# Numerical features
NUMERICAL_FEATURES = [ "SYN size", "TCP win", "TCP SYN TTL" ]

FEATURES = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)

# for feature in FEATURES:
#     print(df[feature].value_counts())

# Round to the higher power of two the values of column "TCP SYN TTL"
df["TCP SYN TTL"] = df["TCP SYN TTL"].apply(lambda x: 2**np.ceil(np.log2(x)))

# Encode categorical features
def split_hex_string_into_bytes(hex_str):
    return [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

for column in ["Client Cipher Suites", "TLS Extension Types", "TLS Extension Lengths", "TLS Elliptic Curves"]:
    split_columns = df[column].apply(split_hex_string_into_bytes).apply(pd.Series)
    split_columns.columns = [f'{column}_byte_{i}' for i in range(split_columns.shape[1])]
    CATEGORICAL_FEATURES.remove(column)
    CATEGORICAL_FEATURES.extend(split_columns.columns)
    df.drop(columns=[column], inplace=True)
    df = pd.concat([df, split_columns], axis=1)

FEATURES = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)

# One-hot encode labels
onehot_encoder = OneHotEncoder()
label_pipe = Pipeline([
    ('onehotencoder', onehot_encoder),
])
y = label_pipe.fit_transform(df[[LABEL]]).toarray()
df.drop(columns=[LABEL], inplace=True)
LABEL = [f"Label_{i}" for i in range(y.shape[1])]
y = pd.DataFrame(y, columns=LABEL)
df = pd.concat([df, y], axis=1)


class CustomCategoricalEncoderImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dtypes_ = {}

    def fit(self, X, y=None):
        self.dtypes_ = X.dtypes.to_dict()
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for column in X.columns:
            if self.dtypes_[column] == 'object':
                X_col_transformed = X[column].fillna("missing")
            elif self.dtypes_[column] in ['float64', 'int64']:
                X_col_transformed = X[column].apply(lambda x: str(int(np.floor(x))) if not np.isnan(x) else "missing")
            else:
                raise ValueError("Not supported dtype. Must be 'object', 'float', or 'int'.")
        
            unique_categories = sorted(X_col_transformed.dropna().unique())
            map_category_to_int = {category: idx for idx, category in enumerate(unique_categories)}

            X_transformed[column] = X_col_transformed.map(map_category_to_int)
            
        return X_transformed.astype(int)

cat_pipeline = Pipeline(steps=[
    ('preprocessor', CustomCategoricalEncoderImputer()),
])
df[CATEGORICAL_FEATURES] = cat_pipeline.fit_transform(df[CATEGORICAL_FEATURES])

####################################################
##                   TRAINNING                    ##
####################################################

y = df[LABEL]
X = df
X.drop(columns=LABEL, inplace=True)

# print(pd.DataFrame(label_pipe.inverse_transform(y).reshape(-1, 1), columns=[LABEL[0]]).value_counts())
# my_show_df_shape(X)

# Define the models
models = {
    'kNN': KNeighborsClassifier(),
    'RF': RandomForestClassifier(random_state=seed),
    'MLP': MLPClassifier(random_state=seed),
}

# Print default hyperparameters of each model
for name, model in models.items():
    print(f"{name}: {model.get_params()}")

# Initialize a dictionary to store the average F1 scores
avg_scores = {}

print()
print()

# Repeat the training and evaluation 10 times
for name, model in models.items():
    scores = []

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    for fold, (train_index, test_index) in enumerate(skf.split(X, np.argmax(y.values, axis=1))):
        X_train = X.loc[train_index].reset_index(drop=True)
        y_train = y.loc[train_index].reset_index(drop=True)
        X_test = X.loc[test_index].reset_index(drop=True)
        y_test = y.loc[test_index].reset_index(drop=True)

        # Starnadize numerical features
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])
        num_pipeline.fit(X_train[NUMERICAL_FEATURES])
        X_train[NUMERICAL_FEATURES] = num_pipeline.transform(X_train[NUMERICAL_FEATURES])
        X_test[NUMERICAL_FEATURES] = num_pipeline.transform(X_test[NUMERICAL_FEATURES])

        # SMOTE oversampling
        # smote = SMOTE(sampling_strategy='auto', random_state=seed)
        # X_resampled, y_resampled = smote.fit_resample(X_train, y_train.values.argmax(axis=1))
        # X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
        # id_matrix = np.eye(y_train.shape[1])
        # y_train = pd.DataFrame(id_matrix[y_resampled], columns=LABEL)
        
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Convert one-hot encoded labels to integers
        y_true = y_test.idxmax(axis=1)
        y_pred = pd.DataFrame(y_pred, columns=LABEL).idxmax(axis=1)
        
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

        # Print model name and F1 micro score
        print(f"{name}: {micro_f1_score:.4f}")

        # Append results of all metrics to the scores list
        scores.append({
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1_score': macro_f1_score,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1_score': micro_f1_score
        })
    
    # Calculate the average of all the scores
    avg_scores[name] = {
        'accuracy': np.mean([score['accuracy'] for score in scores]),
        'balanced_accuracy': np.mean([score['balanced_accuracy'] for score in scores]),
        'precision': np.mean([score['precision'] for score in scores]),
        'recall': np.mean([score['recall'] for score in scores]),
        'f1_score': np.mean([score['f1_score'] for score in scores]),
        'macro_precision': np.mean([score['macro_precision'] for score in scores]),
        'macro_recall': np.mean([score['macro_recall'] for score in scores]),
        'macro_f1_score': np.mean([score['macro_f1_score'] for score in scores]),
        'micro_precision': np.mean([score['micro_precision'] for score in scores]),
        'micro_recall': np.mean([score['micro_recall'] for score in scores]),
        'micro_f1_score': np.mean([score['micro_f1_score'] for score in scores])
    }

print()
print()

# Print the average of scores for each model
for name, scores in avg_scores.items():
    for metric, value in scores.items():
        print(f"{name} {metric}: {value:.4f}")