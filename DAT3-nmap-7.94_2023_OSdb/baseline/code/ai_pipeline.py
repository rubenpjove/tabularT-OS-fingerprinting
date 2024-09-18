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
assert classes_detail in ["family", "minor"], "Invalid classes_detail. Must be 'family' or 'minor'."

# dataset_path = './data/lastovicka_2023_passiveOSRevisited.csv'
dataset_path = config['dataset_path']
assert dataset_path.endswith(".csv"), "Invalid dataset_path. Must be a .csv file."


####################################################
##                 DATASET LOAD                   ##
####################################################

df = pd.read_csv(
         dataset_path,
         header=0,
         sep=','
)


####################################################
##                 PREPROCESSING                  ##
####################################################

target = 'Class.OSfamily_0'
targets = ['Class.vendor_0','Class.OSfamily_0', 'Class.OSgen_0','Class.device_0']

# for target in targets:
#     print(df[target].value_counts())

# Drop rows with NaN values in target column
df.dropna(subset=[target], inplace=True)

if classes_detail == "family":
    df['OS target'] = df[targets[1]].astype(str)
    df.drop(targets, axis=1, inplace=True)
    target = 'OS target'
elif classes_detail == "minor":
    df['OS target'] = df[targets[1]] + " - " + df[targets[2]].astype(str)
    df.drop(targets, axis=1, inplace=True)
    target = 'OS target'

# # Drop not needed columns
# drop_columns = [
#     # All NaN values
#     "flow_ID",
#     "L3 PROTO",
#     "L4 PROTO",
#     "UA OS patch",
#     "UA OS patch minor",
#     "ICMP TYPE",
#     "TLS_ALPN",
#     "TLS_ISSUER_CN",
#     "TLS_SUBJECT_CN",
#     "TLS_SUBJECT_ON",
#     "Unnamed: 111",
#     # Not useful
#     "DST port",
#     "BYTES A",
#     "PACKETS A",
#     "start",
#     "end",
#     "SRC IP",
#     "DST IP",
#     "HTTP Request Host",
#     "URL",
#     "maximumTTLforward",
#     "tcpTimestampFirstPacketforward",
#     "tcpTimestampFirstPacketbackward",
#     "flowDirection",
#     "packetTotalCountforward",
#     "packetTotalCountbackward",
#     "TLS_CLIENT_RANDOM",
# ]
# df.drop(columns=drop_columns, inplace=True)
# df.drop(columns=df.filter(regex="^NPM.*(SERVER|\_B)").columns, inplace=True)
# df.drop(columns=df.filter(regex="^TLS.*(SERVER|\_B|SNI)").columns, inplace=True)
# df.drop(columns=df.filter(regex="^tcp.*(backward)").columns, inplace=True)

# Drop rows with NaN values in selected columns
# drop_nans_rows = [
#     "TCP SYN Size",
#     "TCP Win Size"
# ]
# df.dropna(subset=drop_nans_rows, 
#           inplace=True)

# Drop columns with constant values
numeric_columns = df.select_dtypes(include=['float', 'int'])
low_variance_columns = numeric_columns.columns[numeric_columns.var() <= 0]
df.drop(columns=low_variance_columns, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

# Modify target classes
if classes_detail == "family":
    # Random undersampling
    class_examples = df[df[target] == "Linux"]
    num_examples_to_remove = int(0.75 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    class_examples = df[df[target] == "Windows"]
    num_examples_to_remove = int(0.51 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    class_examples = df[df[target] == "macOS"]
    num_examples_to_remove = int(0.19 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    class_examples = df[df[target] == "iOS"]
    num_examples_to_remove = int(0.04 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)
elif classes_detail == "minor":
    # Random undersampling
    class_examples = df[df[target] == "Linux - 2.6.X"]
    num_examples_to_remove = int(0.8 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    class_examples = df[df[target] == "Windows - XP"]
    num_examples_to_remove = int(0.5 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    class_examples = df[df[target] == "Linux - 3.X"]
    num_examples_to_remove = int(0.45 * len(class_examples))
    class_examples_to_remove = class_examples.sample(num_examples_to_remove, random_state=seed)
    df = df.drop(class_examples_to_remove.index)

    # Combine classes with less than <threshold> examples
    # threshold = 0.01 * df.shape[0]
    threshold = 200
    value_counts = df[target].value_counts()
    to_combine = value_counts[value_counts <= threshold].sort_values(ascending=True)
    rest = value_counts.index
    no_classes = []
    for os_class, count in to_combine.items():
        os_class_splitted = os_class.split(" - ")
        family = os_class_splitted[0] + " - nan"
        if family in rest:
            df.loc[df[target] == os_class, target] = family
        else:
            df.loc[df[target] == os_class, target] = family
            no_classes.append(os_class)

df.reset_index(drop=True, inplace=True)

print(df[target].value_counts())
# df[target].value_counts().to_excel('value_counts.xlsx')

my_show_df_shape(df, target)

####################################################
##                 DATA PREPARATION               ##
####################################################

# Label
LABEL = target

# Categorical features
CATEGORICAL_FEATURES = df.select_dtypes(include=['object']).columns.to_list()
CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f != LABEL]

# Numerical features
NUMERICAL_FEATURES = df.select_dtypes(include=['float', 'int']).columns.to_list()

# print(f"Categorical features:{len(CATEGORICAL_FEATURES)}")
# print(CATEGORICAL_FEATURES)
# print(f"\nNumerical features:{len(NUMERICAL_FEATURES)}")
# print(NUMERICAL_FEATURES)

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

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
                X_col_transformed = X[column].fillna("NO")
            elif self.dtypes_[column] in ['float64', 'int64']:
                X_col_transformed = X[column].apply(lambda x: str(int(np.floor(x))) if not np.isnan(x) else "NO")
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


# print(df[CATEGORICAL_FEATURES].nunique())

# One hot encode categorical features
onehot_encoder = OneHotEncoder()
cat_pipe = Pipeline([
    ('onehotencoder', onehot_encoder),
])
encoded_features = cat_pipe.fit_transform(df[CATEGORICAL_FEATURES]).toarray()
encoded_columns = cat_pipe.named_steps['onehotencoder'].get_feature_names_out(CATEGORICAL_FEATURES)
df_encoded = pd.DataFrame(encoded_features, columns=encoded_columns)
df.drop(columns=CATEGORICAL_FEATURES, inplace=True)
df = pd.concat([df, df_encoded], axis=1)

# my_show_df_shape(df)

####################################################
##                   TRAINNING                    ##
####################################################

y = df[LABEL]
X = df
X.drop(columns=LABEL, inplace=True)

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
        smote = SMOTE(sampling_strategy='auto', random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train.values.argmax(axis=1))

        X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
        id_matrix = np.eye(y_train.shape[1])
        y_train = pd.DataFrame(id_matrix[y_resampled], columns=LABEL)
        
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