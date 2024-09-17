####################################################
##                   IMPORTS                      ##
####################################################

import warnings
import logging
import random
import sys
import os
import absl
import absl.logging
import builtins
import gc
import time

import yaml

import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tab_transformer_pytorch import TabTransformer, FTTransformer
from imblearn.over_sampling import SMOTE

from icecream import ic

logging.captureWarnings(True)
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.INFO)

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

def my_clean_gpu_memory():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

####################################################
##                   ARGUMENTS                    ##
####################################################

def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(*args, **kwargs)

with open(sys.argv[1], 'r') as file:
    config = yaml.safe_load(file)

# transformer_type = "ftt"
transformer_type = config['transformer_type']
assert transformer_type in ["tabt", "ftt"], "Invalid transformer_type. Must be 'tabt' or 'ftt'."

# classes_detail = "family"
classes_detail = config['classes_detail']
assert classes_detail in ["family", "minor"], "Invalid classes_detail. Must be 'family' or 'minor'."

# dataset_path = './data/lastovicka_2023_passiveOSRevisited.csv'
dataset_path = config['dataset_path']
assert dataset_path.endswith(".csv"), "Invalid dataset_path. Must be a .csv file."

# output_path = './'
output_path = os.path.dirname(os.path.abspath(__file__)) + "/results/"

# Torch settings
# num_threads = 32
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optuna settings
optuna_trials = config['optuna_trials']
optuna_jobs = config['optuna_jobs']
optuna_verbose = True

# Training settings
CV_folds = config['CV_folds']
epochs = config['epochs']
early_stopping_rounds = config['early_stopping_rounds']

batch_size_hyperparameter = config['batch_size']

learning_rate_hyperparameter_min = config['learning_rate_min']
learning_rate_hyperparameter_max = config['learning_rate_max']
embedding_dim_hyperparameter = config['embedding_dim']
depth_hyperparameter_min = config['depth_min']
depth_hyperparameter_max = config['depth_max']
heads_hyperparameter_min = config['heads_min']
heads_hyperparameter_max = config['heads_max']
heads_hyperparameter_step = config['heads_step']
attn_dropout_hyperparameter_min = config['attn_dropout_min']
attn_dropout_hyperparameter_max = config['attn_dropout_max']
attn_dropout_hyperparameter_step = config['attn_dropout_step']
ff_dropout_hyperparameter_min = config['ff_dropout_min']
ff_dropout_hyperparameter_max = config['ff_dropout_max']
ff_dropout_hyperparameter_step = config['ff_dropout_step']
use_shared_categ_embed_hyperparameter = config['use_shared_categ_embed']
# mlp_act_hyperparameter = config['mlp_act']
# mlp_hidden_mults_hyp_min_units = config['mlp_hidden_mults_min_units']
# mlp_hidden_mults_hyp_max_units = config['mlp_hidden_mults_max_units']
# mlp_hidden_mults_hyp_min_layers = config['mlp_hidden_mults_min_layers']
# mlp_hidden_mults_hyp_max_layers = config['mlp_hidden_mults_max_layers']

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

# Counts of features
categories_sum = []
for column in df[CATEGORICAL_FEATURES].columns:
    categories_sum.append(len(df[column].unique()))

# print(f"Number of unique categories: {categories_sum}")
# print(f"Total number of unique categories: {sum(categories_sum)}")
# print(categories_sum)

# Calculate class weights
class_weights = torch.tensor(1.0 / df[LABEL].value_counts(normalize=True).sort_index().values, dtype=torch.float)

# Split data
df_train, df_test = train_test_split(df, stratify=df[LABEL], test_size=0.20, random_state=seed)

df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y = df_train[LABEL].copy()
X = df_train.copy()
X.drop(columns=LABEL, inplace=True)

# SMOTE oversampling
smote = SMOTE(sampling_strategy='auto', random_state=seed)
X_resampled, y_resampled = smote.fit_resample(X, y.values.argmax(axis=1))

X = pd.DataFrame(X_resampled, columns=X.columns)
id_matrix = np.eye(y.shape[1])
y = pd.DataFrame(id_matrix[y_resampled], columns=LABEL)

print(pd.DataFrame(label_pipe.inverse_transform(y).reshape(-1, 1), columns=[LABEL[0]]).value_counts())
my_show_df_shape(X)

####################################################
##                   TRAINNING                    ##
####################################################

def preprocess(trial):
    if trial is None:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
    else:
        imputer_strategy = trial.suggest_categorical('imputer_strategy', ['mean', 'median', 'most_frequent'])
        imputer = SimpleImputer(strategy=imputer_strategy)
        
        scaler_name = trial.suggest_categorical('scaler', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
        if scaler_name == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_name == 'RobustScaler':
            scaler = RobustScaler()

    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler)
    ])

    return numeric_transformer


NUM_EPOCHS = epochs


class CatNumDataset(Dataset):
    def __init__(self, cat, num, y):
        self.cat = cat
        self.num = num
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        cat = self.cat[idx]
        num = self.num[idx]
        label = self.y[idx]
        return cat, num, label

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for batch in self.dl:
            yield [b.to(self.device) for b in batch]
    
    def __len__(self):
        return len(self.dl)


def objective(trial):
    global X, y, cat_feature_counts, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, class_weights, CV_folds, batch_size, transformer_type, seed, categories_sum, early_stopping_rounds

    # TabTransformer Hyperparameters
    learning_rate_hyp = trial.suggest_float("learning_rate", learning_rate_hyperparameter_min, learning_rate_hyperparameter_max, log=True)
    embedding_dim_hyp = trial.suggest_categorical('embedding_dim', embedding_dim_hyperparameter)
    depth_hyp = trial.suggest_int("depth", depth_hyperparameter_min, depth_hyperparameter_max)
    heads_hyp = trial.suggest_int("heads", heads_hyperparameter_min, heads_hyperparameter_max, step=heads_hyperparameter_step)
    attn_dropout_hyp = trial.suggest_float("attn_dropout", attn_dropout_hyperparameter_min, attn_dropout_hyperparameter_max, step=attn_dropout_hyperparameter_step)
    ff_dropout_hyp = trial.suggest_float("ff_dropout", ff_dropout_hyperparameter_min, ff_dropout_hyperparameter_max, step=ff_dropout_hyperparameter_step)
    use_shared_categ_embed_hyp = trial.suggest_categorical("use_shared_categ_embed", use_shared_categ_embed_hyperparameter)
    # mlp_hidden_mults_hyp = tuple(trial.suggest_int(f'n_units_l{i}', mlp_hidden_mults_hyp_min_units, mlp_hidden_mults_hyp_max_units) for i in range(trial.suggest_int('n_layers', mlp_hidden_mults_hyp_min_layers, mlp_hidden_mults_hyp_max_layers)))
    # mlp_act_hyp = trial.suggest_categorical('mlp_act', mlp_act_hyperparameter)

    # if mlp_act_hyp == "relu":
    #     mlp_act_hyp = torch.nn.ReLU()
    # elif mlp_act_hyp == "tanh":
    #     mlp_act_hyp = torch.nn.Tanh()
    # elif mlp_act_hyp == "leakyrelu":
    #     mlp_act_hyp = torch.nn.LeakyReLU()

    # TabTransformer Model
    if transformer_type == "tabt":
        model = TabTransformer(
            categories = categories_sum,
            num_continuous = len(NUMERICAL_FEATURES),
            dim = embedding_dim_hyp,
            dim_out = y.shape[1],
            depth = depth_hyp,
            heads = heads_hyp,
            attn_dropout = attn_dropout_hyp,
            ff_dropout = ff_dropout_hyp,
            # mlp_hidden_mults = mlp_hidden_mults_hyp,
            # mlp_act = mlp_act_hyp,
            use_shared_categ_embed = use_shared_categ_embed_hyp,
        )
    elif transformer_type == "ftt":
        model = FTTransformer(
            categories = categories_sum,
            num_continuous = len(NUMERICAL_FEATURES),
            dim = embedding_dim_hyp,
            dim_out = y.shape[1],
            depth = depth_hyp,
            heads = heads_hyp,
            attn_dropout = attn_dropout_hyp,
            ff_dropout = ff_dropout_hyp
        )
    else:
        raise ValueError("Invalid transformer_type. Must be 'tabt' or 'ftt'.")

    batch_size = trial.suggest_categorical("batch_size", batch_size_hyperparameter)

    # Training settings
    LEARNING_RATE = learning_rate_hyp
    optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE
        )
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    skf = StratifiedKFold(n_splits=CV_folds, shuffle=True, random_state=seed)

    results = []

    total_training_time = 0

    for fold, (train_index, val_index) in enumerate(skf.split(X, np.argmax(y.values, axis=1))):
        X_train = X.loc[train_index].reset_index(drop=True)
        y_train = y.loc[train_index].reset_index(drop=True)
        X_val = X.loc[val_index].reset_index(drop=True)
        y_val = y.loc[val_index].reset_index(drop=True) 

        num_preprocessor = preprocess(trial)
        num_preprocessor.fit_transform(X_train[NUMERICAL_FEATURES])
        X_train[NUMERICAL_FEATURES] = num_preprocessor.transform(X_train[NUMERICAL_FEATURES])
        X_val[NUMERICAL_FEATURES] = num_preprocessor.transform(X_val[NUMERICAL_FEATURES])

        X_train_tensor_cat = torch.tensor(X_train[CATEGORICAL_FEATURES].values, dtype=torch.int32)
        X_train_tensor_num = torch.tensor(X_train[NUMERICAL_FEATURES].values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor_cat = torch.tensor(X_val[CATEGORICAL_FEATURES].values, dtype=torch.int32)
        X_val_tensor_num = torch.tensor(X_val[NUMERICAL_FEATURES].values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        del X_train, y_train, X_val, y_val
        my_clean_gpu_memory()

        train_dataset = CatNumDataset(X_train_tensor_cat, X_train_tensor_num, y_train_tensor)
        val_dataset = CatNumDataset(X_val_tensor_cat, X_val_tensor_num, y_val_tensor)

        del X_train_tensor_cat, X_train_tensor_num, y_train_tensor, X_val_tensor_cat, X_val_tensor_num, y_val_tensor
        my_clean_gpu_memory()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        del train_dataset, val_dataset
        my_clean_gpu_memory()

        train_loader = DeviceDataLoader(train_loader, device)
        val_loader = DeviceDataLoader(val_loader, device)

        gc.collect()

        model.to(device)

        print(f"FOLD {fold+1}/{CV_folds}")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")

        # Early stopping parameters
        early_stopping_rounds = early_stopping_rounds
        no_improvement_count = 0
        best_val_result = -np.inf

        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            model.train()
            for X_cat_train_minib, X_num_train_minib, y_train_minib in train_loader:
                optimizer.zero_grad()
                outputs_train_minib = model(X_cat_train_minib, X_num_train_minib)
                loss = loss_fn(outputs_train_minib, y_train_minib)
                loss.backward()
                optimizer.step()

                del X_cat_train_minib, X_num_train_minib, y_train_minib
                my_clean_gpu_memory()
            
            model.eval()
            all_targets = []
            all_probs = []
            with torch.no_grad():
                for X_cat_val_minib, X_num_val_minib, y_val_minib in val_loader:
                    outputs_val_minib = model(X_cat_val_minib, X_num_val_minib)
                    loss = loss_fn(outputs_val_minib, y_val_minib)

                    probabilities_val_minib = torch.softmax(outputs_val_minib, dim=1)
                    all_probs.append(probabilities_val_minib.cpu().numpy())
                    all_targets.append(y_val_minib.cpu().numpy())

                    del X_cat_val_minib, X_num_val_minib, y_val_minib
                    my_clean_gpu_memory()

            all_probs = np.concatenate(all_probs, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # Calculate F1 score (micro)
            val_result = f1_score(np.argmax(all_targets, axis=1), np.argmax(all_probs, axis=1), average='micro')

            trial.report(val_result, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping logic
            if val_result > best_val_result:
                best_val_result = val_result
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= early_stopping_rounds:
                break  # Stop training if no improvement for `early_stopping_rounds` epochs

            # Print progress
            # if epoch % 10 == 0 and optuna_verbose:
            #     print(f"Fold {fold+1}/{CV_folds}, Epoch {epoch+1}/{NUM_EPOCHS}, Val: {val_result:.4f}, Best Val: {best_val_result:.4f}")
        
        results.append(val_result)

        gc.collect()

    end_time = time.time()

    # Calculate training time
    total_training_time += end_time - start_time

    # Measure inference time
    model.eval()
    inference_start_time = time.time()
    with torch.no_grad():
        for X_cat_val_minib, X_num_val_minib, y_val_minib in val_loader:
            outputs_val_minib = model(X_cat_val_minib, X_num_val_minib)
    inference_time = time.time() - inference_start_time

    # Calculate number of parameters
    num_parameters = sum(p.numel() for p in model.parameters())
    # Calculate model memory size
    model_memory_size = sum(param.numel() * param.element_size() for param in model.parameters()) + sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())

    # Set user attributes
    trial.set_user_attr("training_time", total_training_time)
    trial.set_user_attr("inference_time", inference_time)
    trial.set_user_attr("num_parameters", num_parameters)
    trial.set_user_attr("model_memory_size", model_memory_size)

    del model, optimizer, loss_fn, train_loader, val_loader, all_targets, all_probs
    my_clean_gpu_memory()
    
    return np.mean(results)


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.NSGAIISampler(seed=seed))
study.optimize(objective, n_trials=optuna_trials, n_jobs=optuna_jobs)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))


####################################################
##                   EVALUATION                   ##
####################################################

# TabTransformer Hyperparameters
learning_rate_hyp = trial.params['learning_rate']
embedding_dim_hyp = trial.params['embedding_dim']
depth_hyp = trial.params['depth']
heads_hyp = trial.params['heads']
attn_dropout_hyp = trial.params['attn_dropout']
ff_dropout_hyp = trial.params['ff_dropout']
use_shared_categ_embed_hyp = trial.params['use_shared_categ_embed']
# mlp_hidden_mults_hyp = tuple(trial.params[f'n_units_l{i}'] for i in range(trial.params['n_layers']))
# mlp_act_hyp = trial.params['mlp_act']

# if mlp_act_hyp == "relu":
#     mlp_act_hyp = torch.nn.ReLU()
# elif mlp_act_hyp == "tanh":
#     mlp_act_hyp = torch.nn.Tanh()
# elif mlp_act_hyp == "leakyrelu":
#     mlp_act_hyp = torch.nn.LeakyReLU()

# TabTransformer Model
if transformer_type == "tabt":
    model = TabTransformer(
        categories = categories_sum,
        num_continuous = len(NUMERICAL_FEATURES),
        dim = embedding_dim_hyp,
        dim_out = y.shape[1],
        depth = depth_hyp,
        heads = heads_hyp,
        attn_dropout = attn_dropout_hyp,
        ff_dropout = ff_dropout_hyp,
        # mlp_hidden_mults = mlp_hidden_mults_hyp,
        # mlp_act = mlp_act_hyp,
        use_shared_categ_embed = use_shared_categ_embed_hyp,
    )
elif transformer_type == "ftt":
    model = FTTransformer(
        categories = categories_sum,
        num_continuous = len(NUMERICAL_FEATURES),
        dim = embedding_dim_hyp,
        dim_out = y.shape[1],
        depth = depth_hyp,
        heads = heads_hyp,
        attn_dropout = attn_dropout_hyp,
        ff_dropout = ff_dropout_hyp
    )
else:
    raise ValueError("Invalid transformer_type. Must be 'tabt' or 'ftt'.")


# Training settings
LEARNING_RATE = learning_rate_hyp
optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

results = []

total_training_time = 0

# Split data
train_index, val_index = train_test_split(X.index, stratify=y, test_size=0.20, random_state=seed)

X_train = X.loc[train_index].reset_index(drop=True)
y_train = y.loc[train_index].reset_index(drop=True)
X_val = X.loc[val_index].reset_index(drop=True)
y_val = y.loc[val_index].reset_index(drop=True) 

num_preprocessor = preprocess(trial)
num_preprocessor.fit_transform(X_train[NUMERICAL_FEATURES])
X_train[NUMERICAL_FEATURES] = num_preprocessor.transform(X_train[NUMERICAL_FEATURES])
X_val[NUMERICAL_FEATURES] = num_preprocessor.transform(X_val[NUMERICAL_FEATURES])

gc.collect()

X_train_tensor_cat = torch.tensor(X_train[CATEGORICAL_FEATURES].values, dtype=torch.int32)
X_train_tensor_num = torch.tensor(X_train[NUMERICAL_FEATURES].values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor_cat = torch.tensor(X_val[CATEGORICAL_FEATURES].values, dtype=torch.int32)
X_val_tensor_num = torch.tensor(X_val[NUMERICAL_FEATURES].values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

train_dataset = CatNumDataset(X_train_tensor_cat, X_train_tensor_num, y_train_tensor)
val_dataset = CatNumDataset(X_val_tensor_cat, X_val_tensor_num, y_val_tensor)

batch_size = trial.params['batch_size']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

gc.collect()

model.to(device)

# Early stopping parameters
early_stopping_rounds = early_stopping_rounds
no_improvement_count = 0
best_val_result = -np.inf

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    for X_cat_train_minib, X_num_train_minib, y_train_minib in train_loader:
        optimizer.zero_grad()
        outputs_train_minib = model(X_cat_train_minib, X_num_train_minib)
        loss = loss_fn(outputs_train_minib, y_train_minib)
        loss.backward()
        optimizer.step()
    
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for X_cat_val_minib, X_num_val_minib, y_val_minib in val_loader:
            outputs_val_minib = model(X_cat_val_minib, X_num_val_minib)
            loss = loss_fn(outputs_val_minib, y_val_minib)

            probabilities_val_minib = torch.softmax(outputs_val_minib, dim=1)
            all_probs.append(probabilities_val_minib.cpu().numpy())
            all_targets.append(y_val_minib.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_result = f1_score(np.argmax(all_targets, axis=1), np.argmax(all_probs, axis=1), average='micro')
    
    # Early stopping logic
    if val_result > best_val_result:
        best_val_result = val_result
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    
    if no_improvement_count >= early_stopping_rounds:
        break  # Stop training if no improvement for `early_stopping_rounds` epochs

    # Print progress
    # print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Val: {val_result:.4f}, Best Val: {best_val_result:.4f}")

results.append(val_result)

gc.collect()

end_time = time.time()

# Calculate training time
total_training_time += end_time - start_time

X_test = df_test
y_test = X_test[LABEL]
X_test.drop(columns=LABEL, inplace=True)

X_test[NUMERICAL_FEATURES] = num_preprocessor.transform(X_test[NUMERICAL_FEATURES])

gc.collect()

X_test_tensor_cat = torch.tensor(X_test[CATEGORICAL_FEATURES].values, dtype=torch.int32)
X_test_tensor_num = torch.tensor(X_test[NUMERICAL_FEATURES].values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

test_dataset = CatNumDataset(X_test_tensor_cat, X_test_tensor_num, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_loader = DeviceDataLoader(test_loader, device)

start_time = time.time()

model.eval()
all_test_targets = []
all_test_probs = []
with torch.no_grad():
    for X_cat_test_minib, X_num_test_minib, y_test_minib in test_loader:
        outputs_test_minib = model(X_cat_test_minib, X_num_test_minib)
        loss = loss_fn(outputs_test_minib, y_test_minib)

        probabilities_test_minib = torch.softmax(outputs_test_minib, dim=1)
        all_test_probs.append(probabilities_test_minib.cpu().numpy())
        all_test_targets.append(y_test_minib.cpu().numpy())

all_test_probs = np.concatenate(all_test_probs, axis=0)
all_test_targets = np.concatenate(all_test_targets, axis=0)

test_result = f1_score(np.argmax(all_test_targets, axis=1), np.argmax(all_test_probs, axis=1), average='micro')

end_time = time.time()

inference_time = end_time - start_time

# Calculate number of parameters
num_parameters = sum(p.numel() for p in model.parameters())
# Calculate model memory size
model_memory_size = sum(param.numel() * param.element_size() for param in model.parameters()) + sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())

print("FINAL RESULTS")
print("Metric - F1 score (micro): ", test_result)
print("Training time: ", total_training_time)
print("Inference time: ", inference_time)
print("Number of parameters: ", num_parameters)
print(f"Model memory size: {model_memory_size / 1024 / 1024 :.2f} MB")

# Confusion matrix
decoded_targets = label_pipe.inverse_transform(all_test_targets)
decoded_probs = label_pipe.inverse_transform(all_test_probs)
labels = label_pipe.named_steps['onehotencoder'].categories_[0]

confusion_matrix = confusion_matrix(decoded_targets, decoded_probs)

# Save results
results = pd.DataFrame({'Predictions': decoded_probs.flatten(), 'Ground Truth': decoded_targets.flatten()})
results.to_csv(output_path+"results.csv", index=False)
# Save labels to a file
with open(output_path+"labels.txt", "w") as f:
    f.write("\n".join(labels))
