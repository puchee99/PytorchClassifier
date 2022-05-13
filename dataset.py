from __future__ import annotations
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Variable


#------------------------collect .csv file and load---------------------------------
def get_path(filename: str = 'train.csv') -> tuple[bool, str]:
    curr_path = str(os.getcwd())
    path_data = curr_path + '/data/' + filename
    file_extension = path_data.split(".")[-1]
    return path_data, file_extension

def get_dataframe(path_data: str, file_extension: str) -> pd.DataFrame:
    if file_extension == 'xlsx':
        df = pd.read_excel(path_data, engine='openpyxl')
    elif file_extension == 'xls':
        df = pd.read_excel(path_data)
    elif file_extension == 'csv':
        df = pd.read_csv(path_data)
    return df 

#------------------------get features to train, target_values, name_cols---------------------------------
def clean_data(df:pd.DataFrame, label_col_name:str, select_corr_col:bool=True) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    if select_corr_col:
        corr = df.corr().abs()[label_col_name]
        corr = corr[corr != 1]
        corr = corr[corr > 0.05]
        select_columns = list(corr.index)
        df_select_columns = df[select_columns]
        df_train = df_select_columns.copy()
    else:
        df_train = df.copy()
    y = df[label_col_name]
    return df_train, y, select_columns


#--------------------------- get model data ------------------------------------------------
def create_datasets(filename:str, target_col:str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    path, extension = get_path(filename)
    df = get_dataframe(path, extension)
    return clean_data(df, label_col_name=target_col )

def get_train_test_val(X,y) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
    scaler = StandardScaler()#MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_val, y_val = np.array(X_val), np.array(y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val

def get_train_test_val_variable(X_train, X_test, X_val, y_train, y_test, y_val) -> tuple[Variable, Variable, Variable, Variable, Variable, Variable]:
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test  = Variable(torch.from_numpy(X_test)).float()
    y_test  = Variable(torch.from_numpy(y_test)).long()
    X_val  = Variable(torch.from_numpy(X_val)).float()
    y_val  = Variable(torch.from_numpy(y_val)).long()
    return X_train, X_test, X_val, y_train, y_test, y_val

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def get_classifier_datasets(X_train: np.array, X_test:np.array, X_val:np.array, y_train, y_test, y_val) -> tuple[ClassifierDataset,ClassifierDataset,ClassifierDataset]:
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    return train_dataset, val_dataset, test_dataset

def get_weight_features(train_dataset, y_train,print_weights=True) -> tuple[torch.tensor, WeightedRandomSampler]:
    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    class_count = [i for i in get_class_distribution(y_train).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    if print_weights:
        print(class_weights)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    return class_weights, weighted_sampler

def get_loaders(train_dataset, val_dataset, test_dataset, weighted_sampler) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            sampler=weighted_sampler
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_loader, val_loader, test_loader

#--------------------------------------aux---------------------------------------
def create_dataset_without_true_label(train_file:str, test_file:str,label_name:str, index_col_name:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path, extension = get_path(train_file)
    df = get_dataframe(path, extension)
    _,_, selected_col = clean_data(df, label_col_name=label_name )

    path_test, extension_test = get_path(test_file)
    df_test = get_dataframe(path_test, extension_test)
    return df_test[selected_col], df_test[[index_col_name]]
    
def get_test_without_label_variable(df: pd.DataFrame):
    X = np.asarray(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return Variable(torch.from_numpy(X_scaled)).float()

def get_class_distribution(y: np.array) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
