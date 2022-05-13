import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True
plt.style.use('ggplot')
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import torch



script_dir = os.path.dirname(__file__)
results_dir_models = os.path.join(script_dir, 'output/models/')

results_dir_plots = os.path.join(script_dir, 'output/plots/')
if not os.path.isdir(results_dir_models):
    os.makedirs(results_dir_models)
if not os.path.isdir(results_dir_plots):
    os.makedirs(results_dir_plots)

def save_df_local(df: pd.DataFrame, output_name: str='results.csv', create_folder: bool=False, new_folder_path:str = 'output/results', compressed: bool=False ):
    if create_folder:
        os.makedirs(new_folder_path, exist_ok=True) #'folder/subfolder'
        output_name = new_folder_path + '/' + output_name

    if compressed:
        name_compressed = output_name.split(".")[0] + '.zip'
        compression_opts = dict(method='zip',archive_name=output_name.split("/")[-1])  
        df.to_csv(name_compressed, index=False,compression=compression_opts) 
    else:
        df.to_csv(output_name,index=False) 
    return

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            #print(f"\nBest validation loss: {self.best_valid_loss}")
            #print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, results_dir_models + 'best_'+model.name+'.pth')

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    #print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, results_dir_models + 'final_'+model.name+'.pth')


#--------------------------------------PLOTS------------------------------------------------
def save_plots(train_acc, valid_acc, train_loss, valid_loss, model_name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(results_dir_plots + model_name + '_accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_dir_plots + model_name + '_loss.png')
    return 

def save_plot_cm(X_test, y_test, model):
    with torch.no_grad():
        y_pred = model(X_test)#.numpy()
    mat = confusion_matrix(y_test.detach().numpy(), torch.argmax(y_pred, dim=1).detach().numpy())
    plt.figure(figsize=(10, 7))

    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('NN')
    plt.savefig(results_dir_plots + model.name + '_cm.png')
    #plt.show()
    return

def save_plot_roc(X_test, y_test, model):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')

    # One hot encoding
    enc = OneHotEncoder()
    Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

    with torch.no_grad():
        y_pred = model(X_test).numpy()
        fpr, tpr, threshold = roc_curve(Y_onehot.ravel(), y_pred.ravel())
        
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    #plt.show()
    plt.savefig(results_dir_plots +model.name+'_roc.png')
    return 

def plot_features(df,y):
    select_columns = df[["Hour","Sensor_beta","Sensor_alpha_plus","Sensor_gamma"]]
    df_train = select_columns.copy()
    names = ['Lepidoptero', 'Himenoptera', 'Diptera'] #[0,1,2]
    feature_names = list(df_train.columns)
    X = np.asarray(df_train)    

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    one_plot_features(X,names, ax1, feature_names, y,2,3)
    one_plot_features(X, names, ax2, feature_names, y,1,2)
    one_plot_features(X, names, ax3, feature_names,y, 1,3)
    plt.savefig(results_dir_plots+ 'features_distribution.png')
    return

def one_plot_features(X, names, ax , feature_names, y,i, z):
    for target, target_name in enumerate(names):
            X_plot = X[y == target]
            ax.plot(X_plot[:, i], X_plot[:, z], 
                     linestyle='none', 
                     marker='o', 
                     label=target_name)
    ax.set_xlabel(feature_names[i])
    ax.set_ylabel(feature_names[z])
    ax.axis('equal')
    ax.legend()
    #plt.plot()
    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')