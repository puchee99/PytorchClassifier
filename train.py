import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from model import MulticlassClassification, MulticlassSimpleClassification, init_weights
from dataset import create_datasets, get_dataframe, get_path, get_train_test_val, get_train_test_val_variable, get_classifier_datasets, get_weight_features,get_loaders
from utils import SaveBestModel, save_model, save_plots, save_plot_cm, save_plot_roc, plot_features,str2bool
save_best_model = SaveBestModel()
from logs_utils import logging_loader
logger = logging_loader()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Computation device: {device}\n")

#https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=2,
    help='number of epochs to train our network for')
parser.add_argument('-lr', '--learningrate', type=float, default=0.0007,#1e-4,
    help='lerning rate number to train our network')
parser.add_argument('-bl', '--batchlearning', type=str2bool, default=False,
    help='if true creates DataLoaders to use batch in training')
parser.add_argument('-trf', '--trainfile', type=str, default='train.csv',
    help='Name of the training file')
parser.add_argument('-l', '--label', type=str, default='Insect',
    help='Name of the label column')
parser.add_argument('-tf', '--tensorboard', type=str2bool, default=False,
    help='save logs to use in tensorboard')
parser.add_argument('-mp', '--makeplots', type=str2bool, default=True,
    help='make some plots and save to output/plots folder')
parser.add_argument('-r', '--run_test', type=str2bool, default=True,
    help='run train test val accuracy')
parser.add_argument('-rr', '--test_without_label', type=str2bool, default=True,
    help='predict target and save results without knowing y_pred true label')
#parser.add_argument('-m', '--model', type=str, default='model_name',
#    help='name to train just one model')
args = vars(parser.parse_args())

LABEL_NAME = args['label']
TRAIN_FILE = args['trainfile']
LR = args['learningrate']
EPOCHS = args['epochs']
config = {
    "l1": tune.sample_from(lambda _: 2**np.random.randint(5, 8)),
    "l2": tune.sample_from(lambda _: 2**np.random.randint(5, 8)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}

if args['tensorboard']:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('output/log_runs/insects_experiment_1')
if args['makeplots']:
    path, extension = get_path('train.csv')
    df = get_dataframe(path, extension)
    plot_features(df,df['Insect'])

#------------------------train with batch-----------------------------
def run_model_batch(model, train_loader, validate_loader):
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model_batch.parameters(), lr=0.0007)
    #optimizer = torch.optim.Adam(model_batch.parameters(), lr=config['lr'],momentum=0.9)
    accuracy_stats = {}
    loss_stats = {}

    loss_stats['train']  = np.zeros((EPOCHS+1,))
    loss_stats['val'] = np.zeros((EPOCHS+1,))
    accuracy_stats['train']     = np.zeros((EPOCHS+1,))
    accuracy_stats['val'] = np.zeros((EPOCHS+1,))

    for e in tqdm(range(1, EPOCHS+1)):
        train_epoch_loss, train_epoch_acc = train_batch(model, train_loader, criterion, optimizer)
        val_epoch_loss, val_epoch_acc = validate_batch(model, validate_loader, criterion)
        loss_stats['train'][e], accuracy_stats['train'][e] = train_epoch_loss, train_epoch_acc
        loss_stats['train'][e], accuracy_stats['train'][e] = val_epoch_loss, val_epoch_acc
        if e % 100 == 0:
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f} | Train Acc: {train_epoch_acc:.3f}| Val Acc: {val_epoch_acc:.3f}')
    # Create dataframes
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # Plot the dataframes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
    plt.show()
    return

def train_batch(model, train_loader, criterion, optimizer):
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:

        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    return train_epoch_loss/len(train_loader), train_epoch_acc/len(train_loader)

def validate_batch(model, val_loader, criterion):
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    return val_epoch_loss/len(val_loader), val_epoch_acc/len(val_loader)


#-------------------------train without batch-------------------------
def run_multiple_models(X_train, X_test, y_train, y_test):
    #model     = MulticlassClassification(X_train.shape[1], 3)
    model     = MulticlassSimpleClassification(X_train.shape[1], 3).to(device)
    model.apply(init_weights)
    run_model(model, X_train, X_test, y_train, y_test)
    #model2     = MulticlassSimpleClassification(X_train.shape[1], 3).to(device)
    #model2.apply(init_weights)
    #run_model(model2, X_train, X_test, y_train, y_test)
    return 

def run_model(model,X_train,X_test,y_train, y_test):
    print_model_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    criterion   = torch.nn.CrossEntropyLoss()
    train_loss     = np.zeros((EPOCHS,))
    train_acc = np.zeros((EPOCHS,))
    valid_loss     = np.zeros((EPOCHS,))
    valid_acc = np.zeros((EPOCHS,))
    last_loss, dif_last_loss, epoch = np.inf, 1e5, 0

    while dif_last_loss > 0.00000001 and epoch < EPOCHS:
        train_epoch_loss, train_epoch_acc = train(model, X_train,y_train, optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, X_test, y_test, criterion)
        train_loss[epoch], valid_loss[epoch] = train_epoch_loss, valid_epoch_loss
        train_acc[epoch], valid_acc[epoch] = train_epoch_acc, valid_epoch_acc
        dif_last_loss = np.abs(last_loss - valid_epoch_loss)
        last_loss = valid_epoch_loss
        
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion
        )
        if (epoch+1) % 200 == 0:
            print("Epoch [{}/{}],  Loss: {:.4f}  Accuracy: {:.4f} ".format(epoch+1, EPOCHS,  valid_epoch_loss, valid_epoch_acc))
            logger.info("Epoch [{}/{}],  Loss: {:.4f}  Accuracy: {:.4f} ".format(epoch+1, EPOCHS,  valid_epoch_loss, valid_epoch_acc))
        epoch+=1
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "output/checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=valid_epoch_loss, accuracy=valid_epoch_acc)

    save_model(epoch, model, optimizer, criterion)
    if args['tensorboard']:
        writer.add_scalar('training loss',last_loss,epoch)
    logger.info("Finish training model \n")
    save_plots(train_acc[:epoch], valid_acc[:epoch], train_loss[:epoch], valid_loss[:epoch],str(type(model)).split("'")[1].split(".")[-1])# model.name)
    save_plot_cm(X_test, y_test, model)
    save_plot_roc(X_test, y_test, model)
    return valid_acc[:epoch], valid_loss[:epoch], model 

def train(model, X, y, optimizer, criterion):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    correct_train = (torch.argmax(y_pred, dim=1) == y).type(torch.FloatTensor)
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), correct_train.mean()

def validate(model, X_test, y_test, criterion):
    model.eval()
    y_pred_val = model(X_test)
    val_loss = criterion(y_pred_val, y_test)
    correct = (torch.argmax(y_pred_val, dim=1) == y_test).type(torch.FloatTensor)
    return val_loss.item(), correct.mean()


#--------------------------aux--------------------------------------
def print_model_params(model):
    logger.info('Running Model\n: {}'.format(model))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"{total_trainable_params:,} training parameters.")
    return
    
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return torch.round(acc * 100)

if __name__ == "__main__":
    df_train, y, col_names = create_datasets(TRAIN_FILE, LABEL_NAME)
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(df_train,y)
    X_train_t, X_test_t, X_val_t, y_train_t, y_test_t, y_val_t = get_train_test_val_variable(X_train, X_test, X_val, y_train, y_test, y_val)
    if args['batchlearning']:
        train_dataset, val_dataset, test_dataset = get_classifier_datasets(X_train, X_test, X_val, y_train, y_test, y_val)
        class_weights, weighted_sampler = get_weight_features(train_dataset, y_train)
        model_batch = MulticlassClassification(num_feature = X_train.shape[1], num_class=3).to(device)
        model_batch.apply(init_weights)
        train_loader, val_loader, test_loader = get_loaders(train_dataset, val_dataset, test_dataset, weighted_sampler)
        run_model_batch(model_batch, train_loader, val_loader)
    else:
        run_multiple_models(X_train_t, X_test_t, y_train_t, y_test_t)
    if args['run_test'] or args['test_without_label']:
        os.system(f"python test.py -v {args['run_test']} -r {args['test_without_label']}")
    


"""
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

#print(classification_report(y_test, y_pred_list))

"""