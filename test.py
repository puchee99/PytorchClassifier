import argparse

import torch

from dataset import create_datasets, get_train_test_val,get_test_without_label_variable,create_dataset_without_true_label,get_train_test_val_variable
from model import MulticlassClassification, MulticlassSimpleClassification, init_weights
from utils import save_df_local, str2bool
from logs_utils import logging_loader
logger = logging_loader()

# construct the argument parser // python train.py --epochs 25  / python train.py -h
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--validation', type=str2bool, default=True,
    help='Test validation dataset')
parser.add_argument('-z','--zt' ,type=str2bool, default=True,
    help='Test validation dataset')
parser.add_argument('-r', '--results', type=str2bool, default=False,
    help='Predict and create results.csv (data without y_true_label)')
parser.add_argument('-l', '--label', type=str, default='Insect',
    help='Name of the label column')
parser.add_argument('-i', '--indexcol', type=str, default="Unnamed: 0",
    help='Name of the index column')
parser.add_argument('-trf', '--trainfile', type=str, default='train.csv',
    help='Name of the training file')
parser.add_argument('-tsf', '--testfile', type=str, default='test_x.csv',
    help='Name of the file to make results.csv')
parser.add_argument('-o', '--outfile', type=str, default='results.csv',
    help='Name of the file to save output predictions')
args = vars(parser.parse_args())

INDEX_COL = args['indexcol']
LABEL_NAME = args['label']
TRAIN_FILE = args['trainfile']
TEST_FILE = args['testfile']
OUT_FILE = args['outfile']

def test(model, X_test, y_test=False):
    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_test)
        array_pred = torch.argmax(y_pred_val, dim=1)
        if type(y_test) == torch.Tensor and y_test.shape[0] == X_test.shape[0]:
            correct = (array_pred == y_test).type(torch.FloatTensor)
            return  array_pred.detach().numpy(), correct.mean()
        else:
            return array_pred.detach().numpy(), 0

def load_best_model():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Computation device: {device}")
    model = MulticlassSimpleClassification(
        X_train.shape[1], 3
    ).to(device)
    best_model_cp = torch.load(f"output/models/best_"+"MulticlassSimpleClassification"+".pth")
    model.load_state_dict(best_model_cp['model_state_dict'])
    return model

if __name__ == '__main__':  
    df_train, y, col_names = create_datasets(TRAIN_FILE, LABEL_NAME)
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val(df_train,y)
    X_train, X_test, X_val, y_train, y_test, y_val = get_train_test_val_variable(X_train, X_test, X_val, y_train, y_test, y_val)
    model = load_best_model()
    if args['results']:
        df, df_results = create_dataset_without_true_label(TRAIN_FILE, TEST_FILE, LABEL_NAME,INDEX_COL)
        X_pred = get_test_without_label_variable(df)
        results, _ = test(model, X_pred)
        df_results[LABEL_NAME] = results
        save_df_local(df_results, output_name = OUT_FILE,create_folder=True)
        logger.info(f"Predictions from {TEST_FILE} saved in {OUT_FILE}")
    if args['validation']:
        results_train, accuracy_train = test(model, X_train, y_train)
        results_test, accuracy_test = test(model, X_test, y_test)
        results_val, accuracy_val = test(model, X_val, y_val)
        print(f"{accuracy_train} {accuracy_test} {accuracy_val}")
        logger.info(f"{accuracy_train} {accuracy_test} {accuracy_val}")



