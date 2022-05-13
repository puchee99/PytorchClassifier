import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassSimpleClassification(nn.Module):
        def __init__(self, input_dim, output_dim, l1=256, l2=128, l3=64):
            super(MulticlassSimpleClassification, self).__init__()
            self.name = 'MulticlassSimpleClassification'
            self.layer1 = nn.Linear(input_dim, l1)
            self.layer2 = nn.Linear(l1, l2)
            self.layer3 = nn.Linear(l2, l3)
            self.out = nn.Linear(l3, output_dim)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            x = torch.sigmoid(self.layer3(x))
            x = F.softmax(self.out(x), dim=1)
            return x
            
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.name = 'MulticlassClassification'
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        #x = F.softmax(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        # torch.nn.init.zeros_(self.layer_1.bias)


