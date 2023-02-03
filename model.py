import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
torch.manual_seed(2020)
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import pandas as pd
import matplotlib.pyplot as plt

class GAT_AT(torch.nn.Module):
    def __init__(self, num_features, hid, num_classes, head, layer, layer_type='standard'):
        super(GAT_AT, self).__init__()
        self.hid = hid
        self.in_head = head
        self.num_classes = num_classes
        self.out_head = 1
        self.num_features = num_features
        if layer_type == 'v2':
            self.conv1 = GATv2Conv(self.num_features, self.hid, heads=self.in_head, dropout=0.6,concat=1)
            self.conv2 = GATv2Conv(self.hid*self.in_head, self.num_classes, concat=False,
                                heads=self.out_head, dropout=0.6)
            self.conv3 = GATv2Conv(self.num_features, self.num_classes, heads=self.in_head, dropout=0.6,concat=False)
        else:
            self.conv1 = GATConv(self.num_features, self.hid, heads=self.in_head, dropout=0.6,concat=1)
            self.conv2 = GATConv(self.hid*self.in_head, self.num_classes, concat=False,
                                heads=self.out_head, dropout=0.6)
            self.conv3 = GATConv(self.num_features, self.num_classes, heads=self.in_head, dropout=0.6,concat=False)
            
        self.layer = layer
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        if self.layer==1:

            x = F.dropout(x, p=0.6, training=self.training)
            x, att1 = self.conv3(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            return F.log_softmax(x, dim=1), att1, x
        elif self.layer == 2:
            x = F.dropout(x, p=0.6, training=self.training)
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x, att2  = self.conv2(x,edge_index, return_attention_weights=True)
            return F.log_softmax(x, dim=1), [att1,att2], x
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def train(model, data, epochs, df, dt_name, hid, head, layer, epoch_list):
    """Train a GNN model and return the trained model."""
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,weight_decay=5e-4)

    model.train()
    if layer == 1:
        att_list = []
    elif layer == 2:
        att_list = [[],[]]
    feature_list = []
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        out, att1, x = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        # if(epoch % 10 == 0):
            # print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
            #       f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
            #       f'Val Acc: {val_acc*100:.2f}%')  
        if epoch in epoch_list:
            model.eval()
            
            out, att1, x = model(data)
            test_acc = test(model, data)
            row = [dt_name, epoch, hid, head, loss.item(), acc*100, test_acc*100, layer]
            df.loc[len(df)] = row
            if layer==1:
                att_list.append(att1)
            elif layer == 2:
                att_list[0].append(att1[0])
                att_list[1].append(att1[1])

            feature_list.append(x)
    return model, att1, x, att_list, feature_list

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    out, att1, att2  = model(data)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

def viz_loss_or_acc(df, type, i=100):
    plt.figure(figsize=(10,5))
    for key, grp in df.groupby(['Head']):
        plt.plot(grp['Iter'].iloc[:i], grp[type].iloc[:i], label=f'head={key}')
    plt.legend()
    plt.grid()

def viz_feature_variation(flat_fea_heads, heads):
    plt.figure()
    for i in range(len(heads)):
        plt.plot(np.sum(np.abs(np.diff(np.array(flat_fea_heads[i]), axis=0)),axis=1), label=f'head={heads[i]}')
        plt.grid()
    plt.legend()

def viz_att_variation(flat_att_heads, heads):
    plt.figure()
    for i in range(len(heads)):
        plt.plot(np.sum(np.abs(np.diff(np.array(flat_att_heads[i]), axis=0))/heads[i],axis=1), label=f'head={heads[i]}')
        plt.grid()
    plt.legend()

def flatten_fea(fea_all_heads):
    flat_fea_heads = []
    for feature_list in fea_all_heads:
        flat_features = []
        for l in feature_list:
            l = l.tolist()
            flat_list = [item for sublist in l for item in sublist]
            flat_features.append(flat_list)
        flat_fea_heads.append(flat_features)
    return flat_fea_heads

def flatten_att(att_all_heads):
    flat_att_heads = []
    for att_list in att_all_heads:
        flat_attentions = []
        for l in att_list:
            l = l[1].tolist()
            flat_att = [item for sublist in l for item in sublist]
            flat_attentions.append(flat_att)
        flat_att_heads.append(flat_attentions)
    return flat_att_heads