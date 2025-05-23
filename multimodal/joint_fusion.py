import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class FeatureWeightingAttention(nn.Module):
    """
    Attention mechanism for the multimodal model
    """
    def __init__(self, input_dim, hidden_dim):
        super(FeatureWeightingAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, round(hidden_dim/2))
        self.fc3 = nn.Linear(round(hidden_dim/2), input_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if x is 1D
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        attn_weights = torch.softmax(x, dim=1)
        return attn_weights
    
""" Joint Fusion FC Blocks """
class Bert_FC_Block(nn.Module):
    def __init__(self, batch_size, bert_dim, hidden_size, channel_1, channel_2, channel_3, channel_4, dropout_rate):
        super(Bert_FC_Block, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3
        self.channel_4 = channel_4

        self.bert = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.batch_size * bert_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

    def forward(self, x):
        return self.bert(x)

""" Joint Fusion CNN Blocks """
class BERT_Reshape_CNN_Block(nn.Module):
    """
    BERT processing module that reshapes BERT-EE 
    outputs to a single-channel "image" and passes 
    them through convolution layers.
    """
    def __init__(self, bert_dim, batch_size, hidden_size, dropout_rate):
        super(BERT_Reshape_CNN_Block, self).__init__()
        channel_3 = int(hidden_size / 8)
        channel_4 = int(channel_3 / 2)
        
        self.model = nn.Sequential(
            nn.Linear(bert_dim * batch_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Unflatten(0, torch.Size([32, 32, 1])),
            nn.Conv1d(in_channels=32, out_channels=16, padding=2, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=16, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(start_dim=1),
            nn.Linear(24, channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_3, channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_4, batch_size),
        )

    def forward(self, x):
        return self.model(x)
    
""" Joint Fusion CNN NoReshape Blocks """
class BERTOneHot(nn.Module):
    """
    BERT OneHot processing module.
    """
    def __init__(self, bert_dim, hidden_size, dropout_rate):
        super(BERTOneHot, self).__init__()
        channel_1 = int(hidden_size / 2)
        channel_2 = int(channel_1 / 2)
        channel_3 = int(channel_2 / 2)
        channel_4 = int(channel_3 / 2)
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=bert_dim, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(),
            nn.Linear(24, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_1, channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_2, channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_3, channel_4),
        )

    def forward(self, x):
        return self.model(x)

class BERTLabel(nn.Module):
    """
    BERT Label processing module.
    """
    def __init__(self, bert_dim, hidden_size, dropout_rate):
        super(BERTLabel, self).__init__()
        channel_1 = int(hidden_size / 2)
        channel_2 = int(channel_1 / 2)
        channel_3 = int(channel_2 / 2)
        channel_4 = int(channel_3 / 2)
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=bert_dim, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(),
            nn.Linear(12, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_1, channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_2, channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_3, channel_4),
        )

    def forward(self, x):
        return self.model(x)

class BERTCls(nn.Module):
    """
    BERT CLS processing module.
    """
    def __init__(self, bert_dim, hidden_size, dropout_rate):
        super(BERTCls, self).__init__()
        channel_1 = int(hidden_size / 2)
        channel_2 = int(channel_1 / 2)
        channel_3 = int(channel_2 / 2)
        channel_4 = int(channel_3 / 2)
        
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=bert_dim, out_channels=128, padding=0, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=4, padding=1),
            nn.Conv1d(in_channels=128, out_channels=64, padding=0, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=32, padding=0, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(352, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_1, channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_2, channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(channel_3, channel_4),
        )

    def forward(self, x):
        return self.model(x)

""" Joint Fusion FC Models """
class JointFusion_FC(nn.Module):
    """
    Joint fusion model with fully connected layers

    Parameters
    ----------
    batch_size (int)
    bert_dim (int)
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float)
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_FC, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(self.hidden_size/(self.channel_1))
        
        self.bert = Bert_FC_Block(self.batch_size, self.bert_dim, self.hidden_size, self.channel_1, 
                                  self.channel_2, self.channel_3, self.channel_4, self.dropout_rate)
        self.vb = nn.Linear(self.vb_dim, 8)
        self.pt_dem = nn.Linear(self.pt_dem_dim, 16)
        
        bert_size = self.channel_4
        vb_size = 8
        pt_dem_size = 16
        
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        out1 = torch.flatten(x1)
        out2 = torch.flatten(x2)
        out3 = torch.flatten(x3)

        out = torch.cat((out1, out2, out3))
        out = self.classifier(out)

        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out
    
class JointFusion_FC_Losses(nn.Module):
    """
    Joint fusion model with fully connected layers 
        where each modality outputs its own prediction 
        for individual loss backpropagation

    Parameters
    ----------
    batch_size (int)
    bert_dim (int)
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float)
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_FC_Losses, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(self.hidden_size/(self.channel_1))
        
        self.bert = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.batch_size*bert_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3), 
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size), 
            nn.LeakyReLU(),
        )

        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )

        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
                
        bert_size = self.batch_size
        vb_size = self.batch_size
        pt_dem_size = self.batch_size
        
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        out1 = torch.flatten(x1)
        out2 = torch.flatten(x2)
        out3 = torch.flatten(x3)

        out = torch.cat((out1, out2, out3))
        out = self.classifier(out)

        # Apply sigmoid function
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        out = torch.sigmoid(out)

        return out, out1, out2, out3
    
class JointFusion_FC_Attention_BeforeConcatenation(nn.Module):
    """
    Joint fusion model with fully connected layers that applies
        attention weights to each modality prior to concatenation
        for the final classifier

    Parameters
    ----------
    batch_size (int)
    bert_dim (int): BERT-based outputs dimension after flattening
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float): for nn.Dropout()
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_FC_Attention_BeforeConcatenation, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(self.hidden_size/(self.channel_1))
        
        self.bert = Bert_FC_Block(self.batch_size, self.bert_dim, self.hidden_size, self.channel_1, 
                                  self.channel_2, self.channel_3, self.channel_4,self.dropout_rate)
        self.vb = nn.Linear(self.vb_dim, 8)
        self.pt_dem = nn.Linear(self.pt_dem_dim, 16)
        
        bert_size = self.channel_4
        vb_size = 8
        pt_dem_size = 16
    
        self.attention_bert = FeatureWeightingAttention(bert_size, self.hidden_size)
        self.attention_vb = FeatureWeightingAttention(vb_size, self.hidden_size)
        self.attention_pt_dem = FeatureWeightingAttention(pt_dem_size, self.hidden_size)

        self.attention = FeatureWeightingAttention(bert_size+vb_size+pt_dem_size, self.hidden_size)
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        # Apply attention to each modality separately
        attn_weights_bert = self.attention_bert(x1)
        attn_weights_vb = self.attention_vb(x2)
        attn_weights_pt_dem = self.attention_pt_dem(x3)

        out1 = x1*attn_weights_bert
        out2 = x2*attn_weights_vb
        out3 = x3*attn_weights_pt_dem

        out = torch.cat((out1, out2, out3), dim=1)
        out = out.squeeze()
        out = self.classifier(out)

        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out
    
class JointFusion_FC_Attention_AfterConcatenation(nn.Module):
    """
    Joint fusion model with fully connected layers that applies
        attention weights after concatenation for the final classifier

    Parameters
    ----------
    batch_size (int)
    bert_dim (int): BERT-based outputs dimension after flattening
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float): for nn.Dropout()
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_FC_Attention_AfterConcatenation, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(self.hidden_size/(self.channel_1))
        
        self.bert = Bert_FC_Block(self.batch_size, self.bert_dim, self.hidden_size, self.channel_1, 
                                  self.channel_2, self.channel_3, self.channel_4,self.dropout_rate)
        self.vb = nn.Linear(self.vb_dim, 8)
        self.pt_dem = nn.Linear(self.pt_dem_dim, 16)
        
        bert_size = self.channel_4
        vb_size = 8
        pt_dem_size = 16

        self.attention = FeatureWeightingAttention(bert_size+vb_size+pt_dem_size, self.hidden_size)
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        # Ensure tensors have the same batch size dimension
        out = torch.cat((x1, x2, x3))

        attn_weights = self.attention(out)
        out = out*attn_weights
        out = out.squeeze()
        out = self.classifier(out)

        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out

""" Joint Fusion CNN Models"""
class JointFusion_CNN(nn.Module):
    """
    Joint fusion model that reshapes BERT-EE 
        outputs to a 2D "image" and passes 
        to a CNN

    Parameters
    ----------
    batch_size (int)
    bert_dim (int): BERT outputs dimension after flattening
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float): for nn.Dropout()
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/16)

        self.bert = BERT_Reshape_CNN_Block(self.bert_dim, self.batch_size, self.hidden_size, self.dropout_rate)

        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )
        
        bert_size = 4*self.channel_4
        vb_size = self.channel_4
        pt_dem_size = self.channel_4
        
        self.classifier = nn.Linear(bert_size+vb_size+pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)

        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        out = torch.cat((x1, x2, x3))
        out = self.classifier(out)

        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out
    
class JointFusion_CNN_NoReshape(nn.Module):
    """
    Joint fusion model without reshaping that treats
        BERT-EE outputs as an "image" as a 2D 
        "image" and passes to a CNN
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_NoReshape, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/self.channel_1)
        
        # Define sub-networks for each modality
        self.bert_onehot = BERTOneHot(bert_dim, hidden_size, dropout_rate)
        self.bert_label = BERTLabel(bert_dim, hidden_size, dropout_rate)
        self.bert_cls = BERTCls(bert_dim, hidden_size, dropout_rate)
        
        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )
        
        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        bert_size = self.channel_4
        vb_size = self.channel_4
        pt_dem_size = self.channel_4
        
        # Combine features from all modalities
        self.classifier = nn.Linear(bert_size+vb_size+pt_dem_size, self.batch_size)
    
    def forward(self, x1, x2, x3):
        if Config.getstr("bert", "bert_mode") == "discrete":
            if Config.getstr("preprocess", "encode_mode") == "label":
                out1 = self.bert_label(x1)
            elif Config.getstr("preprocess", "encode_mode") == "onehot":        
                out1 = self.bert_onehot(x1)
        elif Config.getstr("bert", "bert_mode") == "cls":
            out1 = self.bert_cls(x1)
        out2 = self.vb(x2)
        out3 = self.pt_dem(x3)

        out1 = torch.flatten(out1)
        out2 = torch.flatten(out2)
        out3 = torch.flatten(out3)

        # Concatenate features
        combined_features = torch.cat((out1, out2, out3))
        out = self.classifier(combined_features)

        out = torch.sigmoid(out)
        
        return out
    
class JointFusion_CNN_Attention_AfterConcatenation(nn.Module):
    """
    Applies attention weights to each feature after 
        concatenation for the classifier
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_Attention_AfterConcatenation, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/self.channel_1)

        self.bert = BERT_Reshape_CNN_Block(bert_dim, batch_size, hidden_size, dropout_rate)

        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        bert_size = 4*self.channel_4
        vb_size = self.channel_4
        pt_dem_size = self.channel_4

        self.attention = FeatureWeightingAttention(bert_size +vb_size + pt_dem_size, self.channel_2)
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        # Ensure tensors have the same batch size dimension
        out = torch.cat((x1, x2, x3))
        
        attn_weights = self.attention(out)
        out = out*attn_weights
        out = out.squeeze()
        out = self.classifier(out)

        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out
    
class JointFusion_CNN_Attention_BeforeConcatenation(nn.Module):
    """
    Applies attention weights to each feature before 
        concatenation for the classifier
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_Attention_BeforeConcatenation, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/self.channel_1)

        self.bert = BERT_Reshape_CNN_Block(bert_dim, batch_size, hidden_size, dropout_rate)

        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
        )

        bert_size = 4*self.channel_4
        vb_size = self.channel_4
        pt_dem_size = self.channel_4

        self.attention_bert = FeatureWeightingAttention(bert_size, self.channel_2)
        self.attention_vb = FeatureWeightingAttention(vb_size, self.channel_2)
        self.attention_pt_dem = FeatureWeightingAttention(pt_dem_size, self.channel_2)

        self.attention = FeatureWeightingAttention(bert_size + vb_size + pt_dem_size, self.channel_2)
        self.classifier = nn.Linear(bert_size + vb_size + pt_dem_size, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)
        x1 = self.bert(x1)
        x2 = self.vb(x2)
        x3 = self.pt_dem(x3)

        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)
        
        # Apply attention to each modality separately
        attn_weights_bert = self.attention_bert(x1)
        attn_weights_vb = self.attention_vb(x2)
        attn_weights_pt_dem = self.attention_pt_dem(x3)

        x1 = x1 * attn_weights_bert
        x2 = x2 * attn_weights_vb
        x3 = x3 * attn_weights_pt_dem

        # Ensure tensors have the same batch size dimension
        out = torch.cat((x1, x2, x3), dim=1)
        
        out = out.squeeze()
        out = self.classifier(out)
        
        # Apply sigmoid function
        out = torch.sigmoid(out)

        return out
    
class JointFusion_CNN_Losses_NoReshape(nn.Module):
    """
    Joint fusion model with CNN for BERT-EE
        outputs without reshaping, where each 
        modality returns independent outputs 
        for backpropagating individual losses

    Returns
    -------
    out (float): probability from concatenated features
    out1 (float): probability from BERT-based output
    out2 (float): probability from VB scores
    out3 (float): probability from patient demographics
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_Losses_NoReshape, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/self.channel_1)
        
        # Define sub-networks for each modality
        self.bert_onehot = nn.Sequential(
            nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(),
            nn.Linear(24, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )

        self.bert_label = nn.Sequential(
            nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(),
            nn.Linear(12, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )

        self.bert_cls = nn.Sequential(
            nn.Conv1d(in_channels=self.bert_dim, out_channels=128, padding=0, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=4, padding=1),
            nn.Conv1d(in_channels=128, out_channels=64, padding=0, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(in_channels=64, out_channels=32, padding=0, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(352, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
        
        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
        
        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.channel_1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_1, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
        
        # Combine features from all modalities
        self.combine = nn.Linear(self.batch_size*3, self.batch_size)
    
    def forward(self, x1, x2, x3):
        if x1.shape[2] == 3:
            out1 = self.bert_label(x1)
        elif Config.getstr("bert", "bert_mode") == "cls":
            out1 = self.bert_cls(x1)
        elif Config.getstr("preprocess", "encode_mode") == "onehot":        
            out1 = self.bert_onehot(x1)
        out2 = self.vb(x2)
        out3 = self.pt_dem(x3)

        out1 = out1.squeeze(dim=1)
        out2 = out2.squeeze(dim=1)
        out3 = out3.squeeze(dim=1)
        
        # Concatenate features
        combined_features = torch.cat((out1, out2, out3))
        out = self.combine(combined_features)

        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        out = torch.sigmoid(out)
        
        return out, out1, out2, out3
    
class JointFusion_CNN_Losses_Reshape(nn.Module):
    """
    Joint fusion model with CNN for BERT-EE 
        outputs with reshaping, where each 
        modality returns independent outputs 
        for backpropagating individual losses

    Returns
    -------
    out (float): probability from concatenated features
    out1 (float): probability from BERT-based output
    out2 (float): probability from VB scores
    out3 (float): probability from patient demographics
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_Losses_Reshape, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.channel_1 = int(self.hidden_size/2)
        self.channel_2 = int(self.channel_1/2)
        self.channel_3 = int(self.channel_2/2)
        self.channel_4 = int(self.channel_3/2)
        self.resize_shape = int(hidden_size/self.channel_1)
        
        self.bert = nn.Sequential(
            nn.Linear(bert_dim*self.batch_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Unflatten(0, torch.Size([32, 32, 1])),
            nn.Conv1d(in_channels=32, out_channels=16, padding=2, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv1d(in_channels=16, out_channels=8, padding=2, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2, padding=1),
            nn.Flatten(start_dim=1), 
            nn.Linear(24, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
        self.fc_bert = nn.Linear(self.channel_4*4, self.batch_size)
        
        self.vb = nn.Sequential(
            nn.Linear(self.vb_dim, self.channel_2),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_2, self.channel_3),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_3, self.channel_4),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.channel_4, self.batch_size),
        )
        
        self.pt_dem = nn.Sequential(
            nn.Linear(self.pt_dem_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.batch_size),
        )
        
        self.combine = nn.Linear(self.batch_size*3, self.batch_size)
    
    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)     
        out1 = self.bert(x1)
        out1 = out1.squeeze(dim=1)

        # to match batch_size
        out1 = self.fc_bert(out1)

        # out1 = self.bert(x1)
        out2 = self.vb(x2)
        out3 = self.pt_dem(x3)

        out2 = out2.squeeze(dim=1)
        out3 = out3.squeeze(dim=1)
        
        # Concatenate features
        combined_features = torch.cat((out1, out2, out3))
        out = self.combine(combined_features)

        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        out = torch.sigmoid(out)
        
        return out, out1, out2, out3
    
""" Transformer-based models """
class JointFusion_Transformer(nn.Module):
    """
    Joint fusion model with transformer encoder for BERT-EE outputs
    If bert_mode = "discrete", contextual embeddings for categorical data
    """
    def __init__(self, batch_size, bert_seq_length, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_Transformer, self).__init__()
        self.batch_size = batch_size
        self.bert_seq_length = bert_seq_length
        self.bert_dim = bert_dim
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.dropout_rate = dropout_rate

        bert_mode = Config.getstr("bert", "bert_mode")
        if bert_mode == "discrete":
            self._init_discrete_mode()
        elif bert_mode == "cls":
            self._init_cls_mode()
      
        # Fully connected layers for x2 and x3
        self.fc_vb = nn.Linear(self.vb_dim, 16)
        self.fc_pt_dem = nn.Linear(self.pt_dem_dim, 16)

    def _init_discrete_mode(self):
        self.bert_discrete_list = Config.getlist("bert", "discrete_categories")
        self.categories = [int(item) for item in self.bert_discrete_list]

        self.num_categories = len(self.categories)
        self.num_unique_categories = sum(self.categories)

        self.num_special_tokens = 1
        total_tokens = self.num_unique_categories + self.num_special_tokens
        
        categories_offset = F.pad(torch.tensor(list(self.categories)), (1, 0), value = self.num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.register_buffer('categories_offset', categories_offset)
    
        self.embedding_dim = 16
        self.categorical_embeds = nn.Embedding(total_tokens, self.embedding_dim)

        self.num_heads = 4
        self.transformer_discrete = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim ,
                nhead=self.num_heads,
                dropout=self.dropout_rate,
                batch_first=False,
                norm_first=True
            ),
            num_layers=8,
        )
        self.fc_bert_discrete = nn.Linear(self.embedding_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        # Final classification layer
        self.classifier = nn.Linear(1 + 16 + 16, self.batch_size)
    
    def _process_discrete(self, x1, extended_attention_mask=None):
        # Offset categorical indices, no reshaping here
        x1 = x1.long() + self.categories_offset
        
        # Reshape the input before embedding
        x1 = x1.view(x1.shape[0], x1.shape[1] * x1.shape[2])
        
        # Embed all categorical indices using the single embedding layer
        x_embedded = self.categorical_embeds(x1)
        
        # cls token prepending
        b = x_embedded.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
       
        x_embedded = torch.cat((cls_tokens, x_embedded), dim = 1)

        # Adjust attention mask to account for CLS token
        new_mask = torch.cat((torch.zeros((b, 1), dtype=extended_attention_mask.dtype, device=extended_attention_mask.device),
                             extended_attention_mask.repeat(1, self.num_categories)), dim=1)
         
        x1 = self.transformer_discrete(x_embedded.transpose(0, 1), src_key_padding_mask=new_mask)
        
        #take CLS token
        x1 = x1[0]
    
        return F.leaky_relu(self.fc_bert_discrete(x1))
    
    def _init_cls_mode(self):
        self.hidden_size = 768
        self.num_heads = 32
        self.transformer_cls = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dropout=self.dropout_rate,
                batch_first=False,
                norm_first=False
            ),
            num_layers=8,
        )
        self.fc_bert_cls = nn.Linear(self.hidden_size, 1)

        # Final classification layer
        self.classifier = nn.Linear(self.bert_seq_length + 16 + 16, self.batch_size)

    def forward(self, x1, x2, x3):
        # Create attention mask for bert events
        mask = (x1.sum(dim=-1) != 0).float()
        extended_attention_mask = (1.0 - mask) * -10000.0

        bert_mode = Config.getstr("bert", "bert_mode")

        if bert_mode == "discrete":
            x1 = self._process_discrete(x1, extended_attention_mask)
        elif bert_mode == "cls":
            x1 = self._process_cls(x1, extended_attention_mask)

        # Fully connected layers
        x2 = F.leaky_relu(self.fc_vb(x2))
        x3 = F.leaky_relu(self.fc_pt_dem(x3))

        x1 = x1.view(-1)
        x2 = x2.squeeze()
        x3 = x3.squeeze()

        # Concatenate all modalities
        out = torch.cat((x1, x2, x3))

        # Final classification
        out = self.classifier(out)
        out = torch.sigmoid(out)

        return out

    def _process_cls(self, x1, extended_attention_mask):
        x1 = self.transformer_cls(x1.transpose(0, 1), src_key_padding_mask=extended_attention_mask)
        return F.leaky_relu(self.fc_bert_cls(x1))
