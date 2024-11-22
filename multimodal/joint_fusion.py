import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class Attention(nn.Module):
    """
    Attention mechanism for the multimodal model
    """
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
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
    
class BERTProcessing_Large(nn.Module):
    """
    BERT processing module that reshapes BERT-EE 
    outputs to a single-channel "image" and passes 
    them through convolution layers.
    """
    def __init__(self, bert_dim, batch_size, hidden_size, dropout_rate):
        super(BERTProcessing_Large, self).__init__()
        channel_3 = int(hidden_size / 8)
        channel_4 = int(channel_3 / 2)
        
        self.model = nn.Sequential(
            nn.Linear(bert_dim * batch_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Unflatten(0, torch.Size([32, 32, 1])),
            nn.Conv1d(in_channels=32, out_channels=64, padding=2, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, padding=0),
            nn.Dropout(p=dropout_rate),
            nn.Conv1d(in_channels=64, out_channels=128, padding=2, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, padding=0),
            nn.Conv1d(in_channels=128, out_channels=256, padding=2, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, batch_size),
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
    Joint fusion model with fully connected layers where
        each modality outputs its own prediction for 
        individual loss backpropagation

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
    
        self.attention_bert = Attention(bert_size, self.hidden_size)
        self.attention_vb = Attention(vb_size, self.hidden_size)
        self.attention_pt_dem = Attention(pt_dem_size, self.hidden_size)

        self.attention = Attention(bert_size+vb_size+pt_dem_size, self.hidden_size)
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

        self.attention = Attention(bert_size+vb_size+pt_dem_size, self.hidden_size)
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
        outputs to a single-channel "image" 
        and passed to convolution layers

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
    Joint fusion model with CNN that treats
        BERT-based model output as an "image"
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
    Applies attention weights to the concatenated features
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

        self.attention = Attention(bert_size +vb_size + pt_dem_size, self.channel_2)
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
    Applies attention weights to each feature before concatenation for the final classifier
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

        self.attention_bert = Attention(bert_size, self.channel_2)
        self.attention_vb = Attention(vb_size, self.channel_2)
        self.attention_pt_dem = Attention(pt_dem_size, self.channel_2)

        self.attention = Attention(bert_size + vb_size + pt_dem_size, self.channel_2)
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
    Joint fusion model that returns independent outputs for each modality
        to be used for backpropagating individual losses 
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
    Joint fusion model that returns independent outputs for each modality
        to be used for backpropagating individual losses 
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
    
class JointFusion_Transformer(nn.Module):
    def __init__(self, batch_size, bert_dim, in_channels, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_Transformer, self).__init__()
        self.batch_size = batch_size
        self.bert_dim = bert_dim
        self.in_channels = in_channels
        self.vb_dim = vb_dim
        self.pt_dem_dim = pt_dem_dim
        self.dropout_rate = dropout_rate
        if Config.getstr("bert", "bert_mode") == "discrete":
            self.num_heads = 3
            if Config.getstr("preprocess", "encode_mode") == "onehot":
                self.hidden_size = 15
            elif Config.getstr("preprocess", "encode_mode") == "label":
                self.hidden_size = 3
        elif Config.getstr("bert", "bert_mode") == "cls":
            self.hidden_size = 768
            self.num_heads = 32
        
        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.hidden_size, 
                                           nhead=self.num_heads,
                                           batch_first=False, 
                                           norm_first=True), 
                                           num_layers=8,
                                           )

        self.bn = nn.BatchNorm1d(self.hidden_size)

        # Fully connected layers for x2 and x3
        self.fc_vb = nn.Linear(self.vb_dim, 32)
        self.fc_pt_dem = nn.Linear(self.pt_dem_dim, 32)

        # Final classification layer
        self.classifier = nn.Linear(self.hidden_size+32+32, self.batch_size)

    def forward(self, x1, x2, x3):      
        # Create attention mask for bert events
        mask = (x1.sum(dim=-1) != 0).float()
        extended_attention_mask = (1.0 - mask) * -10000.0

        # Apply transformer to x1
        x1 = x1.transpose(0, 1)
        x1 = self.transformer(x1, src_key_padding_mask=extended_attention_mask)

        # Apply batch normalization
        x1 = self.bn(x1.transpose(1, 2)).transpose(1, 2)

        # Pooling
        x1 = x1.mean(dim=0)

        # Process x2 and x3
        x2 = F.leaky_relu(self.fc_vb(x2))
        x3 = F.leaky_relu(self.fc_pt_dem(x3))

        x1 = x1.squeeze()
        x2 = x2.squeeze()
        x3 = x3.squeeze()

        # Concatenate all modalities
        out = torch.cat((x1, x2, x3))

        # Final classification
        out = self.classifier(out)
        out = torch.sigmoid(out)

        return out
    
""" CNN Models with increasing convolution size """
class JointFusion_CNN_Large(nn.Module):
    """
    Joint fusion model that reshapes BERT-EE 
        outputs to a single-channel "image" 
        and passed to convolution layers

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
        super(JointFusion_CNN_Large, self).__init__()
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

        self.bert = BERTProcessing_Large(self.bert_dim, self.batch_size, self.hidden_size, self.dropout_rate)

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
    
class JointFusion_CNN_Large_Attention_BeforeConcatenation(nn.Module):
    """
    Applies attention weights to each feature before concatenation for the final classifier
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(JointFusion_CNN_Large_Attention_BeforeConcatenation, self).__init__()
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

        self.bert = BERTProcessing_Large(self.bert_dim, self.batch_size, self.hidden_size, self.dropout_rate)

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

        self.attention_bert = Attention(bert_size, self.channel_2)
        self.attention_vb = Attention(vb_size, self.channel_2)
        self.attention_pt_dem = Attention(pt_dem_size, self.channel_2)

        self.attention = Attention(bert_size + vb_size + pt_dem_size, self.channel_2)
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

# class JointFusion_CNN_NoReshape_Attention(nn.Module):
#     """
#     Joint fusion model with CNN that treats
#         BERT-based model output as an "image"
#     """
#     def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
#         super(JointFusion_CNN_NoReshape_Attention, self).__init__()
#         self.batch_size = batch_size
#         self.bert_dim = bert_dim
#         self.vb_dim = vb_dim
#         self.pt_dem_dim = pt_dem_dim
#         self.hidden_size = hidden_size
#         self.dropout_rate = dropout_rate
#         self.channel_1 = int(self.hidden_size/2)
#         self.channel_2 = int(self.channel_1/2)
#         self.channel_3 = int(self.channel_2/2)
#         self.channel_4 = int(self.channel_3/2)
#         self.resize_shape = int(hidden_size/self.channel_1)
        
#         # Define sub-networks for each modality
#         self.bert_onehot = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
#             nn.BatchNorm1d(8),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
#             nn.BatchNorm1d(4),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Flatten(),
#             nn.Linear(24, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )

#         self.bert_label = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
#             nn.BatchNorm1d(8),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
#             nn.BatchNorm1d(4),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Flatten(),
#             nn.Linear(12, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )

#         self.bert_cls = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=128, padding=0, kernel_size=3),
#             nn.BatchNorm1d(128),
#             nn.MaxPool1d(kernel_size=4, padding=1),
#             nn.Conv1d(in_channels=128, out_channels=64, padding=0, kernel_size=3),
#             nn.BatchNorm1d(64),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Conv1d(in_channels=64, out_channels=32, padding=0, kernel_size=3),
#             nn.BatchNorm1d(32),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Flatten(),
#             nn.Linear(352, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )
        
#         self.vb = nn.Sequential(
#             nn.Linear(self.vb_dim, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )
        
#         self.pt_dem = nn.Sequential(
#             nn.Linear(self.pt_dem_dim, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )

#         bert_size = self.channel_4
#         vb_size = self.channel_4
#         pt_dem_size = self.channel_4

#         self.attention_bert = Attention(bert_size, self.channel_2)
#         self.attention_vb = Attention(vb_size, self.channel_2)
#         self.attention_pt_dem = Attention(pt_dem_size, self.channel_2)

#         self.attention = Attention(bert_size + vb_size + pt_dem_size, self.channel_2)
        
#         # Combine features from all modalities
#         self.classifier = nn.Linear(bert_size+vb_size+pt_dem_size, self.batch_size)
    
#     def forward(self, x1, x2, x3):
#         if Config.getstr("bert", "bert_mode") == "discrete":
#             if Config.getstr("preprocess", "encode_mode") == "label":
#                 out1 = self.bert_label(x1)
#             elif Config.getstr("preprocess", "encode_mode") == "onehot":        
#                 out1 = self.bert_onehot(x1)
#         elif Config.getstr("bert", "bert_mode") == "cls":
#             out1 = self.bert_cls(x1)
#         out2 = self.vb(x2)
#         out3 = self.pt_dem(x3)

#         out1 = torch.flatten(out1)
#         out2 = torch.flatten(out2)
#         out3 = torch.flatten(out3)

#         # Apply attention to each modality separately
#         attn_weights_bert = self.attention_bert(out1)
#         attn_weights_vb = self.attention_vb(out2)
#         attn_weights_pt_dem = self.attention_pt_dem(out3)

#         out1 = out1 * attn_weights_bert
#         out2 = out2 * attn_weights_vb
#         out3 = out3 * attn_weights_pt_dem

#         # Concatenate features
#         combined_features = torch.cat((out1, out2, out3))
#         out = self.classifier(combined_features)

#         out = torch.sigmoid(out)
        
#         return out
    
# class JointFusion_CNN_NoReshape_Permute(nn.Module):
#     """
#     Joint fusion model with CNN that treats
#         BERT-based model output as an "image"
#         and convolutional layer slides over
#         the sequence length
#     """
#     def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
#         super(JointFusion_CNN_NoReshape_Permute, self).__init__()
#         self.batch_size = batch_size
#         self.vb_dim = vb_dim
#         self.pt_dem_dim = pt_dem_dim
#         self.hidden_size = hidden_size
#         self.dropout_rate = dropout_rate
#         self.channel_1 = int(self.hidden_size/2)
#         self.channel_2 = int(self.channel_1/2)
#         self.channel_3 = int(self.channel_2/2)
#         self.channel_4 = int(self.channel_3/2)
#         self.resize_shape = int(hidden_size/self.channel_1)
#         if Config.getstr("bert", "bert_mode") == "discrete":
#             if Config.getstr("preprocess", "encode_mode") == "label":
#                 self.bert_dim = 3
#             elif Config.getstr("preprocess", "encode_mode") == "onehot":
#                 self.bert_dim = 15
#         elif Config.getstr("bert", "bert_mode") == "cls":
#             self.bert_dim = 768
        
#         # Define sub-networks for each modality
#         self.bert_onehot = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
#             nn.BatchNorm1d(8),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
#             nn.BatchNorm1d(4),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Flatten(),
#             nn.Linear(16, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_4, self.batch_size),
#         )

#         self.bert_label = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=8, padding=2, kernel_size=3),
#             nn.BatchNorm1d(8),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Conv1d(in_channels=8, out_channels=4, padding=2, kernel_size=3),
#             nn.BatchNorm1d(4),
#             nn.MaxPool1d(kernel_size=2, padding=1),
#             nn.Flatten(),
#             nn.Linear(12, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_4, self.batch_size),
#         )

#         self.bert_cls = nn.Sequential(
#             nn.Conv1d(in_channels=self.bert_dim, out_channels=256, padding=0, kernel_size=3),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=4, padding=1),
#             nn.Conv1d(in_channels=256, out_channels=128, padding=0, kernel_size=3),
#             nn.BatchNorm1d(128),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Conv1d(in_channels=128, out_channels=64, padding=0, kernel_size=3),
#             nn.BatchNorm1d(64),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Flatten(),
#             nn.Linear(128, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_4, self.batch_size),
#         )
        
#         self.vb = nn.Sequential(
#             nn.Linear(self.vb_dim, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_4, self.batch_size),
#         )
        
#         self.pt_dem = nn.Sequential(
#             nn.Linear(self.pt_dem_dim, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.hidden_size, self.batch_size),
#         )
        
#         # Combine features from all modalities
#         self.classifier = nn.Linear(self.batch_size*3, self.batch_size)
    
#     def forward(self, x1, x2, x3):
#         if Config.getstr("bert", "bert_mode") == "discrete":
#             if Config.getstr("preprocess", "encode_mode") == "label":
#                 # Permute tensor to match Conv1d input requirements (batch_size, in_channels, seq_length)
#                 x1 = x1.permute(0, 2, 1)
#                 out1 = self.bert_label(x1)
#             elif Config.getstr("preprocess", "encode_mode") == "onehot":
#                 # Permute tensor to match Conv1d input requirements (batch_size, in_channels, seq_length)
#                 x1 = x1.permute(0, 2, 1)        
#                 out1 = self.bert_onehot(x1)
#         elif Config.getstr("bert", "bert_mode") == "cls":
#             # Permute tensor to match Conv1d input requirements (batch_size, in_channels, seq_length)
#             x1 = x1.permute(0, 2, 1)    
#             out1 = self.bert_cls(x1)

#         out2 = self.vb(x2)
#         out3 = self.pt_dem(x3)

#         out1 = out1.squeeze(dim=1)
#         out2 = out2.squeeze(dim=1)
#         out3 = out3.squeeze(dim=1)
        
#         # Concatenate features
#         combined_features = torch.cat((out1, out2, out3))
#         out = self.classifier(combined_features)

#         out = torch.sigmoid(out)
        
#         return out

# class AdaptiveFusion(nn.Module):
#     def __init__(self, bert_dim, vb_dim, pt_dem_dim):
#         super(AdaptiveFusion, self).__init__()
#         self.bert = nn.Parameter(torch.ones(1))
#         self.vb = nn.Parameter(torch.ones(1))
#         self.pt_dem = nn.Parameter(torch.ones(1))
#         self.fc = nn.Linear(bert_dim + vb_dim + pt_dem_dim, 128)

#     def forward(self, x1, x2, x3):
#         x1_weighted = self.bert * x1
#         x2_weighted = self.vb * x2
#         x3_weighted = self.pt_dem * x3
#         combined_features = torch.cat((x1_weighted, x2_weighted, x3_weighted), dim=1)
#         out = self.fc(combined_features)
#         return out

# class JointFusion_CNN_Adaptive(nn.Module):
#     """
#     Joint fusion model that reshapes BERT-EE 
#         outputs to a single-channel "image" 
#         and passed to convolution layers

#     Parameters
#     ----------
#     batch_size (int)
#     bert_dim (int): BERT outputs dimension after flattening
#     vb_dim (int)
#     pt_dem_dim (int)
#     hidden_size (int)
#     dropout_rate (float): for nn.Dropout()
#     """
#     def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
#         super(JointFusion_CNN_Adaptive, self).__init__()
#         self.batch_size = batch_size
#         self.bert_dim = bert_dim
#         self.vb_dim = vb_dim
#         self.pt_dem_dim = pt_dem_dim
#         self.dropout_rate = dropout_rate
#         self.hidden_size = hidden_size
#         self.channel_1 = int(self.hidden_size/2)
#         self.channel_2 = int(self.channel_1/2)
#         self.channel_3 = int(self.channel_2/2)
#         self.channel_4 = int(self.channel_3/2)
#         self.resize_shape = int(hidden_size/16)

#         self.bert = BERTProcessing(bert_dim, batch_size, hidden_size, dropout_rate)

#         self.vb = nn.Sequential(
#             nn.Linear(self.vb_dim, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )

#         self.pt_dem = nn.Sequential(
#             nn.Linear(self.pt_dem_dim, self.channel_1),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_1, self.channel_2),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_2, self.channel_3),
#             nn.LeakyReLU(),
#             nn.Dropout(p=self.dropout_rate),
#             nn.Linear(self.channel_3, self.channel_4),
#         )
        
#         bert_size = 4*self.channel_4
#         vb_size = self.channel_4
#         pt_dem_size = self.channel_4
        
#         self.adaptive_fusion = AdaptiveFusion(bert_size, vb_size, pt_dem_size)
#         self.classifier = nn.Linear(128, self.batch_size)

#     def forward(self, x1, x2, x3):
#         x1 = torch.flatten(x1)

#         x1 = self.bert(x1)
#         x2 = self.vb(x2)
#         x3 = self.pt_dem(x3)

#         # x1 = torch.flatten(x1)
#         # x2 = torch.flatten(x2)
#         # x3 = torch.flatten(x3)
#         x1 = x1.transpose(0, 1)
#         out = self.adaptive_fusion(x1, x2, x3)

#         out = torch.flatten(out)
#         out = self.classifier(out)

#         # Apply sigmoid function
#         out = torch.sigmoid(out)

#         return out