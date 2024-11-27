import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusion_FC_Average_Losses(nn.Module):
    """
    Late fusion model with fully connected layers where
        each modality outputs its own probability, which 
        is averaged and binarized for the final prediction

    Parameters
    ----------
    batch_size (int)
    bert_dim (int)
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float)

    Returns
    -------
    out (float): averaged probability
    out1 (float): probability from BERT-based output
    out2 (float): probability from VB scores
    out3 (float): probability from patient demographics
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(LateFusion_FC_Average_Losses, self).__init__()
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

        # Apply sigmoid function
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)

        # Concatenate the probabilities
        out = torch.cat((out1, out2, out3), dim=0)
        out = torch.mean(out, dim=0, keepdim=True)

        return out, out1, out2, out3
    
class LateFusion_CNN_Average_Losses(nn.Module):
    """
    Late fusion model with CNN for BERT-based outputs where
        each modality outputs its own probability, which 
        is averaged and binarized for the final prediction

    Parameters
    ----------
    batch_size (int)
    bert_dim (int)
    vb_dim (int)
    pt_dem_dim (int)
    hidden_size (int)
    dropout_rate (float)

    Returns
    -------
    out (float): averaged probability
    out1 (float): probability from BERT-based output
    out2 (float): probability from VB scores
    out3 (float): probability from patient demographics
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(LateFusion_CNN_Average_Losses, self).__init__()
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

        # Apply sigmoid function
        out1 = torch.sigmoid(out1)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)

        # Concatenate the probabilities
        out = torch.cat((out1, out2, out3), dim=0)
        out = torch.mean(out, dim=0, keepdim=True)
        
        return out, out1, out2, out3