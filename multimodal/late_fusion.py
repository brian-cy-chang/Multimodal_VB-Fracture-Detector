import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusion_FC(nn.Module):
    """
    Late fusion model with fully connected layers

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
        super(LateFusion_FC, self).__init__()
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
        
        self.input_size = self.bert_dim + self.vb_dim + self.pt_dem_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
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

    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        x = torch.cat((x1, x2, x3))
        x = self.classifier(x)

        # Apply sigmoid function
        x = torch.sigmoid(x)

        return x

class LateFusion_CNN(nn.Module):
    """
    Late fusion model that reshapes the concatenated features as
        a multi-channel "image" passed to convolution layers

    Parameters
    ----------
    batch_size (int)
    bert_dim (int)
    vb_dim (int)
    pt_dim (int)
    hidden_size (int)
    dropout_rate (float)
    """
    def __init__(self, batch_size, bert_dim, vb_dim, pt_dem_dim, hidden_size, dropout_rate=0.5):
        super(LateFusion_CNN, self).__init__()
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
                
        self.input_size = self.bert_dim + self.vb_dim + self.pt_dem_dim
        self.bert = nn.Sequential(
            nn.Linear(self.input_size, 1024),
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
            # nn.LeakyReLU(),
        )
        
        # for reducing the output to match the label size
        self.fc = nn.Linear(self.channel_4*4, self.batch_size)

    def forward(self, x1, x2, x3):
        x1 = torch.flatten(x1)
        x2 = torch.flatten(x2)
        x3 = torch.flatten(x3)

        x = torch.cat((x1, x2, x3))
        
        x = self.classifier(x)
        x = x.squeeze()
        x = self.fc(x)

        # Apply sigmoid function
        x = torch.sigmoid(x)

        return x



