from os import listdir
from os.path import isfile, join, splitext
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import Config

class MultimodalDataset(Dataset):
    """
    Finish data preprocessing and convert each data modality to tensors for PyTorch DataLoader

    Parameters
    ----------
    vb_path (str)
    BERT_file (str)
    pt_dem_file (str)
    cols (dict): expected columns from one-hot encoding BERT discrete events
    bert_dict (dict): for label encoding BERT discrete events
    fx_labels (dict): for binarizing m2ABQ fracture classification
    """
    def __init__(self, vb_path, BERT_file, pt_dem_file, cols, bert_dict, fx_labels):
        # self.__data = data
        self.__image_ids = []
        self.__bert_lst = []
        self.__bert_events = []
        self.__bert_padded = None
        self.__vb_scores = None
        self.__vb_padded = None
        self.__pt_dems = None
        self.__labels = None
        self.cols = cols
        self.bert_dict = bert_dict
        self.fx_labels = fx_labels

        self.vb_path = vb_path
        self.BERT_file = BERT_file
        self.pt_dem_file = pt_dem_file
        # self.encode_mode = Config.getstr("bert", "bert_mode")

        from preprocessing import preprocess
        self.preprocess = preprocess()

        if Config.getstr("bert", "bert_mode") == "discrete":
            self.__combined_dict = self.preprocess.combine_modalities(self.vb_path, self.BERT_file, self.pt_dem_file)
            for image_id in sorted(self.__combined_dict):
                df = self.__combined_dict[image_id]["BERT_discrete"]
                if 'FractureAnatomy' not in df.columns:
                    df['FractureAnatomy'] = np.NaN
                if 'FractureCause' not in df.columns:
                    df['FractureCause'] = np.NaN
                if 'FractureAssertion' not in df.columns:
                    df['FractureAssertion'] = np.NaN
                df['image_id'] = str(image_id)
                self.__bert_lst.append(df)
            
            if Config.getstr("preprocess", "encode_mode") == "onehot":
                # one hot encode the fracture argument roles
                onehot_encoded_bert = pd.get_dummies(pd.concat(self.__bert_lst), columns=['FractureAnatomy', 'FractureCause', 'FractureAssertion'])

                for col in cols:
                        if col not in onehot_encoded_bert.columns:
                            onehot_encoded_bert[col] = 0

                for pt in onehot_encoded_bert.image_id.unique().tolist():
                    slc = onehot_encoded_bert.loc[onehot_encoded_bert.image_id == pt]
                    slc = slc.drop(columns=['image_id'], axis=1)

                    array = np.array(slc).astype(np.float32)
                    bert_tensor = torch.Tensor(array).to(torch.float32)
                    self.__bert_events.append(bert_tensor)

                self.__bert_padded = [F.pad(sequence, (0, 0, Config.getint("bert", "discrete_max_length_pad")-sequence.size(0), 0), value=0) for sequence in self.__bert_events]
            elif Config.getstr("preprocess", "encode_mode") == "label":
                self.__df = pd.concat(self.__bert_lst)
                self.__df = self.__df.replace(self.bert_dict)
                for pt in self.__df.image_id.unique().tolist():
                    slc = self.__df.loc[self.__df.image_id == pt]
                    slc = slc.drop(columns=['image_id'], axis=1)

                    array = np.array(slc).astype(np.float32)
                    bert_tensor = torch.Tensor(array).to(torch.float32)
                    self.__bert_events.append(bert_tensor)

                self.__bert_padded = [F.pad(sequence, (0, 0, Config.getint("bert", "discrete_max_length_pad")-sequence.size(0), 0), value=0) for sequence in self.__bert_events]

        elif Config.getstr("bert", "bert_mode") == "cls":
            self.__combined_dict = self.preprocess.combine_modalities(self.vb_path, self.BERT_file, self.pt_dem_file)
            for image_id in sorted(self.__combined_dict):
                cls_embedding = self.__combined_dict[image_id]["BERT_cls"]
                bert_tensor = torch.Tensor(cls_embedding).to(torch.float32).squeeze(dim=1)
                self.__bert_lst.append(bert_tensor)

            self.__bert_padded = [F.pad(sequence, (0, 0, Config.getint("bert", "cls_max_length_pad")-sequence.size(0), 0), value=0) for sequence in self.__bert_lst]
            # self.__bert_padded = [F.pad(sequence, (0, 0, 0, 0, Config.getint("bert", "cls_max_length_pad")-sequence.size(0), 0), value=0) for sequence in self.__bert_lst]

        # return combined_dict keys (image IDs) as list
        for filename in list(self.__combined_dict.keys()):
            self.__image_ids.append(filename)

        # get label tensors
        if Config.getstr("general", "mode") == "train":
            self.__labels = self.preprocess.get_gt_labels_to_tensor(label_file=Config.getstr("labels", "groundtruth_file"), 
                                                                    ID_list=self.__image_ids, 
                                                                    fx_labels=self.fx_labels)
        elif Config.getstr("general", "mode") == "predict":
            self.__label_zeroes = np.zeros(len(self.__image_ids))
            self.__labels = torch.tensor((self.__label_zeroes).astype(np.float32), dtype=torch.float32)

        # get VB scores and convert to PyTorch tensor and pad to fixed length
        # self.__vbs = self.preprocess.get_max_vb_to_tensor(self.__combined_dict)
        self.__vb_scores = self.preprocess.get_vbs_to_tensor(self.__combined_dict)
        self.__vb_padded = [F.pad(torch.tensor(sequence, dtype=torch.float32), (Config.getint("vb", "vb_max_length_pad")-torch.tensor(sequence).size(0), 0), value=0) for sequence in self.__vb_scores]

        # convert patient demographics to PyTorch tensor
        self.__pt_dems = self.preprocess.pt_dem_to_tensor(self.__combined_dict)

    def __len__(self):
        return len(self.__image_ids)
    
    def __getitem__(self, index):
        data = {"bert": self.__bert_padded[index], 
                "vb": self.__vb_padded[index], 
                "pt_dem": self.__pt_dems[index], 
                "label": self.__labels[index], 
                "index": index}
            
        return data, self.__image_ids[index]
    
class CustomCollate:
    def __call__(self, batch):
        bert_events = torch.stack([item[0]["bert"] for item in batch])
        max_vbs = torch.stack([item[0]["vb"] for item in batch])
        pt_dems = torch.stack([item[0]["pt_dem"] for item in batch])
        labels = torch.stack([item[0]["label"] for item in batch])
        indexes = torch.stack([torch.tensor(item[0]["index"]) for item in batch])
        image_ids = [item[1] for item in batch]

        return [bert_events, max_vbs, pt_dems, labels, indexes], image_ids
