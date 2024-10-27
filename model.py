import os
import sys
import time
import datetime
import pickle
from os import listdir
from os.path import isfile, join, splitext
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.optim as optim

from config import Config
# from multimodal_dataloader import MultimodalDataset
from utils import WeightedFocalLoss, kaiming_init, xavier_init, he_init, calculate_specificity, calculate_npv
import user_params as uparams

class Multimodal_VB_Fracture_Detector(nn.Module):
    """
    Parameters
    ----------
    Set in config.ini

    model (PyTorch model class): call PyTorch model (NOT AS STRING)
    model_name (str): same as model
    preprocess_mode (str): for dataset preprocessing
    output_path (str): directory to save model state_dict and predictions
    output_folder (str): name of the model state_dict to save
    use_fine_tuned (booelan): to load fine-tuned model
    fine_tuned_path (str): directory of fine-tuned model if use_fined_tuned = True

    vb_path (str): directory of predicted VB classification files from imaging analysis pipeline
    pt_dem_file (str): path to csv file with patient demographics data
    label_file (str): path to csv file with ground truth labels by subject ID

    batch_size (int): for preparing PyTorch dataset
    weight_init (str): weight initialization option [None, "xavier", "kaiming", "he"]
    learning_rate (float): learning rate of the optimizer
    learning_rate_decay_factor (float): decay factor of the learning rate for scheduler
    momentum (float): momentum for the optimizer
    epochs (int): max number of epochs for training
    threshold (float): for binary classification
    patience (int): max number of epochs before early stopping
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mode = Config.getstr("general", "mode")
        self.model_name = Config.getstr("general", "model_name")
        self.bert_mode = Config.getstr("bert", "bert_mode")
        self.output_path = Config.getstr("general", "output_path")
        self.output_folder = Config.getstr("general", "output_folder")
        self.save_name = Config.getstr("general", "save_name")
        self.use_fine_tuned = Config.getboolean("general", "use_fine_tuned")
        self.fine_tuned_path = Config.getstr("general", "fine_tuned_path")
        self.save_train_metrics = Config.getboolean("general", "save_train_metrics")
       
        self.encode_mode = Config.getstr("preprocess", "encode_mode")
        self.loss_function = Config.getstr("model", "loss_function")
        self.batch_size = Config.getint("model", "batch_size")
        self.weight_init = Config.getstr("model", "weight_init")
        self.learning_rate = Config.getfloat("model", "learning_rate", usefallback=True, fallback=0.001)
        self.learning_rate_decay_factor = Config.getfloat("model", "learning_rate_decay_factor", usefallback=True, fallback=0.5)
        self.weight_decay = Config.getfloat("model", "weight_decay", usefallback=True, fallback=0.01)
        self.momentum = Config.getfloat("model", "momentum")
        self.epochs = Config.getint("model", "epochs", usefallback=True, fallback=100)
        self.threshold = Config.getfloat("model", "threshold", usefallback=True, fallback=0.5)
        self.patience = Config.getint("model", "patience", usefallback=True, fallback=10)
        self.dropout_rate = Config.getfloat("model", "dropout_rate", usefallback=True, fallback=0.5)
        self.grad_clipping = Config.getfloat("model", "grad_clipping", usefallback=True, fallback=1.0)

        # make output_folder + output_path
        self.results_path = Config.get_resultfolder()

        from multimodal_dataloader import CustomCollate, MultimodalDataset
        from utils import SaveBestModel_F1, SaveBestModel_ValidationLoss
        if self.save_name:
            self.save_best_model = SaveBestModel_ValidationLoss(os.path.join(self.results_path, self.save_name))
        else:
            self.save_name = f"{self.model_name}_"+f"{self.loss_function}_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
            self.save_best_model = SaveBestModel_ValidationLoss(os.path.join(self.results_path, self.save_name))
        self.collate_fn = CustomCollate()
        self.__to_device()

        self.__train_dataset = None
        self.__validation_dataset = None
        self.__train_loader = None
        self.__validation_loader = None
        self.__predict_dataset = None
        self.__predict_loader = None
        
        if self.bert_mode == "discrete":  
            if self.mode == "train":
                print(f"******** Preparing discrete training dataset with {self.encode_mode} encoding ********")
                self.__train_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "train_path"), 
                                                         BERT_file=Config.getstr("bert", "train_discrete_file"), 
                                                         pt_dem_file=Config.getstr("patient", "pt_dem_file"), 
                                                         cols=uparams.BERT_ONEHOT_COLS, 
                                                         bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                         fx_labels=uparams.GT_LABELS_DICT)
                self.__validation_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "validation_path"), 
                                                              BERT_file=Config.getstr("bert", "validation_discrete_file"), 
                                                              pt_dem_file=Config.getstr("patient", "pt_dem_file"), 
                                                              cols=uparams.BERT_ONEHOT_COLS, 
                                                              bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                              fx_labels=uparams.GT_LABELS_DICT)
                
                self.__train_loader = DataLoader(self.__train_dataset, batch_size=1, shuffle=True, collate_fn=self.collate_fn)
                self.__validation_loader = DataLoader(self.__validation_dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

            elif self.mode == "predict":
                print(f"******** Preparing discrete prediction dataset with {self.encode_mode} encoding ********")
                self.__predict_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "predict_path"), 
                                                           BERT_file=Config.getstr("bert", "predict_discrete_file"), 
                                                           pt_dem_file=Config.getstr("patient", "pt_dem_file"), 
                                                           cols=uparams.BERT_ONEHOT_COLS, 
                                                           bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                           fx_labels=uparams.GT_LABELS_DICT)
            
                self.__predict_loader = DataLoader(self.__predict_dataset, batch_size=1, collate_fn=self.collate_fn)
        elif self.bert_mode == "cls":
            self.predict_BERT_cls_file = Config.getstr("bert", "predict_cls_file")
            if self.mode == "train":
                print(f"******** Preparing [CLS] training dataset with {self.encode_mode} encoding ********")
                self.__train_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "train_path"), 
                                                         BERT_file=Config.getstr("bert", "train_cls_file"), 
                                                         pt_dem_file=Config.getstr("patient", "pt_dem_file"), 
                                                         cols=uparams.BERT_ONEHOT_COLS, 
                                                         bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                         fx_labels=uparams.GT_LABELS_DICT)
                self.__validation_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "validation_path"), 
                                                              BERT_file=Config.getstr("bert", "validation_cls_file"), 
                                                              pt_dem_file=Config.getstr("patient", "pt_dem_file"), 
                                                              cols=uparams.BERT_ONEHOT_COLS, 
                                                              bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                              fx_labels=uparams.GT_LABELS_DICT)
                
                self.__train_loader = DataLoader(self.__train_dataset, batch_size=1, shuffle=True, collate_fn=self.collate_fn)
                self.__validation_loader = DataLoader(self.__validation_dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn)

            elif self.mode == "predict":
                print(f"******** Preparing [CLS] prediction dataset with {self.encode_mode} encoding ********")
                self.__predict_dataset = MultimodalDataset(vb_path=Config.getstr("vb", "predict_path"), 
                                                           BERT_file=Config.getstr("bert", "predict_discrete_file"), 
                                                           pt_dem_file=Config.getstr("bert", "predict_cls_file"), 
                                                           cols=uparams.BERT_ONEHOT_COLS, 
                                                           bert_dict=uparams.BERT_FRACTURE_DICT, 
                                                           fx_labels=uparams.GT_LABELS_DICT)
                
                self.__predict_loader = DataLoader(self.__predict_dataset, batch_size=1, collate_fn=self.collate_fn)

    def __to_device(self):
        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
        else:
            self.__device = torch.device("cpu")
    
    def init_model(self, dataloader):
        """
        Initialize a model with weight initialization by weight_init
        """
        
        if "Transformer" in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1],
                                in_channels=bert_batch.shape[2],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=256,
                                dropout_rate=self.dropout_rate)

                # Weight initialization
                if self.weight_init == "xavier":
                    model.apply(xavier_init)
                elif self.weight_init == "kaiming":
                    model.apply(kaiming_init)
                elif self.weight_init == "he":
                    model.apply(he_init)
                else:
                    pass

                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=int(patience/2), mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)
                break
        elif "NoReshape" not in self.model_name and "Transformer" not in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1]*bert_batch.shape[2],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=128,
                                dropout_rate=self.dropout_rate)

                # Weight initialization
                if self.weight_init == "xavier":
                    model.apply(xavier_init)
                elif self.weight_init == "kaiming":
                    model.apply(kaiming_init)
                elif self.weight_init == "he":
                    model.apply(he_init)
                else:
                    pass

                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=patience-5, mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)
                break
        elif "NoReshape" in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=128,
                                dropout_rate=self.dropout_rate)

                # Weight initialization
                if self.weight_init == "xavier":
                    model.apply(xavier_init)
                elif self.weight_init == "kaiming":
                    model.apply(kaiming_init)
                elif self.weight_init == "he":
                    model.apply(he_init)
                else:
                    pass

                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=int(patience/2), mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)
                break

        return model, criterion, optimizer, scheduler
    
    def load_model(self, dataloader, fine_tuned_path):
        """
        Load model with saved model weights if use_fine_tuned = True 
            from fine_tuned_path
        """
        if "Transformer" in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1],
                                in_channels=bert_batch.shape[2],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=256,
                                dropout_rate=self.dropout_rate)
                
                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=int(patience/2), mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)

                # load the fine_tuned_path checkpoint
                checkpoint = torch.load(fine_tuned_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                criterion.load_state_dict(checkpoint["criterion_state_dict"])
                break
        
        elif "NoReshape" not in self.model_name and "Transformer" not in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1]*bert_batch.shape[2],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=128,
                                dropout_rate=self.dropout_rate)

                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=patience-5, mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)

                # load the fine_tuned_path checkpoint
                checkpoint = torch.load(fine_tuned_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                criterion.load_state_dict(checkpoint["criterion_state_dict"])
                break

        elif "NoReshape" in self.model_name:
            for step, batch in enumerate(dataloader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]
                model = self.model(batch_size=1,
                                bert_dim=bert_batch.shape[1],
                                vb_dim=vb_batch.shape[1],
                                pt_dem_dim=pt_dem_batch.shape[1],
                                hidden_size=128,
                                dropout_rate=self.dropout_rate)

                model.to(self.__device)
                if self.loss_function == "bce":
                    criterion = nn.BCELoss()
                elif self.loss_function == "focal":
                    criterion = WeightedFocalLoss(alpha=Config.getfloat("model", "focal_loss_alpha"), 
                                                gamma=Config.getfloat("model", "focal_loss_gamma"))
                # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
                optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                # scheduler = ReduceLROnPlateau(optimizer, patience=int(patience/2), mode="max")
                scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=self.learning_rate_decay_factor)

                # load the fine_tuned_path checkpoint
                checkpoint = torch.load(fine_tuned_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                criterion.load_state_dict(checkpoint["criterion_state_dict"])
                break                
        
        return model, optimizer, criterion, scheduler

    def train_model(self):
        """
        Run training for N epochs and save the best model based on 
            validation loss to output_path

        Returns
        -------
        train_metrics (dict): with accuracy and losses
        """
        # Initialize or load fine-tuned model
        if "Losses" not in self.model_name:
            if not self.use_fine_tuned:
                print(f"******** Initializing model {self.model_name} ********")
                self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.init_model(self.__train_loader)
                self.__model.to(self.__device)

            else:
                print(f"******** Loading fine-tuned {self.model_name} model from {self.fine_tuned_path} ********")
                self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.load_model(self.__train_loader, self.fine_tuned_path)
                self.__model.to(self.__device)
        else:
            print(f"******** {self.model_name} is not applicable for this model training ********")
        
        print(f"******** Starting training with {self.loss_function} loss ********")
        self.__train_metrics = {}
        self.__train_preds = {}
        self.__validation_preds = {}
        self.__train_df = None
        self.__validation_df = None
        # self.__best_f1_epoch = 0
        self.__best_val_loss_epoch = 0
        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            ### Training
            self.__model.train()
            self.__train_image_ids = []
            self.__train_labels = []
            self.__train_scores = []
            self.__train_outputs = []
            self.__validation_image_ids = []
            self.__validation_scores = []
            self.__validation_labels = []
            self.__validation_outputs = []
            self.__running_loss = 0.0
            # Loop over batches in an epoch using DataLoader
            for step, batch in enumerate(self.__train_loader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]

                # 1. Forward pass
                y_pred = self.__model(bert_batch, vb_batch, pt_dem_batch)
                out = (y_pred>self.threshold).float()

                self.__train_scores.append(y_pred.detach().cpu().numpy().astype(float)[0])
                self.__train_outputs.append(out.cpu().numpy().astype(int)[0])
                self.__train_labels.append(label_batch.cpu().numpy().astype(int)[0])
                self.__train_image_ids.append(image_id[0])
            
                # 2. Calculate loss/accuracy
                loss = self.__criterion(y_pred, label_batch)
                
                # 3. Optimizer zero grad
                self.__optimizer.zero_grad()

                # 4. Gradient clipping
                if Config.getstr("model", "grad_clipping"):
                    nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
                
                # 5. Loss backwards
                loss.backward()
            
                # 6. Optimizer step
                self.__optimizer.step()
                self.__running_loss += loss.item()*label_batch.size(0)
                
            train_loss_epoch = self.__running_loss / len(self.__train_loader)

            tn, fp, fn, tp = confusion_matrix(self.__train_labels, self.__train_outputs).ravel()
            train_epoch_acc = accuracy_score(self.__train_labels, self.__train_outputs)*100
            train_precision = precision_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_recall = recall_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_f1 = f1_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_specificty = calculate_specificity(tn, fp)*100
            train_npv = calculate_npv(tn, fn)*100

            ### Evaluation
            self.__model.eval()
            self.__validation_running_loss = 0.0
            with torch.no_grad():
                for step, batch in enumerate(self.__validation_loader):
                    batch_size = len(batch[0][0])
                    validation_bert_batch = batch[0][0].to(self.__device)
                    validation_vb_batch = batch[0][1].to(self.__device)
                    validation_pt_dem_batch = batch[0][2].to(self.__device)
                    validation_label_batch = batch[0][3].to(self.__device)
                    validation_image_id = batch[1]

                    # 1. Forward pass
                    validation_pred = self.__model(validation_bert_batch, validation_vb_batch, validation_pt_dem_batch)
                    validation_out = (validation_pred>self.threshold).float()

                    self.__validation_scores.append(validation_pred.detach().cpu().numpy().astype(float)[0])
                    self.__validation_outputs.append(validation_out.cpu().numpy().astype(int)[0])
                    self.__validation_labels.append(validation_label_batch.cpu().numpy().astype(int)[0])
                    self.__validation_image_ids.append(validation_image_id[0])

                    # 2. Caculate loss/accuracy
                    validation_loss = self.__criterion(validation_pred, validation_label_batch)     
                    self.__validation_running_loss += validation_loss.item()*validation_label_batch.size(0)

                validation_loss_epoch = self.__validation_running_loss / len(self.__validation_loader)
                
                validation_tn, validation_fp, validation_fn, validation_tp = confusion_matrix(self.__train_labels, self.__train_outputs).ravel()
                validation_epoch_acc = accuracy_score(self.__validation_labels, self.__validation_outputs)*100
                validation_precision = precision_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                validation_recall = recall_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                validation_f1 = f1_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                validation_specificty = calculate_specificity(validation_tn, validation_fp)*100
                validation_npv = calculate_npv(validation_tn, validation_fn)*100

            # 7. Scheduler step
            self.__scheduler.step(validation_loss_epoch)

            # safe each epoch training metrics
            self.__train_metrics[epoch+1] = {'train_loss': train_loss_epoch, 'train_accuracy': train_epoch_acc, "train_specificity": train_specificty,
                                             "train_precision": train_precision, "train_recall": train_recall, "train_f1": train_f1, "train_npv": train_npv,
                                             "train_tp": tp, "train_fp": fp, "train_fn": fn, "train_tn": tn,
                                             'validation_loss': validation_loss_epoch, 'validation_accuracy': validation_epoch_acc, "validation_specificity": validation_specificty,
                                             "validation_precision": validation_precision, "validation_recall": validation_recall, "validation_f1": validation_f1, "validation_npv": validation_npv,
                                             "validation_tp": validation_tp, "validation_fp": validation_fp, "validation_fn": validation_fn, "validation_tn": validation_tn,
                                             "learning_rate": self.__scheduler.get_last_lr()[0]}
            
            self.__train_preds[epoch+1] = {"image_id": self.__train_image_ids, "score": self.__train_scores, "pred": self.__train_outputs, "label": self.__train_labels}
            self.__validation_preds[epoch+1] = {"image_id": self.__validation_image_ids, "score": self.__validation_scores, "pred": self.__validation_outputs, "label": self.__validation_labels}

            # early stopping if validation loss does not improve for patience
            # Update best validation loss
            # if validation_f1 > self.save_best_model.best_f1:
            if validation_loss_epoch < self.save_best_model.best_val_loss:
                # self.save_best_model(self.__model, self.__optimizer, self.__criterion, epoch, validation_f1)
                # self.__best_f1_epoch = epoch
                # save the trained model weights with the best validation loss
                self.save_best_model(self.__model, self.__optimizer, self.__criterion, epoch, validation_loss_epoch)
                self.__best_val_loss_epoch = epoch

                print(f"Epoch: {epoch+1} | Train Loss: {train_loss_epoch:.5f}, Train Accuracy: {train_epoch_acc:.3f}% | Validation loss: {validation_loss_epoch:.5f}, Validation Accuracy: {validation_epoch_acc:.3f}%")
                print(f"Training | Precision: {train_precision:.3f}%, Recall: {train_recall:.3f}%, F1 Score: {train_f1:.3f}%")
                print(f"Validation | Precision: {validation_precision:.3f}%, Recall: {validation_recall:.3f}%, F1 Score: {validation_f1:.3f}%")
                print("\n")

            # Early stopping check
            # if epoch - self.__best_f1_epoch >= self.patience:
            if epoch - self.__best_val_loss_epoch >= self.patience:
                print(f"******** Early stopping at epoch {epoch+1} ********")
                # print(f"*** Best F1 score = {self.save_best_model.best_f1:.3f}% at epoch {self.save_best_model.epoch+1} ***")
                print(f"*** Best validation loss = {self.save_best_model.best_val_loss:.3f} at epoch {self.save_best_model.epoch+1} ***")

                break
        
        # print last learning rate
        print(f"Final learning rate = {self.__scheduler.get_last_lr()[0]}")

        if self.save_train_metrics:
            self.save_training_metrics(self.__train_metrics)

        # saves results from the best epoch
        self.__train_df = pd.DataFrame.from_dict(data=self.__train_preds[self.__best_val_loss_epoch+1])
        self.save_predictions(self.__train_df, save_name=f"train_epoch{self.__best_val_loss_epoch+1}")

        self.__validation_df = pd.DataFrame.from_dict(data=self.__validation_preds[self.__best_val_loss_epoch+1])
        self.save_predictions(self.__validation_df, save_name=f"validation_epoch{self.__best_val_loss_epoch+1}")
            
    def train_model_losses(self):
        """
        Training for models that output individual predictions for loss backpropagation
        """
        # Instantiate the model
        if "Losses" in self.model_name:
            if not self.use_fine_tuned:
                print(f"******** Initializing model {self.model_name} ********")
                self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.init_model(self.__train_loader)
                self.__model.to(self.__device)

            elif self.use_fine_tuned:
                print(f"******** Loading fine-tuned {self.model_name} model from {self.fine_tuned_path} ********")
                self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.load_model(self.__train_loader, self.fine_tuned_path)
                self.__model.to(self.__device)
        else:
            print(f"******** {self.model_name} is not applicable for this model training ********")

        print(f"******** Starting training with {self.loss_function} loss ********")
        self.__train_metrics = {}
        self.__train_preds = {}
        self.__validation_preds = {}
        self.__train_df = None
        self.__validation_df = None
        self.__best_val_loss_epoch = 0
        # self.__best_f1_epoch = 0
        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            ### Training
            self.__model.train()
            self.__train_image_ids = []
            self.__train_labels = []
            self.__train_scores = []
            self.__train_outputs = []
            self.__validation_image_ids = []
            self.__validation_scores = []
            self.__validation_labels = []
            self.__validation_outputs = []
            self.__running_loss = 0.0 
            # Loop over batches in an epoch using DataLoader
            for step, batch in enumerate(self.__train_loader):
                batch_size = len(batch[0][0])
                bert_batch = batch[0][0].to(self.__device)
                vb_batch = batch[0][1].to(self.__device)
                pt_dem_batch = batch[0][2].to(self.__device)
                label_batch = batch[0][3].to(self.__device)
                image_id = batch[1]

                # 1. Forward pass
                y_pred, out1, out2, out3 = self.__model(bert_batch, vb_batch, pt_dem_batch)
                out = (y_pred>self.threshold).float()

                self.__train_scores.append(y_pred.detach().cpu().numpy().astype(float)[0])
                self.__train_outputs.append(out.cpu().numpy().astype(int)[0])
                self.__train_labels.append(label_batch.cpu().numpy().astype(int)[0])
                self.__train_image_ids.append(image_id[0])
            
                # 2. Calculate loss/accuracy
                # Calculate individual losses
                loss1 = self.__criterion(out1, label_batch)
                loss2 = self.__criterion(out2, label_batch)
                loss3 = self.__criterion(out3, label_batch)
                loss_output = self.__criterion(y_pred, label_batch)

                # Combine losses
                loss = loss1 + loss2 + loss3 + loss_output
                self.__running_loss += loss.item()*label_batch.size(0)
                        
                # 3. Optimizer zero grad
                self.__optimizer.zero_grad()

                # 4. Gradient clipping
                if Config.getstr("model", "grad_clipping"):
                    nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
                
                # 5. Backpropagation
                loss.backward()
            
                # 6. Optimizer step
                self.__optimizer.step()
                self.__running_loss += loss.item()*label_batch.size(0)
                
            train_loss_epoch = self.__running_loss / len(self.__train_loader)

            tn, fp, fn, tp = confusion_matrix(self.__train_labels, self.__train_outputs).ravel()
            train_epoch_acc = accuracy_score(self.__train_labels, self.__train_outputs)*100
            train_precision = precision_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_recall = recall_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_f1 = f1_score(self.__train_labels, self.__train_outputs, zero_division=0)*100
            train_specificity = tn / (tn+fp)

            ### Evaluation
            self.__model.eval()
            self.__validation_running_loss = 0.0
            with torch.no_grad():
                for step, batch in enumerate(self.__validation_loader):
                    batch_size = len(batch[0][0])
                    validation_bert_batch = batch[0][0].to(self.__device)
                    validation_vb_batch = batch[0][1].to(self.__device)
                    validation_pt_dem_batch = batch[0][2].to(self.__device)
                    validation_label_batch = batch[0][3].to(self.__device)
                    validation_image_id = batch[1]

                    # 1. Forward pass
                    validation_pred, validation_out1, validation_out2, validation_out3 = self.__model(validation_bert_batch, validation_vb_batch, validation_pt_dem_batch)
                    validation_out = (validation_pred>self.threshold).float()

                    self.__validation_scores.append(validation_pred.detach().cpu().numpy().astype(float)[0])
                    self.__validation_outputs.append(validation_out.cpu().numpy().astype(int)[0])
                    self.__validation_labels.append(validation_label_batch.cpu().numpy().astype(int)[0])
                    self.__validation_image_ids.append(validation_image_id[0])

                    # 2. Caculate loss/accuracy
                    # Calculate individual losses
                    validation_loss1 = self.__criterion(validation_out1, validation_label_batch)
                    validation_loss2 = self.__criterion(validation_out2, validation_label_batch)
                    validation_loss3 = self.__criterion(validation_out3, validation_label_batch)
                    validation_loss_output = self.__criterion(validation_pred, validation_label_batch)

                    # Combine losses
                    validation_loss = validation_loss1 + validation_loss2 + validation_loss3 + validation_loss_output
                    self.__validation_running_loss += validation_loss.item()*validation_label_batch.size(0)

                validation_loss_epoch = self.__validation_running_loss / len(self.__validation_loader)

                validation_tn, validation_fp, validation_fn, validation_tp = confusion_matrix(self.__validation_labels, self.__validation_outputs).ravel()
                validation_epoch_acc = accuracy_score(self.__validation_labels, self.__validation_outputs)*100
                validation_precision = precision_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                validation_recall = recall_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                validation_f1 = f1_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100

            # 7. Scheduler step
            self.__scheduler.step(validation_loss_epoch)

            # safe each epoch training metrics
            self.__train_metrics[epoch+1] = {'train_loss': train_loss_epoch, 'train_accuracy': train_epoch_acc, 
                                             "train_precision": train_precision, "train_recall": train_recall, "train_f1": train_f1, 
                                             "train_tp": tp, "train_fp": fp, "train_fn": fn, "train_tn": tn,
                                             'validation_loss': validation_loss_epoch, 'validation_accuracy': validation_epoch_acc, 
                                             "validation_precision": validation_precision, "validation_recall": validation_recall, "validation_f1": validation_f1, 
                                             "validation_tp": validation_tp, "validation_fp": validation_fp, "validation_fn": validation_fn, "validation_tn": validation_tn,
                                             "learning_rate": self.__scheduler.get_last_lr()[0]}
            
            self.__train_preds[epoch+1] = {"image_id": self.__train_image_ids, "score": self.__train_scores, "pred": self.__train_outputs, "label": self.__train_labels}
            self.__validation_preds[epoch+1] = {"image_id": self.__validation_image_ids, "score": self.__validation_scores, "pred": self.__validation_outputs, "label": self.__validation_labels}

            # early stopping if validation loss does not improve for patience
            # Update best F1 epoch
            # if validation_f1 > self.save_best_model.best_f1:
            if validation_loss_epoch < self.save_best_model.best_val_loss:
                # self.save_best_model(self.__model, self.__optimizer, self.__criterion, epoch, validation_f1)
                # save the trained model weights with best validation loss
                # self.__best_f1_epoch = epoch
                self.save_best_model(self.__model, self.__optimizer, self.__criterion, epoch, validation_loss_epoch)
                self.__best_val_loss_epoch = epoch

                print(f"Epoch: {epoch+1} | Train Loss: {train_loss_epoch:.5f}, Train Accuracy: {train_epoch_acc:.3f}% | Validation loss: {validation_loss_epoch:.5f}, Validation Accuracy: {validation_epoch_acc:.3f}%")
                print(f"Training | Precision: {train_precision:.3f}%, Recall: {train_recall:.3f}%, F1 Score: {train_f1:.3f}%")
                print(f"Validation | Precision: {validation_precision:.3f}%, Recall: {validation_recall:.3f}%, F1 Score: {validation_f1:.3f}%")
                print("\n")

            # Early stopping check
            # if epoch - self.__best_f1_epoch >= self.patience:
            if epoch - self.__best_val_loss_epoch >= self.patience:
                print(f"******** Early stopping at epoch {epoch+1}******** ")
                # print(f"*** Best F1 score = {self.save_best_model.best_f1:.3f}% at epoch {self.save_best_model.epoch+1} ***")
                print(f"*** Best validation loss = {self.save_best_model.best_val_loss:.3f} at epoch {self.save_best_model.epoch+1} ***")
                break
        
        # print last learning rate
        print(f"Final learning rate = {self.__scheduler.get_last_lr()[0]}")

        if self.save_train_metrics:
            self.save_training_metrics(self.__train_metrics)
        
        # saves results from the best epoch
        # self.__train_df = pd.DataFrame.from_dict(data=self.__train_preds[self.__best_f1_epoch+1])
        self.__train_df = pd.DataFrame.from_dict(data=self.__train_preds[self.__best_val_loss_epoch+1])
        self.save_predictions(self.__train_df, save_name=f"train_epoch{self.__best_val_loss_epoch+1}")

        # self.__validation_df = pd.DataFrame.from_dict(data=self.__validation_preds[self.__best_f1_epoch+1])
        self.__validation_df = pd.DataFrame.from_dict(data=self.__validation_preds[self.__best_val_loss_epoch+1])
        self.save_predictions(self.__validation_df, save_name=f"validation_epoch{self.__best_val_loss_epoch+1}")

    def evaluate_model(self, dataloader):
        self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.load_model(self.__predict_loader, self.fine_tuned_path)
        self.__model.to(self.__device)

        self.__loss_lst = []
        self.__acc_lst = []

        self.__preds = []
        self.__scores = []
        self.__image_ids = []
        self.__running_loss = 0.0

        if "Losses" not in self.model_name:
            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    batch_size = len(batch[0][0])
                    bert_batch = batch[0][0].to(self.__device)
                    vb_batch = batch[0][1].to(self.__device)
                    pt_dem_batch = batch[0][2].to(self.__device)
                    label_batch = batch[0][3].to(self.__device)
                    image_id = batch[1]
                    # 1. Forward pass
                    y_pred = self.__model(bert_batch, vb_batch, pt_dem_batch)
                    out = (y_pred>self.threshold).float()

                    self.__image_ids.append(image_id[0])
                    self.__scores.append(y_pred.detach().cpu().numpy().astype(float)[0])
                    self.__preds.append(out.detach().cpu().numpy().astype(int)[0])

                    # 2. Calculate loss/accuracy
                    self.__loss = self.__criterion(y_pred.to(self.__device), label_batch.to(self.__device))       
                    self.__running_loss += self.__loss.item()*label_batch.size(0)

                loss_total = self.__running_loss / len(dataloader)
                self.__loss_lst.append(loss_total)

                epoch_acc = accuracy_score(self.__validation_labels, self.__validation_outputs)*100
                precision = precision_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                recall = recall_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                f1 = f1_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                self.__acc_lst.append(epoch_acc)

                print(f"Precision: {precision:.3f}%, Recall: {recall:.3f}%, F1 Score: {f1:.3f}%")

            return loss_total, epoch_acc, precision, recall, f1
        
        elif "Losses" in self.model_name:
            with torch.no_grad():
                for step, batch in enumerate(dataloader):
                    batch_size = len(batch[0][0])
                    bert_batch = batch[0][0].to(self.__device)
                    vb_batch = batch[0][1].to(self.__device)
                    pt_dem_batch = batch[0][2].to(self.__device)
                    label_batch = batch[0][3].to(self.__device)
                    image_id = batch[1]
                    # 1. Forward pass
                    y_pred, out1, out2, out3 = self.__model(bert_batch, vb_batch, pt_dem_batch)
                    out = (y_pred>self.threshold).float()

                    self.__image_ids.append(image_id[0])
                    self.__scores.append(y_pred.detach().cpu().numpy().astype(float)[0])
                    self.__preds.append(out.detach().cpu().numpy().astype(int)[0])

                # 2. Calculate loss/accuracy
                    self.__loss = self.__criterion(y_pred.to(self.__device), label_batch.to(self.__device))       
                    self.__running_loss += self.__loss.item()*label_batch.size(0)

                loss_total = self.__running_loss / len(dataloader)
                self.__loss_lst.append(loss_total)

                epoch_acc = accuracy_score(self.__validation_labels, self.__validation_outputs)*100
                precision = precision_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                recall = recall_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                f1 = f1_score(self.__validation_labels, self.__validation_outputs, zero_division=0)*100
                self.__acc_lst.append(epoch_acc)

                # self.__evaluation_metrics[epoch+1] = {'evaluation_loss': self.__loss_lst, 'evaluation_accuracy': self.__acc_lst}

                print(f"Epoch: {epoch+1} | Loss: {loss_total:.5f}, Accuracy: {epoch_acc:.3f}%")
                print(f"Precision: {precision:.3f}%, Recall: {recall:.3f}%, F1 Score: {f1:.3f}%")

    def predict(self):
        """
        Saves the predictions for a dataset using a fine-tuned model to output_folder
        """
        if not self.fine_tuned_path:
            raise ValueError(f"Fine-tuned path for model is not defined")

        # load model from fine-tuned
        self.__model, self.__criterion, self.__optimizer, self.__scheduler = self.load_model(self.__predict_loader, self.patience, self.fine_tuned_path)
        self.__model.to(self.__device)
        self.__preds_df = None
        self.__preds = []
        self.__scores = []
        self.__image_ids = []
        if "Losses" not in self.model_name:
            with torch.no_grad():
                for step, batch in enumerate(self.__predict_loader):
                    bert_batch = batch[0][0].to(self.__device)
                    vb_batch = batch[0][1].to(self.__device)
                    pt_dem_batch = batch[0][2].to(self.__device)
                    # label_batch = batch[0][3]
                    image_id = batch[1]

                    y_pred = self.__model(bert_batch, vb_batch, pt_dem_batch)
                    out = (y_pred>self.threshold).float()

                    self.__image_ids.append(image_id[0])
                    self.__scores.append(y_pred.detach().cpu().numpy().astype(float)[0])
                    self.__preds.append(out.detach().cpu().numpy().astype(int)[0])

            assert len(self.__image_ids) == len(self.__preds), "Number of Image IDs do not match the number of predictions"
            self.__preds_df = pd.DataFrame(data={"image_id": self.__image_ids, "score": self.__scores, "predicted_fx": self.__preds})
            self.save_predictions(self.__preds_df)

        if "Losses" in self.model_name:
            with torch.no_grad():
                for step, batch in enumerate(self.__predict_loader):
                    bert_batch = batch[0][0]
                    vb_batch = batch[0][1]
                    pt_dem_batch = batch[0][2]
                    # label_batch = batch[0][3]
                    image_id = batch[1]

                    pred, out1, out2, out3 = self.__model(bert_batch.to(self.__device), vb_batch.to(self.__device), pt_dem_batch.to(self.__device))
                    out = (pred>self.threshold).float()

                    self.__image_ids.append(image_id[0])
                    self.__preds.append(out.detach().cpu().numpy().astype(int)[0])

            assert len(self.__image_ids) == len(self.__preds), "Number of Image IDs do not match the number of predictions"
            self.__preds_df = pd.DataFrame(data={"image_id": self.__image_ids, "predicted_fx": self.__preds})
            self.save_predictions(self.__preds_df)
        print(f"Predictions saved to {self.results_path}")
                
    def save_predictions(self, preds_df, save_name=None):
        """
        Save predictions to a csv file

        Parameters
        ----------
        preds_df (Pandas DataFrame)
        save_name (str)
        """
        if save_name:
            preds_df.to_csv(os.path.join(self.results_path, f"{self.save_name}_"+f"{save_name}_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')+f"_predictions.csv"), index=False)
        else:    
            preds_df.to_csv(os.path.join(self.results_path, f"{self.save_name}_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')+f"_predictions.csv"), index=False)

    def save_training_metrics(self, train_metrics):
        """
        Saves training metrics to csv file

        Parameters
        ----------
        train_metrics (dict): from train_model() or train_model_losses()
        """
        self.__train_metrics_df = pd.DataFrame.from_dict(train_metrics, orient="index")
        self.__train_metrics_df.to_csv(os.path.join(self.results_path, f"{self.save_name}_"+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')+f"_training_metrics.csv"), index=False)