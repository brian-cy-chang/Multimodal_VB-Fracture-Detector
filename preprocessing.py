import os
import sys
import time
import datetime
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import Config
import user_params as uparams

class preprocess(Dataset):
    """
    This class preprocesses outputs from the BERT-based models,
        imaging analysis pipeline, and other structured EHR data
        for a PyTorch DataLoader

    Parameters
    ----------
    VB_path (str): vb_classifcation2 output path from imaging analysis pipeline
    pt_dem_file (str): patient demographics csv file path 
    labels_file (str): patient level ground truth annotations csv file path
    BERT_discrete_file (pkl): ensemble majority vote discrete events
    BERT_CLS_file (pkl): ensemble average CLS embeddings
    """
    def __init__(self):
        self.__to_device()

    def __to_device(self):

        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
        else:
            self.__device = torch.device("cpu")

    def pt_onehot_encode(self, pt_dem_file):
        """
        One-hot encode patient demographics

        """
        # get patient demographics
        self.__pt_dem_df = pd.read_csv(pt_dem_file)

        # Normalize column names to title format
        # pt_dem_df.columns = pt_dem_df.columns.str.lower()

        # drop columns if they exist
        self.__pt_dem_df = self.__pt_dem_df.drop(columns=["InstitutionName", "Manufacturer", "ManufacturerModelName"], errors="ignore")

        # rename columns 
        self.__pt_dem_df = self.__pt_dem_df.rename(columns={"AgeCalc": "Age", "Filename_x": "image_id", "Multiple Races": "MultipleRaces"})

        # get subject ID from the image ID
        self.__pt_dem_df["subject_id"] = self.__pt_dem_df.image_id.apply(lambda x: x[:5])

        # one hot encode patient demographics
        self.__onehot_encoded_pt_dem = pd.get_dummies(self.__pt_dem_df, columns = ['Race', 'PatientSex', 'MultipleRaces', 'Ethnicity'])
        self.__onehot_encoded_pt_dem = self.__onehot_encoded_pt_dem.drop_duplicates()

        return self.__onehot_encoded_pt_dem

    def pt_label_encode(self, pt_dem_file):
        """
        Label encode patient demographics with custom dictionary

        Parameters
        ----------
        pt_file (csv): directory to csv of patient demographics with columns
            PatientID | Sex | Age | Race | MultipleRaces | Ethnicity

        Returns
        -------
        pt_tensors (PyTorch tensor): label encoded patient demographic float tensors
        """
        self.__pt_dem_df = pd.read_csv(pt_dem_file)

        # drop columns if they exist
        self.__pt_dem_df = self.__pt_dem_df.drop(columns=["InstitutionName", "Manufacturer", "ManufacturerModelName"], errors="ignore")

        # get subject ID from the image ID
        self.__pt_dem_df["subject_id"] = self.__pt_dem_df.image_id.apply(lambda x: x[:5])

        # rename columns
        self.__pt_dem_df = self.__pt_dem_df.rename(columns={"AgeCalc": "Age", "Filename_x": "image_id", "Multiple Races": "MultipleRaces"})

        # label encode each column
        self.__pt_dem_df = self.__pt_dem_df.replace(uparams.RACE_DICT)
        self.__pt_dem_df = self.__pt_dem_df.replace(uparams.SEX_DICT)
        self.__pt_dem_df = self.__pt_dem_df.replace(uparams.MULTIPLE_RACES_DICT)
        self.__pt_dem_df = self.__pt_dem_df.replace(uparams.ETHNICITY_DICT)

        self.__label_encoded_pt_dem = self.__pt_dem_df.drop_duplicates()

        return self.__label_encoded_pt_dem
    
    def pt_dem_to_tensor(self, combined_dict):
        # combined_dict = self.combine_modalities()
        
        pt_dem_lst = []
        for image_id in sorted(combined_dict):
            df = combined_dict[image_id]["pt_dem"]
            pt_dem_lst.append(df)

        pt_dem_df = pd.concat(pt_dem_lst)
        pt_dem_tensors = torch.tensor(np.array(pt_dem_df).astype(np.float32), dtype=torch.float32)

        return pt_dem_tensors

    def get_vb_files(self, vb_path):
        """
        Returns a list of image IDs and file paths for VB classification outputs from the imaging analysis pipeline
        """
        classification_csv = [f for f in listdir(vb_path) if isfile(join(vb_path, f))and f.endswith('.csv')]
        classification_image_ids = [f.split(".")[0] for f in classification_csv]
        classification_files = sorted([join(vb_path, f) for f in classification_csv])

        return classification_image_ids, classification_files
    
    def combine_modalities(self, vb_path, BERT_file, pt_dem_file):
        """
        Combine the modalities into a dictionary by subject ID key
        """
        classification_image_ids, classification_files = self.get_vb_files(vb_path)
        self.__bert_pt_events = self.load_bert_events(BERT_file)
        if Config.getstr("preprocess", "encode_mode") == "onehot":
            self.__pt_dem_df = self.pt_onehot_encode(pt_dem_file)
        elif Config.getstr("preprocess", "encode_mode") == "label":
            self.__pt_dem_df = self.pt_label_encode(pt_dem_file)
        # combine into one dictionary by subject ID
        self.__combined_dict = {}
        if Config.getstr("bert", "bert_mode") == "discrete":
            for image_id in sorted(self.__pt_dem_df.image_id.unique().tolist()):
                slc = self.__pt_dem_df.loc[self.__pt_dem_df.image_id == image_id]
                if slc.image_id.values[0] in classification_image_ids:
                    vb_file = next((s for s in classification_files if image_id in s), None)
                    if vb_file and slc['subject_id'].values[0] in self.__bert_pt_events.keys():
                        self.__combined_dict[image_id] = {"BERT_discrete": pd.DataFrame.from_records(self.__bert_pt_events[slc['subject_id'].values[0]]["FxEvents"]), 
                                                          "vb": pd.read_csv(vb_file), 
                                                          "pt_dem": slc.drop(columns=["image_id", "subject_id"]).drop_duplicates()}
        elif Config.getstr("bert", "bert_mode") == "cls":
            for image_id in sorted(self.__pt_dem_df.image_id.unique().tolist()):
                slc = self.__pt_dem_df.loc[self.__pt_dem_df.image_id == image_id]
                if slc.image_id.values[0] in classification_image_ids:
                    vb_file = next((s for s in classification_files if image_id in s), None)
                    if vb_file and slc['subject_id'].values[0] in self.__bert_pt_events.keys():
                        self.__combined_dict[image_id] = {"BERT_cls": self.__bert_pt_events[slc['subject_id'].values[0]], 
                                                          "vb": pd.read_csv(vb_file), 
                                                          "pt_dem": slc.drop(columns=["image_id", "subject_id"]).drop_duplicates()}
                        
        assert len(self.__combined_dict) > 0, "Data preprocessing returned empty dataset. Please double check the input files."
        return self.__combined_dict
                    
    def load_bert_events(self, BERT_file):
        """
        Load the patient-level BERT-EE ensemble outputs
        """
        with open(BERT_file, "rb") as f:
            self.__bert_pt_events = pickle.load(f)

        return self.__bert_pt_events

    def get_max_vb_to_tensor(self, combined_dict):
        """
        Get max vb score
        """
        self.__vb_scores = []
        for image_id in sorted(combined_dict):
            df = combined_dict[image_id]["vb"]
            max_score = np.max(df.ensemble_averaging_predicted_prob)
            self.__vb_scores.append(max_score)
        self.__vb_tensors = torch.tensor(self.__vb_scores, dtype=torch.float32)

        return self.__vb_tensors
    
    def get_vbs_to_tensor(self, combined_dict):
        """
        Get max vb score
        """
        self.__vb_scores = []
        for image_id in sorted(combined_dict):
            df = combined_dict[image_id]["vb"]
            # max_score = np.max(df.ensemble_averaging_predicted_prob)
            self.__vb_scores.append(df.ensemble_averaging_predicted_prob.values)
        # self.__vb_tensors = torch.tensor(self.__vb_scores, dtype=torch.float32)

        return self.__vb_scores
    
    def get_gt_labels_to_tensor(self, label_file, ID_list, fx_labels=uparams.GT_LABELS_DICT):
        UW_annotations_df = pd.read_csv(label_file)
        UW_annotations_df = UW_annotations_df.drop_duplicates()
        UW_annotations_df = UW_annotations_df.rename(columns={"mABQ label": "m2ABQClass"})

        """
        Map the m2ABQ classification to binary fracture vs. no fracture
        """
        self.__UW_annotations_dict = {}
        for image_id in UW_annotations_df.Image.unique().tolist():
            slc = UW_annotations_df.loc[UW_annotations_df.Image == image_id]
            slc['fracture_label'] = slc.m2ABQClass.map(fx_labels)

            if any(slc.fracture_label == 1):
                self.__UW_annotations_dict[image_id] = {'fracture_label': 1}
            elif all(slc.fracture_label == 0):
                self.__UW_annotations_dict[image_id] = {'fracture_label': 0}
            else:
                self.__UW_annotations_dict[image_id] = {'fracture_label': 0}

        self.__label_lst = []
        for image_id in sorted(ID_list):
            if image_id in self.__UW_annotations_dict.keys():
                self.__label_lst.append(self.__UW_annotations_dict[image_id]["fracture_label"])

        self.__label_tensors = torch.tensor(np.array(self.__label_lst).astype(np.float32), dtype=torch.float32)

        return self.__label_tensors