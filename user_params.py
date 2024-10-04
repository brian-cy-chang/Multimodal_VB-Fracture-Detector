"""
User-defined dictionaries for multimodal data preprocessing
"""
import numpy as np

# BERT discrete events dictionary for label encoding
BERT_FRACTURE_DICT = {"spine": 8, "UE": 1, "LE": 2, "chest": 9, "pelvis": 4, "head": 5, "unspecified": 6, "hip": 7, "no_mention": 3, np.NaN: 10,
                      "incidental": 1, "traumatic": 2, "no_mention": 3, np.NaN: 10,
                      "absent": 2, "present": 1, "possible": 4, "no_mention": 3, np.NaN: 10,
                    }

# columns for one-hot encoding
BERT_ONEHOT_COLS = ['image_id', 'FractureAnatomy_LE', 'FractureAnatomy_UE',
                    'FractureAnatomy_chest', 'FractureAnatomy_head', 'FractureAnatomy_hip',
                    'FractureAnatomy_no_mention', 'FractureAnatomy_pelvis',
                    'FractureAnatomy_spine', 'FractureCause_no_mention',
                    'FractureCause_traumatic', 'FractureAssertion_absent',
                    'FractureAssertion_no_mention', 'FractureAssertion_possible',
                    'FractureAssertion_present', 'FractureCause_incidental']

# patient demographics dictionaries for label encoding
RACE_DICT = {
            "White": 0,
            "Asian": 1,
            "Pacific Islander": 2,
            "American India or Alaska Native": 7,
            "Black or African American": 4,
            "Native Hawaiian or Other Pacific Islander": 5,
            "Unknown": 6,
            "Multiple Races": 3,
            np.NaN: 8, 
            "nan": 8
            }

SEX_DICT = {
            "M": 0,
            "F": 1,
            np.NaN: 2
            }

MULTIPLE_RACES_DICT = {"White;Asian;": 0, 
                        "Asian;Native Hawaiian or Other Pacific Islander;": 1, 
                        "American India or Alaska Native;White;" :2, "Black or African American;American India or Alaska Native;": 3, 
                        "White;Native Hawaiian or Other Pacific Islander;": 4, 
                        np.NaN: 5, 
                        "nan": 5}
        
ETHNICITY_DICT = {"Hispanic or Latino": 0, 
                  "Not Hispanic or Latino": 1, 
                  "Unknown": 2, 
                  np.NaN: 3, 
                  "nan": 3}

# ground truth annotations dictionary to binarize m2ABQ fracture classification
GT_LABELS_DICT = {'Normal': 0, 
                  'Non-fracture Deformity': 0, 
                  "mABQ0 Fracture\n(< 20% height loss)": 0, 
                  "mABQ1 Fracture\n(20%-25% height loss)": 1, 
                  "mABQ2 Fracture\n(25%-40% height loss)": 1, 
                  "mABQ3 Fracture\n(> 40% height loss)": 1, 
                  "Non-diagnostic": 2}