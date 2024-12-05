import argparse

from config import Config
from model import Multimodal_VB_Fracture_Detector
from multimodal.joint_fusion import *
from multimodal.late_fusion import *
from utils import set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--general_mode', default=None)
    parser.add_argument('--general_seed', type=int, default=None)
    parser.add_argument('--general_output_path', default=None)
    parser.add_argument('--general_output_folder', default=None)
    parser.add_argument('--general_model_name', default=None)
    parser.add_argument('--general_use_fine_tuned', action='store_true', default=None)
    parser.add_argument('--general_fine_tuned_path', default=None)
    parser.add_argument('--general_save_name', default=None)
    parser.add_argument('--general_save_train_metrics', action='store_true', default=None)
    parser.add_argument('--bert_bert_mode', default=None)
    parser.add_argument('--bert_discrete_max_length_pad', default=None)
    parser.add_argument('--bert_cls_max_length_pad', default=None)
    parser.add_argument('--bert_train_discrete_file', default=None)
    parser.add_argument('--bert_validation_discrete_file', default=None)
    parser.add_argument('--bert_predict_discrete_file', default=None)
    parser.add_argument('--bert_train_cls_file', default=None)
    parser.add_argument('--bert_validation_cls_file', default=None)
    parser.add_argument('--bert_predict_cls_file', default=None)
    parser.add_argument('--vb_train_path', default=None)
    parser.add_argument('--vb_validation_path', default=None)
    parser.add_argument('--vb_predict_path', default=None)
    parser.add_argument('--vb_vb_max_length_pad', default=None)
    parser.add_argument('--patient_pt_dem_file', default=None)
    parser.add_argument('--labels_groundtruth_file', default=None)
    parser.add_argument('--preprocess_encode_mode', default=None)
    parser.add_argument('--model_loss_function', default=None)
    parser.add_argument('--model_batch_size', type=int, default=None)
    parser.add_argument('--model_weight_init', type=float, default=None)
    parser.add_argument('--model_epochs', type=int, default=None)
    parser.add_argument('--model_focal_loss_alpha', type=float, default=None)
    parser.add_argument('--model_focal_loss_gamma', type=float, default=None)
    parser.add_argument('--model_learning_rate', type=float, default=None)
    parser.add_argument('--model_learning_rate_decay_factor', type=float, default=None)
    parser.add_argument('--model_weight_decay', type=float, default=None)
    parser.add_argument('--model_momentum', type=float, default=None)
    parser.add_argument('--model_threshold', type=float, default=None)
    parser.add_argument('--model_patience', type=int, default=None)
    parser.add_argument('--model_dropout_rate', type=float, default=None)
    parser.add_argument('--model_grad_clipping', type=float, default=None)

    args = parser.parse_args()
    Config.init(args)

    set_seed(Config.getint("general", "seed"))

    if Config.getstr("general", "model_name") == "JointFusion_CNN":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    if Config.getstr("general", "model_name") == "JointFusion_CNN_NoReshape":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN_NoReshape)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_FC":
        model = Multimodal_VB_Fracture_Detector(JointFusion_FC)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
            model.save_predictions()
    elif Config.getstr("general", "model_name") == "JointFusion_FC_Losses":
        model = Multimodal_VB_Fracture_Detector(JointFusion_FC_Losses)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model_losses()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
            model.save_predictions()
    elif Config.getstr("general", "model_name") == "JointFusion_FC_Attention_BeforeConcatenation":
        model = Multimodal_VB_Fracture_Detector(JointFusion_FC_Attention_BeforeConcatenation)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_FC_Attention_AfterConcatenation":
        model = Multimodal_VB_Fracture_Detector(JointFusion_FC_Attention_AfterConcatenation)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_CNN_Attention_AfterConcatenation":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN_Attention_AfterConcatenation)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_CNN_Attention_BeforeConcatenation":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN_Attention_BeforeConcatenation)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "LateFusion_FC":
        model = Multimodal_VB_Fracture_Detector(LateFusion_FC_Average_Losses)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "LateFusion_CNN":
        model = Multimodal_VB_Fracture_Detector(LateFusion_CNN_Average_Losses)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_CNN_Losses_NoReshape":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN_Losses_NoReshape)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model_losses()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_CNN_Losses_Reshape":
        model = Multimodal_VB_Fracture_Detector(JointFusion_CNN_Losses_Reshape)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model_losses()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_Transformer":
        model = Multimodal_VB_Fracture_Detector(JointFusion_Transformer)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
    elif Config.getstr("general", "model_name") == "JointFusion_ALiBi_Transformer":
        model = Multimodal_VB_Fracture_Detector(JointFusion_ALiBi_Transformer)
        if Config.getstr("general", "mode").lower() == "train" :
            model.train_model()
        elif Config.getstr("general", "mode").lower() == "predict":
            model.predict()
            
if __name__ == '__main__':
    main()