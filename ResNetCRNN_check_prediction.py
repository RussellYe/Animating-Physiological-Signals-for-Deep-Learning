
# This code is implemented based on the project from https://github.com/HHTseng/video-classification/tree/master/ResNetCRNN


import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# This line is to fix the error of
# OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
# When plotting
os.environ['KMP_DUPLICATE_LIB_OK']='True'


data_path = "Data pre-processing/Video cuts collection/Video frames collection/training_split1/"    # define ginger bread data path
class_name_path = '3_classes.pkl'
save_model_path = "save_model/"


# use same encoder CNN saved!
CNN_fc_hidden1, CNN_fc_hidden2, CNN_fc_hidden3 = 1024, 768, 666
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.1       # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 3  # 12          # number of target category
batch_size = 25
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 26, 1


with open(class_name_path, 'rb') as f:
    action_names = pickle.load(f)   # load UCF101 actions names

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)


actions = []
fnames = os.listdir(data_path)

# fnames
all_names = []
for f in fnames:
    loc1 = f.find('_')
    actions.append(f[:(loc1-1)])
    all_names.append(f)


# list all data files
all_X_list = all_names              # all video file names
all_y_list = labels2cat(le, actions)    # all video labels
if __name__ == '__main__':
    # data loading parameters
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.486, 0.457, 0.405], std=[0.228, 0.223, 0.226])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # reset data loader
    all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # batch_size = batch_size, shuffle= True
    all_data_loader = data.DataLoader(Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)


    # reload CRNN model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, fc_hidden3 =CNN_fc_hidden3, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch10.pth')))
    rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch10.pth')))
    print('CRNN model reloaded!')


    # make all video predictions by reloaded model
    print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))
    all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)


    # write in pandas dataframe
    df = pd.DataFrame(data={'filename': fnames, 'y': cat2labels(le, all_y_list), 'y_pred': cat2labels(le, all_y_pred)})
    df.to_pickle("./Ginger_bread.pkl")  # save pandas dataframe
    # pd.read_pickle("./all_videos_prediction.pkl")
    print('video prediction finished!')




