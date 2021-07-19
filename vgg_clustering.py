from model import Vgg,MLP
from pytorch_metric_learning.utils import common_functions
from data import get_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap
import torch
from torchvision import transforms
import logging
import numpy as np
from matplotlib import pyplot as plt
torch.manual_seed(0)
from fcm_k_means import visualizer_hook
from utils import fcm_k_means,get_list_label

if __name__ == '__main__':
    device = 'cuda'
    evaluate_on_fcm=True
    mu, st = 0, 255
    test_transform = transforms.Compose([
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    train_transform = test_transform
    train_data,test_data = get_data(train_transform = train_transform,test_transform=test_transform)

    label_test = get_list_label(test_data)
    label_train = get_list_label(train_data)

    test_loader = DataLoader(test_data,batch_size=128,shuffle=False,num_workers=2)
    train_loader = DataLoader(train_data,batch_size=128,shuffle=False,num_workers=2)

    output_test_features = []
    output_train_features = []

    #extract feature for train data
    vgg = Vgg().to(device)
    vgg.load_state_dict(torch.load('VGGNet')["params"])
    vgg.eval()
    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(train_loader)):
            datas,labels = datas.to(device),labels.to(device)
            bs,ncrops,c,h,w = datas.shape
            datas = datas.view(-1,c,h,w)
            outputs = vgg(datas)
            outputs = outputs.view(bs,ncrops,-1)
            outputs = torch.sum(outputs,dim=1)/ncrops
            output_train_features.append(outputs)
    output_train_features = torch.cat(output_train_features,dim=0).detach().cpu().numpy()

    #extract feature for test data
    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(test_loader)):
            datas,labels = datas.to(device),labels.to(device)
            bs,ncrops,c,h,w = datas.shape
            datas = datas.view(-1,c,h,w)
            outputs = vgg(datas)
            outputs = outputs.view(bs,ncrops,-1)
            outputs = torch.sum(outputs,dim=1)/ncrops
            output_test_features.append(outputs)

    output_test_features = torch.cat(output_test_features,dim=0).detach().cpu().numpy()
    remaped_label = fcm_k_means(output_train_features,output_test_features,label_train,label_test,evaluate_on_fcm)
    visualize_embedding_pred_n_gt(output_test_features,label_test,remaped_label)
