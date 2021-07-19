from model import Vgg,MLP
from pytorch_metric_learning.utils import common_functions
from data import get_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
torch.manual_seed(0)
from torchvision import transforms
evaluate_on_fcm = True
from utils import fcm_k_means,get_list_label

if __name__ == '__main__':

    device = 'cuda'
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

    trunk = Vgg().to(device)
    trunk_output_size = trunk.lin3.in_features
    trunk.lin3 = common_functions.Identity()
    trunk.load_state_dict(torch.load('example_saved_models/trunk_best68.pth'))
    embedder = MLP([trunk_output_size, 7]).to(device)
    embedder.load_state_dict(torch.load('example_saved_models/embedder_best68.pth'))

    models = {"trunk":trunk,"embedder":embedder}
    test_loader = DataLoader(test_data,batch_size=128,shuffle=False,num_workers=2)
    train_loader = DataLoader(train_data,batch_size=128,shuffle=False,num_workers=2)

    output_test_features = []
    output_train_features = []
    models["embedder"].eval()
    models["trunk"].eval()


    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(train_loader)):
            datas,labels = datas.to(device),labels.to(device)
            bs,ncrops,c,h,w = datas.shape
            datas = datas.view(-1,c,h,w)
            outputs = models["embedder"](models["trunk"](datas))
            outputs = outputs.view(bs,ncrops,-1)
            outputs = torch.sum(outputs,dim=1)/ncrops
            output_train_features.append(outputs)
    output_train_features = torch.cat(output_train_features,dim=0).detach().cpu().numpy()

    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(test_loader)):
            datas,labels = datas.to(device),labels.to(device)
            bs,ncrops,c,h,w = datas.shape
            datas = datas.view(-1,c,h,w)
            outputs = models["embedder"](models["trunk"](datas))
            outputs = outputs.view(bs,ncrops,-1)
            outputs = torch.sum(outputs,dim=1)/ncrops
            output_test_features.append(outputs)


    output_test_features = torch.cat(output_test_features,dim=0).detach().cpu().numpy()
    remapped_labels = fcm_k_means(output_train_features,output_test_features,label_train,label_test,evaluate_on_fcm)
    visualize_embedding_pred_n_gt(output_test_features,label_test,remaped_label)
