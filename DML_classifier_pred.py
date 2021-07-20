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
from utils import fcm_k_means,get_list_label,visualize_embedding_pred_n_gt
from fuzzy_c_means import FCM
from remap_label import remap_labels
from sklearn.metrics import precision_score,f1_score,recall_score,confusion_matrix

if __name__ == '__main__':

    device = 'cuda'
    mu, st = 0, 255
    test_transform = transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize((mu,),(st,))
        ])

    train_transform = test_transform
    train_data,test_data = get_data(train_transform = train_transform,test_transform=test_transform)

    label_test = get_list_label(test_data)
    label_train = get_list_label(train_data)

    trunk = Vgg().to(device)
    trunk_output_size = trunk.lin3.in_features
    trunk.lin3 = common_functions.Identity()
    trunk.load_state_dict(torch.load('saved_classifier_models/trunk_best5.pth'))
    trunk = torch.nn.DataParallel(trunk)
    embedder = MLP([trunk_output_size, 64]).to(device)
    embedder.load_state_dict(torch.load('saved_classifier_models/embedder_best5.pth'))
    embedder =  torch.nn.DataParallel(embedder)
    classifier = MLP([64, 7]).to(device)
    classifier.load_state_dict(torch.load('saved_classifier_models/classifier_best5.pth'))
    classifier = torch.nn.DataParallel(classifier)
    num_classes = 7
    fc = FCM(num_classes,num_classes).to(device)
    fc.register_centroids()
    fc.load_state_dict(torch.load('saved_classifier_models/fcm_best5.pth'))
    fc = torch.nn.DataParallel(fc)
    trunk.eval()
    embedder.eval()
    classifier.eval()
    fc.eval()

    models = {"trunk":trunk,"embedder":embedder,"classifier":classifier,'fcm':fc}
    test_loader = DataLoader(test_data,batch_size=128,shuffle=False,num_workers=2)
    train_loader = DataLoader(train_data,batch_size=128,shuffle=False,num_workers=2)

    output_test_features = []
    output_train_features = []

    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(train_loader)):
            datas,labels = datas.to(device),labels.to(device)
            outputs = models['classifier'](models["embedder"](models["trunk"](datas)))
            u = models['fcm'](outputs)
            output_train_features.append(u)
    output_train_features = torch.cat(output_train_features,dim=0).detach().cpu().numpy()

    test_pred = []
    gts = []
    with torch.no_grad():
        for i,(datas,labels) in enumerate(tqdm(test_loader)):
            datas,labels = datas.to(device),labels.to(device)
            outputs = models['classifier'](models["embedder"](models["trunk"](datas)))
            u = models['fcm'](outputs)
            output_test_features.append(u)
            test_pred.append(u.argmax(dim=-1).squeeze(0))
            gts.append(labels)
    gts = torch.cat(gts).detach().cpu().numpy()
    test_pred = torch.cat(test_pred).detach().cpu().numpy()
    output_test_features = torch.cat(output_test_features,dim=0).detach().cpu().numpy()

    accuracy,remapped_labels = remap_labels(test_pred,gts)
    print('accuracy clustering:',accuracy)
    print("Precision: %2.6f" % precision_score(gts, remapped_labels, average='micro'))
    print("Recall: %2.6f" % recall_score(gts, remapped_labels, average='micro'))
    print("F1 Score: %2.6f" % f1_score(gts, remapped_labels, average='micro'))
    print("Confusion Matrix:\n", confusion_matrix(gts, remapped_labels), '\n')
    visualize_embedding_pred_n_gt(output_test_features,label_test,output_train_features,label_train,remapped_labels)
