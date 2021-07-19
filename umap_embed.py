from data import get_data
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
import skfuzzy as fuzz
from fcm import visualizer_hook
sns.set(style='white', context='poster')

train_data,test_data = get_data()
data_train = []
target_train = []
data_test = []
target_test = []
for (test_item,label_test) in test_data:
    data_test.append(test_item.view(-1,40*40))
    target_test.append(label_test)

for (train_item,label_train) in train_data:
    data_train.append(train_item.view(-1,40*40))
    target_train.append(label_train)

data_train = torch.cat(data_train).numpy()
target_train = torch.tensor(target_train).numpy()
data_test = torch.cat(data_test).numpy()
target_test = torch.tensor(target_test).numpy()
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral',
         'sad', 'surprise']



embedding_train = umap.UMAP().fit_transform(data_train,y=target_train)

fig, ax = plt.subplots(1, figsize=(14, 10))
plt.scatter(*embedding_train.T, s=0.1, c=target_train, cmap='Spectral', alpha=1.0)
plt.setp(ax, xticks=[], yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(7))
cbar.set_ticklabels(classes)
plt.title('Fer 2013 Embedded train via UMAP using Labels');
plt.show()

embedding_test = umap.UMAP().fit_transform(data_test,y=target_test)
fig,ax = plt.subplots(1,figsize=(14,10))
plt.scatter(*embedding_test.T,s=0.3,c=target_test,cmap='Spectral',alpha=1.0)
plt.setp(ax,xticks=[],yticks=[])
cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
cbar.set_ticks(np.arange(7))
cbar.set_ticklabels(classes)
plt.title('Fer 2013 test Embedded via UMAP')
plt.show()
