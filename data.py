from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
class FerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample



def load_data(path='fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping



def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label



def get_data(train_transform=None,test_transform=None):
    if train_transform is None and test_transform is None:
        train_transform = transforms.Compose([transforms.Resize(40),
                                              transforms.RandomCrop(40, padding=4, padding_mode='reflect'),
                                              # transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                              transforms.RandomAffine(
                                                  degrees=30,
                                                  translate=(0.01, 0.12),
                                                  shear=(0.01, 0.03),
                                              ),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()])

        test_transform = transforms.Compose([transforms.Grayscale(),
                                                 transforms.Resize(40),
                                                 transforms.ToTensor()])
    else:
        train_transform = train_transform
        test_transform = test_transform
    path='fer2013/fer2013.csv'
    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

    train_data =FerDataset(xtrain, ytrain, train_transform)
    test_data = FerDataset(xtest, ytest, test_transform)
    return train_data,test_data


def get_writer():
    writer = SummaryWriter('runs/facial_expression')
    return writer



if __name__ == '__main__':
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    classes = ('angry', 'disgust', 'fear', 'happy', 'neutral',
            'sad', 'surprise')
    train_data,test_data = get_data()
    writer = get_writer()
    datas = []
    labels = []
    for data,label in train_data:
      datas.append(data)
      labels.append(label)
    data_items = torch.stack(datas)
    label_items = torch.tensor(labels)

    def select_n_random(data,labels,n=500):
      assert len(data) == len(labels)
      perm = torch.randperm(len(data))
      return data[perm][:n],labels[perm][:n]
    images, labels = select_n_random(data_items,label_items)
    class_labels = [classes[lab] for lab in labels]
    features = images.view(-1,40*40)
    writer.add_embedding(features,metadata=class_labels,label_img = images)
    writer.close()
