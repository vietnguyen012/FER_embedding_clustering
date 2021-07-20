import torch
from torch import nn, optim
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pytorch_metric_learning import losses, miners, samplers, trainers
from global_embedding_space import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
import pytorch_metric_learning.utils.logging_presets as LP
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import logging
import record_keeper
import pytorch_metric_learning
import torch.nn.functional as F
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)
from model import Vgg,MLP
from data import get_data,get_writer
torch.manual_seed(0)
from torchvision import transforms
from utils import get_list_label
from train_with_classifier import TrainWithClassifier
from fuzzy_c_means import FCM

mu, st = 0, 255
train_transform = transforms.Compose([
                transforms.Resize(40),
                transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=40),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((mu,),(st,))
        ])
test_transform = transforms.Compose([
        transforms.Resize(40),
        transforms.ToTensor(),
        transforms.Normalize((mu,),(st,))
    ])

train_data, test_data = get_data(train_transform = train_transform,test_transform=test_transform)

writer = get_writer()

label_items = get_list_label(train_data)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trunk = Vgg().to(device)
trunk.load_state_dict(torch.load('VGGNet')["params"])
trunk.eval()


data = torch.rand(128,1,40,40).to(device)
writer.add_graph(trunk,data)
writer.close()


trunk_output_size = trunk.lin3.in_features
print(trunk_output_size)
trunk.lin3 = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk).to(device)
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))
classifier = torch.nn.DataParallel(MLP([64, 7]).to(device))
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
classifier_optimizer =torch.optim.Adam(classifier.parameters(), lr=0.0001, weight_decay=0.0001)
embedder_optimizer =torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)
trunk_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trunk_optimizer, mode='max', factor=0.75, patience=5, verbose=True)
embedder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='max', factor=0.75, patience=5, verbose=True)
classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedder_optimizer, mode='max', factor=0.75, patience=5, verbose=True)
trunk.eval()
embedder.eval()
classifier.eval()

summary(trunk, input_size=(1, 40, 40), batch_size=128)
summary(classifier, input_size=(1,4096), batch_size=128)

num_classes=7
embedding_size=64

loss = losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64)
classification_loss = torch.nn.CrossEntropyLoss()

miner = miners.MultiSimilarityMiner(epsilon=0.1)

sampler = samplers.MPerClassSampler(label_items, m=16, length_before_new_iter=len(train_data))

batch_size = 32
num_epochs = 300
fc = FCM(num_classes,num_classes).to(device)
fc.register_centroids()

models = {"trunk": trunk,"embedder":embedder,"classifier": classifier,'fcm':fc}
optimizers = {"trunk_optimizer": trunk_optimizer,  "embedder_optimizer": embedder_optimizer,"classifier_optimizer": classifier_optimizer}
loss_funcs = {"metric_loss": loss,"classifier_loss": classification_loss}
mining_funcs = {"tuple_miner": miner}
loss_weights = {"metric_loss": 1, "classifier_loss": 0.5}
schedulers = {'trunk_scheduler_by_plateau':trunk_scheduler,'embedder_scheduler_by_plateau':embedder_scheduler}

record_keeper, _, _ = LP.get_record_keeper("classifier_logs", "example_tensorboard")
dataset_dict = {"val": test_data}
model_folder = "saved_classifier_models"

hooks = LP.HookContainer(record_keeper,
    record_group_name_prefix=None,
    primary_metric="precision_at_1",
    validation_split_name="val",
    save_models=True)



tester = GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook,
                                            dataloader_num_workers = 2,
                                            accuracy_calculator=AccuracyCalculator(k="max_bin_count"))


end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                            dataset_dict,
                                            model_folder,
                                            test_interval = 1)
trainer = TrainWithClassifier(models,
                        optimizers,
                        batch_size,
                        loss_funcs,
                        mining_funcs,
                        train_data,
                        fcm = fc,
                        sampler=sampler,
                        lr_schedulers=schedulers,
                        loss_weights = loss_weights,
                        dataloader_num_workers = 2,
                        end_of_iteration_hook = hooks.end_of_iteration_hook,
                        end_of_epoch_hook = end_of_epoch_hook)

trainer.train(num_epochs=num_epochs)
