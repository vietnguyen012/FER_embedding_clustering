from pytorch_metric_learning.utils import common_functions as c_f
from metric_only import MetricLossOnly
import numpy as np
from fuzzy_c_means import FCM
import torch
import torch.nn.functional as F

class TrainWithClassifier(MetricLossOnly):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings = self.compute_embeddings(data)
        logits = self.maybe_get_logits(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )
        self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels)
    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            u = self.fcm(logits)
            new_centers = self.fcm.recalc_centers(logits,u)
            return self.loss_funcs["classifier_loss"](
                c_f.to_device(u,logits), c_f.to_device(labels,logits)
            ) + F.mse_loss(self.fcm.centers,new_centers)
        return 0

    def maybe_get_logits(self, embeddings):
        if (
            self.models.get("classifier", None)
            and self.loss_weights.get("classifier_loss", 0) > 0
        ):
            return self.models["classifier"](embeddings)
        return None

    def modify_schema(self):
        self.schema["models"].keys += ["classifier"]
        self.schema["loss_funcs"].keys += ["classifier_loss"]
