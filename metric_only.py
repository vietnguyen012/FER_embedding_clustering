from base_trainer import BaseTrainer
import torch

class MetricLossOnly(BaseTrainer):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        # bs,ncrops,c,h,w = data.shape
        # data = data.view(-1,c,h,w)
        embeddings = self.compute_embeddings(data)
        # embeddings = embeddings.view(bs,ncrops,-1)
        # embeddings = torch.sum(embeddings,dim=1)/ncrops
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )

    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            return self.loss_funcs["metric_loss"](embeddings, labels, indices_tuple)
        return 0
