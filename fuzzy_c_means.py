import torch
import torch.nn.functional as F
import torch.nn as nn


class FCM(nn.Module):
    _EPS = 1e-12
    def __init__(self,n_clusters,n_features,m=1.7):
        super(FCM,self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.m = m
        self.centers = torch.rand((self.n_clusters,self.n_features)).to('cuda')
        self.last_u = None
    def set_centers(self,new_centers):
        self.centers = new_centers
    def register_centroids(self):
         init_centers = self.centers
         del self.centers
         self.register_parameter('centers',torch.nn.Parameter(init_centers))
    @staticmethod
    def _cdist(x1,x2):
        # x1_norm = x1.pow(2).sum(dim=-1,keepdim=True)
        # x2_norm = x2.pow(2).sum(dim=-1,keepdim=True)
        # res = x2^2 -2(x1 @ x2) + x1^2
        res =  torch.cdist(x1,x2,p=2).squeeze(0)
        res = res.clamp_min(FCM._EPS)
        return res
    def recalc_centers(self,x,u):
        um = u**self.m
        v = torch.einsum('mi,mj->ij',um,x)
        v/=um.sum(dim=0).clamp_min_(FCM._EPS).unsqueeze(1)
        return v
    def forward(self,x):
        d = FCM._cdist(x,self.centers)
        u = d**(-2/(self.m-1))
        self.last_u = u/u.sum(dim=1,keepdim=True)
        return self.last_u
