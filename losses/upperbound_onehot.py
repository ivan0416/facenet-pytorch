import torch
import torch.nn as nn
import numpy as np


class UpperBound_onehot(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(UpperBound_onehot, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu   
        self.label = torch.LongTensor(self.num_classes, 1)
        for i in range(self.num_classes):
            self.label[i] = i
        self.centers= nn.Parameter(torch.zeros(self.num_classes, self.feat_dim).scatter_(1, self.label, 1))
        print(self.centers)
        
        

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        unmask = labels.ne(classes.expand(batch_size, self.num_classes))
        near_dist, near_ind = torch.min(distmat, 1) #最近距離，最近index
        acc = 0
        for i in range(batch_size):
            if mask[i, near_ind[i]]:
                acc += 1
        i_dist = distmat * mask.float() # 各x與centroid距離
        k_dist = distmat * unmask.float() # 各x與其他centroid距離

        loss = torch.sum(i_dist, dim=1) 
        #print(loss, '\n')
        #print(torch.sum(k_dist, dim=1)/(3*(self.num_classes-1)))
        loss -= torch.sum(k_dist, dim=1)/(3*(self.num_classes-1))
        #loss *= (batch_size/self.num_classes-1) * (batch_size/self.num_classes) 
        loss = loss.sum()/batch_size

        return loss, acc
