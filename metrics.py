import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
import numpy as np

class KLLoss(nn.Module):
    def __init__(self,reduction = 'batchmean'):
        super(KLLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction)

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred,'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        y_pred = F.log_softmax(y_pred,dim=1)
        y_true = F.softmax(y_true, dim=1)
        out = self.kl(y_pred, y_true)
        return out

class CCLoss(nn.Module):
    def __init__(self):
        super(CCLoss, self).__init__()

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred, 'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        """
        计算均值
        """
        mean_y_pred = torch.mean(y_pred,dim=1).unsqueeze(1)
        mean_y_true = torch.mean(y_true,dim=1).unsqueeze(1)
        # 计算协方差
        cov_xy = torch.mean((y_pred - mean_y_pred) * (y_true - mean_y_true),dim=1)
        # 计算标准差
        std_y_pred = torch.std(y_pred,dim=1)
        std_y_true = torch.std(y_true,dim=1)
        #计算cc
        cc_output = cov_xy / (std_y_pred * std_y_true)
        # cc_output = cc_output.cpu().numpy()
        cc_output = cc_output[~torch.isnan(cc_output)]
        return cc_output.mean()

class SIMLoss(nn.Module):
    def __init__(self):
        super(SIMLoss, self).__init__()

    def forward(self,y_pred,y_true):
        y_pred = rearrange(y_pred, 'b h w -> b (h w)')
        y_true = rearrange(y_true, 'b h w -> b (h w)')
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True)
        y_true = y_true / y_true.sum(dim=1, keepdim=True)
        sims = torch.min(y_pred, y_true).sum(dim=1)
        # sims = sims.cpu().numpy()
        sims = sims[~torch.isnan(sims)]
        return sims.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp

        return torch.mean(focal_loss)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        weight = torch.tensor(weight).cuda()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss

class Fuse_metrics(nn.Module):
    def __init__(self):
        super(Fuse_metrics, self).__init__()
        self.mse_metrics = nn.MSELoss()
        self.mae_metrics = nn.L1Loss()
        self.KL = KLLoss()
        self.CC = CCLoss()
        self.SIM = SIMLoss()

    def forward(self,output_map, label_map):
        mse_val = self.mse_metrics(100*output_map,100*label_map)
        mae_val = self.mae_metrics(100 * output_map, 100 * label_map)
        KL_vaule = self.KL(100 * output_map,100 * label_map)
        CC_vaule = self.CC(100 * output_map,100 * label_map)
        SIM_vaule = self.SIM(100 * output_map,100 * label_map)
        rmse = torch.sqrt(mse_val)
        return mse_val,mae_val,rmse,KL_vaule,CC_vaule,SIM_vaule


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLossWrapper, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))  # 初始化权重参数

    def forward(self, regression_loss, classification_loss):
        # 计算带有不确定性权重的损失函数
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])

        # 组合损失函数
        loss = precision1 * regression_loss + precision2 * classification_loss + self.log_vars[0] + self.log_vars[1]
        return loss

# 假设你有两个任务：回归和分类
# multi_task_loss = MultiTaskLossWrapper(num_tasks=2)
# total_loss = multi_task_loss(regression_loss, classification_loss)



# if __name__ == '__main__':
#     y_pred = torch.randn((4, 256,256))
#     y_true = torch.randn((4, 256, 256))
#     y_pred = torch.abs(y_pred)
#     y_true = torch.abs(y_true)
#     KLloss = KLLoss()
#     output = KLloss(y_pred,y_true)
#     pass
