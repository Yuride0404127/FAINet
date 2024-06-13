import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_iou
from KD_model_3.KD_loss import BalancedCrossEntropyLoss, CriterionAT, CriterionCWD

from einops import rearrange

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

IOU = pytorch_iou.IOU(size_average=True)

class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.KLD = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, x_student, x_teacher):

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all


class Adaptation_loss(nn.Module):
    def __init__(self, in_channel=1):
        super(Adaptation_loss, self).__init__()
        self.bce = BCELOSS()

    def forward(self, x_s, x_t1, x_t2):
        loss_t1 = self.bce(x_s, torch.sigmoid(x_t1)) + IOU(x_s, torch.sigmoid(x_t1))
        # loss_t2 = self.bce(x_s, torch.sigmoid(x_t2)) + IOU(x_s, torch.sigmoid(x_t2))

        loss = loss_t1 #+ loss_t2

        return loss


class At_loss(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(At_loss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

class Frequence_Attention(nn.Module):
    def __init__(self, in_channel, reduction = 16):
        super(Frequence_Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Linear = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_q_max = torch.max(x, dim=3, keepdim=True)[0]
        x_k_mean = torch.mean(x, dim=2, keepdim=True)
        x_2d = rearrange(self.avgpool(x), 'b c h w -> b (c h w)')
        x_v = rearrange(self.Linear(x_2d), 'b c -> b c 1 1')
        x_qk = x_q_max * x_k_mean
        x_qkv = x_qk * x_v
        x_out = x_qkv + x

        return x_out

"""
exp1  share Frequence_Attention
exp2  setting hpyer_parameters
version1:HCL     0.0609
version2:ATLoss  
version3:DICE    
version4:KLD  
version5:3fa     0.0610   
"""


class Frequence_Base_loss(nn.Module):
    def __init__(self, in_channel):
        super(Frequence_Base_loss, self).__init__()
        self.kld = KLDLoss()
        self.fa = Frequence_Attention(in_channel)



    def forward(self, x_s, x_t1, x_t2):
        x_fa_s = self.fa(x_s)
        x_fa_t1 = self.fa(x_t1)
        # x_fa_t2 = self.fa(x_t2)
        loss1 = self.kld(x_fa_s, x_fa_t1)
        # loss2 = self.kld(x_fa_s, x_fa_t2)

        loss = loss1 #+ loss2
        # loss = loss2

        return loss

class Hypersphere_linear(nn.Module):
    def __init__(self, in_channel, out_channel, m=4, phiflag=True):
        super(Hypersphere_linear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.Tensor(in_channel, out_channel))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.fc = nn.Linear(in_channel, in_channel)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.re_fc = nn.Linear(512, in_channel * hw * hw)

        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x:4 * x ** 3 - 3 * x,
            lambda x:8 * x ** 4 -8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x):
        x = self.maxpool(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x_linear = self.fc(x)

        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        x_len = x_linear.pow(2).sum(1).pow(0.5)
        w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x_linear.mm(ww)
        cos_theta = cos_theta / x_len.view(-1, 1) / w_len.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_m_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            theta_m = theta * self.m
            phi_theta = 1 - theta_m ** 2 / math.factorial(2) + theta_m ** 4 / math.factorial(4) - \
                        theta_m ** 6 /math.factorial(6) + theta_m ** 8 / math.factorial(8) - theta_m ** 9 / math.factorial(9)

            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        phi_theta = phi_theta * x_len.view(-1, 1)
        return phi_theta

class Hypersphere_loss(nn.Module):
    def __init__(self, batchsize, in_channel):
        super(Hypersphere_loss, self).__init__()
        self.Hyper_Linear_t1 = Hypersphere_linear(in_channel, in_channel)
        self.Hyper_Linear_t2 = Hypersphere_linear(in_channel, in_channel)
        self.Hyper_Linear_s = Hypersphere_linear(in_channel, in_channel)
        self.CRLoss = ContrastiveLoss(batchsize)


    def forward(self, x_s, x_t1, x_t2):
        x_s_linear = F.softmax(self.Hyper_Linear_s(x_s), dim=1)
        x_t1_linear = F.softmax(self.Hyper_Linear_t1(x_t1), dim=1)
        # x_t2_linear = F.softmax(self.Hyper_Linear_t2(x_t2), dim=1)

        loss_t1 = self.CRLoss(x_s_linear, x_t1_linear)
        # loss_t2 = self.CRLoss(x_s_linear, x_t2_linear)

        loss = loss_t1 #+ loss_t2

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__ == "__main__":
    teacher_1 = torch.randn(6, 512, 64, 64).cuda()
    teacher_2 = torch.randn(6, 512, 64, 64).cuda()
    student = torch.randn(6, 512, 64, 64).cuda()

    teacher_stage_1 = torch.randn(6, 1, 256, 256).cuda()
    teacher2_stage_1 = torch.randn(6, 1, 256, 256).cuda()
    student_stage_1 = torch.randn(6, 1, 256, 256).cuda()

    fbloss = Frequence_Base_loss(512).cuda()
    loss = fbloss(student, teacher_1, teacher_2)

    AdLoss = Adaptation_loss()
    loss_AL = AdLoss(student_stage_1, teacher_stage_1, teacher2_stage_1)


    # FA = Frequence_Attention(512).cuda()
    # out = FA(teacher_1)

    print("loss_AL: ", loss_AL)
    # print("out", out.shape)