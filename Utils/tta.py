import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy.stats import ttest_rel

def calculate_entropy(prob, logits): # Take average for a batch of samples, do not consider single sample
    # prob: (B, C) tensor where B is the batch size and C is the number of classes
    log_prob = F.log_softmax(logits, dim=1)  # Apply log-softmax for numerical stability
    entropy = -torch.sum(prob * log_prob, dim=1)  # Compute entropy for each row
    average_entropy = entropy.mean()  # Compute the average entropy across the batch
    return average_entropy.item()
def entropy_minimization_loss(logits):

    probabilities = F.softmax(logits, dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)

    # 计算平均熵作为损失
    loss = torch.mean(entropy)

    return loss

def compute_energy(logits):
    # log-sum-up, energy function
    # logits are the raw output, not softmaxed
    energy = -torch.logsumexp(logits + 1e-8, dim=-1)
    return energy


def select_source_like_samples_and_compute_average_energy(logits, alpha=0.5):

    energy = compute_energy(logits)

    k = int(alpha * energy.size(0)) # how many % data are used as source like data
    source_like_indices = torch.argsort(energy)[:k]
    average_energy = torch.mean(energy[source_like_indices])

    return average_energy


def compute_sfea_loss(logits):

    target_energy = compute_energy(logits)
    average_source_energy = select_source_like_samples_and_compute_average_energy(logits)

    sfea_loss = torch.mean(torch.log(1 + torch.exp(target_energy - average_source_energy + 1e-8)))

    return sfea_loss


def compute_S(x_logits, W, y_hat):
    W = W.T  # W = [65, 512]
    # print(W.shape)
    # f_hat_y = torch.matmul(W.T, W[:, y_hat]) # weights, argmax
    # cos_similarity = F.cosine_similarity(x_logits.unsqueeze(0), f_hat_y.unsqueeze(0), dim=1)
    f_hat_y = torch.matmul(W.T, W[:, y_hat]) # [C, B] x [B, 1] = [C, 1]
    # print(f_hat_y.shape)
    f_hat_y = f_hat_y.T  # 转置为 [B, K]
    # print(x_logits, f_hat_y)
    # 计算余弦相似度
    # print(x_logits.shape, f_hat_y.shape, W.shape)
    cos_similarity = F.cosine_similarity(F.softmax(x_logits, dim=-1), F.softmax(f_hat_y, dim=-1), dim=1)  # 形状为 [B]

    return cos_similarity


def compute_LCS_loss(x_logits, W, C0 = 0.5):

    _, y_hat = torch.max(x_logits,dim=1)
    confidence = torch.softmax(x_logits, dim=1)
    confidence = confidence.gather(1, y_hat.unsqueeze(1)).squeeze(1)  # 提取每个样本对应伪标签的置信度，形状为 [B]

    w_x = torch.exp(confidence - C0) # C(x)
    # print(w_x.shape)
    S_x = compute_S(x_logits, W, y_hat)

    LCS_loss = torch.mean(w_x * (1 - S_x) * (confidence >= C0).float())
    # C(x)(1-S(x)) * I(conf(x) >= C0)

    return LCS_loss

def compute_LCS_loss_train(x_logits, y_hat, W, C0 = 0.5):

    confidence = torch.softmax(x_logits + 1e-6, dim=1)
    confidence = confidence.gather(1, y_hat.unsqueeze(1)).squeeze(1)

    w_x = torch.exp(confidence - C0) + 1e-6
    # print(w_x.shape)
    S_x = compute_S(x_logits, W, y_hat) + 1e-6

    LCS_loss = torch.mean(w_x * (1 - S_x) * (confidence >= C0).float())

    return LCS_loss



def Difference_Confidence_Accuracy(outputs, labels):
    loss = 0
    # we consider all predicted labels for training, rather than consider the highest conf score predictions.
    B, C = outputs.size() # [B, C], outputs are logits.
    # s = F.softmax(outputs, dim=-1)
    # c = F.one_hot(labels, num_classes=C).float()
    # loss = torch.mean(torch.abs(c - s)) # 1/N * |c_i - s_i|
    init_output = torch.softmax(outputs, dim=1)  # [B, C]
    init_conf, init_pred_labels = torch.max(init_output, dim=1)  # [B, 1], [B,1]
    loss = torch.abs(init_conf.mean() - (init_pred_labels == labels).float().mean())

    #     # # MDCA
    # outputs = torch.softmax(outputs, dim=1)
    # batch, classes = outputs.shape
    # for c in range(classes):
    #     avg_count = (labels == c).float().mean()
    #     avg_conf = torch.mean(outputs[:, c])
    #     loss += torch.abs(avg_conf - avg_count)
    # loss /= C
    return loss

def t_test(B, C):
    assert B.shape == C.shape, "B和C的形状必须相同"

    # 计算差值矩阵
    D_diff = B - C
    print(D_diff)
    # 将差值矩阵展平为一维数组
    D_diff_flat = D_diff.flatten()

    # 进行配对样本t检验
    t_stat, p_val = ttest_rel(D_diff_flat, np.zeros_like(D_diff_flat))

    print("t统计量:", t_stat)
    print("p值:", p_val)

    # 根据p值做出决策
    alpha = 0.05
    if p_val < alpha:
        print("拒绝原假设，两个模型的预测概率存在显著差异")
    else:
        print("不能拒绝原假设，两个模型的预测概率没有显著差异")


def semi_split(Image_F, Text_F, Label_F):

    torch.manual_seed(42)

    # 获取总样本数
    total_samples = Image_F.size(0)

    # 计算分割点
    split_point = int(total_samples * 0)

    # 生成随机索引
    indices = torch.randperm(total_samples)

    # 分割索引
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # 根据索引分割数据
    train_data = {
        'Image_F': Image_F[train_indices],
        'Text_F': Text_F[train_indices],
        'Label_F': Label_F[train_indices]
    }

    val_data = {
        'Image_F': Image_F[val_indices],
        'Text_F': Text_F[val_indices],
        'Label_F': Label_F[val_indices]
    }

    return train_data, val_data


def decomposition(Tensor, topk=128):
    u,s,v = torch.svd(Tensor)
    u = u[:, :topk]
    s = s[:topk]
    v = v[:, :topk]
    return u, s, v.T

