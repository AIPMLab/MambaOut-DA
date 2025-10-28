import os
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from Utils.transform import get_transform
from Utils.data_utils import ImageList
from Utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from loss.mmd_loss import MMD_loss, confidence_weighted_mmd_loss, LMMDLoss, CORAL, BNM
from .Metrics import AverageMeter
from .Valid import valid
from Utils.tta import compute_LCS_loss_train,compute_LCS_loss, entropy_minimization_loss


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.learning_rate * (1. + 0.0003 * float(x)) ** (-0.75))
    return scheduler

def self_supervised_pseudo_label_loss(tgt_logits, conf_threshold=0.7):
    """
    计算基于高置信度伪标签的自监督损失。

    Args:
        tgt_logits (torch.Tensor): 目标域样本的原始模型输出 (logits)。形状: [B, C]
        conf_threshold (float): 用于筛选高置信度伪标签的阈值。

    Returns:
        torch.Tensor: 计算得到的交叉熵损失。如果所有样本置信度都低，返回0。
    """
    # 1. 将 logits 转换为概率分布
    softmax_out = nn.Softmax(dim=1)(tgt_logits)

    # 2. 获取每个样本的最大概率（置信度）和对应的类别索引（伪标签）
    max_probs, pseudo_labels = torch.max(softmax_out, dim=1)

    # 3. 根据置信度阈值筛选高置信度样本
    high_conf_indices = max_probs > conf_threshold

    # 4. 如果有高置信度样本，则计算损失
    if high_conf_indices.sum() > 0:
        # 获取高置信度样本的 logits 和伪标签
        high_conf_logits = tgt_logits[high_conf_indices]
        high_conf_labels = pseudo_labels[high_conf_indices]

        # 5. 计算交叉熵损失，将伪标签视为ground truth
        # reduction='mean' 会对所有高置信度样本的损失求平均
        loss = F.cross_entropy(high_conf_logits, high_conf_labels)
        return loss
    else:
        # 如果没有高置信度样本，则返回0，不进行训练
        return torch.tensor(0.0, device=tgt_logits.device, requires_grad=True)


def advanced_self_supervised_loss(tgt_logits, conf_threshold=0.9, T=1.0):
    """
    计算一个高级的自监督损失，结合了软伪标签和样本加权。

    Args:
        tgt_logits (torch.Tensor): 目标域样本的原始模型输出 (logits)。形状: [B, C]
        conf_threshold (float): 用于筛选高置信度样本的阈值。
        T (float): 温度系数，用于软化概率分布。

    Returns:
        torch.Tensor: 优化后的损失。
    """
    # 1. 将 logits 转换为概率分布（可选温度系数）
    # 使用温度系数T来软化概率分布，防止one-hot化过于严重
    softmax_out = nn.Softmax(dim=1)(tgt_logits / T)

    # 2. 获取每个样本的最大概率（置信度）
    max_probs, pseudo_labels = torch.max(softmax_out, dim=1)

    # 3. 动态样本加权，而非硬性阈值
    # 这里我们使用置信度本身作为权重，而不是一个固定的0或1
    sample_weights = (max_probs > conf_threshold).float() * max_probs

    # 如果没有高置信度样本，返回0
    if sample_weights.sum() == 0:
        return torch.tensor(0.0, device=tgt_logits.device, requires_grad=True)

    # 4. 计算软伪标签（即，高置信度样本的原始概率分布）
    # 这一步是高级版本的核心，我们用软标签来计算损失
    soft_pseudo_labels = softmax_out.detach()  # detach 以确保梯度不回传到标签本身

    # 5. 计算带权重的软交叉熵损失
    # 我们使用 `F.kl_div` 来计算软标签的损失，它等价于软交叉熵
    loss = F.kl_div(F.log_softmax(tgt_logits, dim=1), soft_pseudo_labels, reduction='none')
    loss = torch.sum(loss, dim=1)  # 对每个样本的损失求和

    # 6. 应用样本权重，并对所有高置信度样本的损失求平均
    weighted_loss = loss * sample_weights
    final_loss = weighted_loss.sum() / sample_weights.sum()

    return final_loss

logger = None

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    # torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def im(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss  # Ent - gent, minimization problem
    im_loss = entropy_loss * 1.0
    return im_loss


def train(args, model,**kwargs):
    global logger
    logger = args.logger

    """ Train the model """
    os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size
    transform_train, transform_test = get_transform(args.img_size)

    train_data_list = open(args.train_list).readlines()
    test_data_list = open(args.test_list).readlines()

    train_loader = torch.utils.data.DataLoader(
        ImageList(train_data_list, args.dataset_path, transform=transform_train, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    target_loader = torch.utils.data.DataLoader(
        ImageList(test_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        ImageList(test_data_list, args.dataset_path, transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

    main_loader, secondary_loader, main_is_source = ((train_loader, target_loader, True)
                                                     if len(train_data_list) > len(test_data_list)
                                                     else (target_loader, train_loader, False))

    optimizer = torch.optim.SGD(params=[{'params': kwargs['mamba'].parameters()},
                                        {'params': kwargs['visual_projector'].parameters()},
                                        {'params': kwargs['classifier_head'].parameters()}],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        # scheduler = get_scheduler(optimizer, args)
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * 1)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    loss_mmd = MMD_loss()
    loss_lmmd = LMMDLoss(num_class=args.num_classes)
    loss_fct = torch.nn.CrossEntropyLoss()

    if args.mode == True:
        kwargs['mamba'].load_state_dict(torch.load('./SavedModel/office31/mamba_WtoD.pth'))
        kwargs['visual_projector'].load_state_dict(torch.load('./SavedModel/office31/Projector_WtoD.pth'))
        kwargs['classifier_head'].load_state_dict(torch.load('./SavedModel/office31/Head_WtoD.pth'))
        accuracy, cacc = valid(args, model, test_loader, global_step, mamba=kwargs['mamba'],
                               visual_projector=kwargs['visual_projector'], classifier_head=kwargs['classifier_head'])
        print(accuracy)
    else:
        while True:
            model.train()
            epoch_iterator = tqdm(main_loader,
                                  desc="Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)

            secondary_iterator = iter(secondary_loader)
            for step, batch in enumerate(epoch_iterator):
                try:
                    secondary_target = next(secondary_iterator)
                except StopIteration:
                    secondary_iterator = iter(secondary_loader)
                    secondary_target = next(secondary_iterator)

                if main_is_source:
                    batch_source = tuple(t.to(args.device) for t in batch)
                    x_source, y_source = batch_source

                    batch_target = tuple(t.to(args.device) for t in secondary_target)
                    x_target, y_target = batch_target
                else:
                    batch_source = tuple(t.to(args.device) for t in secondary_target)
                    x_source, y_source = batch_source

                    batch_target = tuple(t.to(args.device) for t in batch)
                    x_target, y_target = batch_target

                # source_features, logits = model(x_source)
                source_features = kwargs['mamba'](x_source)  # [B, 256]
                source_features = kwargs['visual_projector'](source_features)
                logits = kwargs['classifier_head'](source_features)
                # source_features = (source_features + src_f) / 2
                # logits = (logits + logits_mamba) / 2

                # CE Loss
                # loss = loss_fct(logits.view(-1, kwargs['classifier_head'].out_features), y_source.view(-1))
                # loss = loss_fct(logits.view(-1, args.num_classes), y_source.view(-1))
                loss = loss_fct(logits, y_source)

                if not args.source_only:
                    # target_features, tgt_logits = model(x_target)
                    target_features = kwargs['mamba'](x_target)  # [B, 256]
                    target_features = kwargs['visual_projector'](target_features)
                    # target_features = (target_features + tgt_f) / 2
                    tgt_logits = kwargs['classifier_head'](target_features)

                    # tgt_logits = (tgt_logits + tgt_logits_mamba) / 2
                    # loss_mmd_cal = loss_mmd(source_features, target_features) + BNM(target_features)
                    loss_mmd_cal = confidence_weighted_mmd_loss(source_features, target_features, tgt_logits)
                    # loss_mmd_cal = loss_lmmd(source_features, target_features, y_source, tgt_logits) # source, target, source_label, target_logits

                    loss_im = 0.7 * im(tgt_logits) + 0.3 * im(logits)
                    # loss_tta = (1. * (compute_LCS_loss_train(logits, y_source, kwargs['classifier_head'].weight,
                    #                                          C0=0.5) + compute_LCS_loss(tgt_logits, kwargs[
                    #     'classifier_head'].weight,C0=0.5))
                    #             )

                    loss_self_supervised = self_supervised_pseudo_label_loss(tgt_logits, conf_threshold=args.tau)

                    # loss = 0.7 * loss + (0.3 * loss_mmd_cal) + (0.01 * loss_im) + args.lambda_1 * loss_self_supervised + 0.1 * loss_tta
                    loss = 0.5 * loss + 0.5 * loss_mmd_cal + 0.05 * loss_im + 0.4 * loss_self_supervised
                    # loss = (0.7 * loss) + (0.3 * loss_mmd_cal) + 0.01 * loss_im + 0.4 *
                    # loss = (0.7 * loss) + (0.3 * loss_mmd_cal) + 0.1 * loss_self_supervised

                loss.backward()

                losses.update(loss.item())
                torch.nn.utils.clip_grad_norm_(kwargs['mamba'].head.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(kwargs['visual_projector'].parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(kwargs['classifier_head'].parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0:
                    accuracy, cacc = valid(args, model, test_loader, global_step, mamba=kwargs['mamba'],
                                           visual_projector=kwargs['visual_projector'],
                                           classifier_head=kwargs['classifier_head'])

                    logger.info(f"Accuracy: {accuracy}")

                    if best_acc < accuracy:
                        # save_model(args, model)
                        best_acc = accuracy
                        # torch.save(kwargs['mamba'].state_dict(), './SavedModel/OfficeHome/mamba_'+str(args.src)+'to'+str(args.tgt)+'.pth')
                        # torch.save(kwargs['visual_projector'].state_dict(), './SavedModel/OfficeHome/projector_'+str(args.src)+'to'+str(args.tgt)+'.pth')
                        # torch.save(kwargs['classifier_head'].state_dict(), './SavedModel/OfficeHome/head_'+str(args.src)+'to'+str(args.tgt)+'.pth')
                    model.train()

                if global_step % t_total == 0:
                    break

            losses.reset()
            if global_step % t_total == 0:
                break

        writer.close()
        logger.info("Best Accuracy: \t%f" % best_acc)
        logger.info("End Training!")