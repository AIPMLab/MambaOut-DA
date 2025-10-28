import logging
import timm
from pyexpat import features
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm
import torch
import numpy as np
from Utils.Plots import plot_reliability_diagram
from .Metrics import AverageMeter, simple_accuracy
from Utils.utils import visda_acc
import torch.nn.functional as F
logger = None


def valid(args, model, test_loader, global_step, **kwargs):
    global logger
    logger = args.logger
    # mamba = timm.create_model('mambaout_base.in1k', pretrained=True, num_classes=65).to('cuda')
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_probs = []
    all_pre_label = []
    all_gt = []
    # all_preds_mamba = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            # logits = model(x, return_features_only=True)

            features = kwargs['mamba'](x)

            features = kwargs['visual_projector'](features)

            logits = kwargs['classifier_head'](features)

            # logits_mamba = mamba(x)

            # logits = (logits + logits_mamba) / 2

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_probs.append(F.softmax(logits).detach().cpu().numpy())
            all_pre_label.append(preds.detach().cpu().numpy())
            all_gt.append(y.detach().cpu().numpy())
            # preds_mamba = torch.argmax(logits_mamba, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            # all_preds_mamba.append(preds_mamba.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            # all_preds_mamba[0] = np.append(all_preds_mamba[0], preds_mamba.detach().cpu().numpy(), axis=0)

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    all_probs = np.concatenate(all_probs)
    all_pre_label = np.concatenate(all_pre_label)
    all_gt = np.concatenate(all_gt)
    confs = all_probs[range(all_probs.shape[0]), all_pre_label]
    all_preds, all_label = all_preds[0], all_label[0]
    # plot_reliability_diagram(all_pre_label, confs, all_gt, 10, None, './Calibration/Office31_WtoD.pdf')
    # all_preds_mamba = all_preds_mamba[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)
        classWise_acc = []
        # accuracy_mamba = simple_accuracy(all_preds_mamba, all_label)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    print("Valid Accuracy: %2.5f" % accuracy)
    print('Valid each class accuracy:', classWise_acc)
    args.acc_visda = classWise_acc
    # print("Valid Accuracy with Mamba: %2.5f" % accuracy_mamba)
    bacc = balanced_accuracy_score(all_label, all_preds)
    f1 = f1_score(all_label, all_preds, average='macro')
    # print('Vaid BACC: %2.5f' % (bacc))
    # print('Vaid F1: %2.5f' % (f1))

    if accuracy > args.acc:
        args.acc = accuracy
        if args.dataset == 'visda17':
            args.acc_visda = classWise_acc

    if args.dataset == 'visda17':
        logger.info(classWise_acc)

    # writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)

    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None
