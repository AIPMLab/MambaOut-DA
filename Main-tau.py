import logging
import argparse
import random
import timm
import torch
import numpy as np

from Model import EUDA
from Process.Train import train
from Process.Test import test

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def setup(args):

    model = EUDA(args.backbone_type, args.backbone_size, args.bottleneck_size,args.num_classes)
    model.to(args.device)

    for name, param in model.named_parameters():
        if "norm" in name or "bottleneck" in name or "head" in name:
            continue

        param.requires_grad = False

    num_params = count_parameters(model)

    logger.info(f"Backbone Type: {args.backbone_type}, "
                f"Backbone Size: {args.backbone_size}, "
                f"Bottleneck Size: {args.bottleneck_size}")
    logger.info("Training parameters %s", num_params)

    return args, model


def main(dataset_path, train_list, test_list, lambda_1, **kwargs):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", help="Name of this run. Used for monitoring.",default='office')
    parser.add_argument("--dataset", help="Which downstream task.", default='office')
    parser.add_argument("--train_list", help="Path of the training data.", default=train_list)
    parser.add_argument("--test_list", help="Path of the test data.", default=test_list)
    parser.add_argument("--num_classes", default=65, type=int, help="Number of classes in the dataset.")
    parser.add_argument("--backbone_type", choices=["ViT-B_16", "DINOv2"],
                        default="DINOv2", help="Which variant to use.")
    parser.add_argument("--backbone_size", choices=["base", "large", "huge"],default='base')
    parser.add_argument("--bottleneck_size", choices=["small", "base", "large", "huge"],default='base')

    # TODO: Change pretrained path
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")
    parser.add_argument("--source_only", default=False, action="store_true",
                        help="Train without SDAL.")
    parser.add_argument("--dataset_path", type=str, help="Base path of the dataset.", default=dataset_path)

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=0.03, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=6000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="linear",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="ID of GPU")
    parser.add_argument("--acc", default=0, type=float,
                        help="test acc")
    parser.add_argument("--acc_visda", default=0, type=float,
                        help="test acc")
    parser.add_argument("--tau", default=0.95, type=float)
    parser.add_argument("--lambda_1", default=lambda_1, type=float)
    parser.add_argument("--mode", default=False)
    args = parser.parse_args()
    args.logger = logger

    logging.warning(f"Adapting {args.train_list} to {args.test_list}")

    # Setup CUDA, GPU & distributed training
    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process device: %s, gpu id: %s" %
                   (args.device, args.gpu_id))

    args, model = setup(args)

    # Set seed
    set_seed(args)

    args.src = kwargs['src']
    args.tgt = kwargs['tgt']

    if args.is_test:
        test(args, model)
    else:
         # mambaout_base_plus_rw.sw_e150_in12k_ft_in1k
        mamba = timm.create_model('mambaout_base_plus_rw.sw_e150_in12k_ft_in1k', pretrained=True,num_classes=0).to('cuda')
        all_params = list(mamba.parameters())
        # print(mamba)
        # for name, param in model.named_parameters():
        #     if 'head' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        for param in mamba.parameters():
            param.requires_grad = False
        # for param in mamba.head.parameters():
        #     param.requires_grad = True
        # print(mamba)
        visual_projector = torch.nn.Linear(3072, 256).to('cuda')
        # visual_projector = torch.nn.Sequential(torch.nn.Linear(3072, 256), # 3072
        #                                        torch.nn.LayerNorm(256),
        #                                        torch.nn.SiLU(),
        #                                        torch.nn.Linear(256,256),
        #                                        torch.nn.LayerNorm(256),
                                               # ).to('cuda')
        classifier_head = torch.nn.Linear(256, args.num_classes).to('cuda')
        # classifier_head = torch.nn.Sequential(torch.nn.Linear(256, 256),
        #                                       torch.nn.Linear(256, args.num_classes)).to('cuda')
        train(args, model,mamba=mamba,visual_projector=visual_projector,classifier_head=classifier_head)
        print(args.acc)
        print(args.acc_visda)
    return round(args.acc * 100., 2), args.acc_visda

if __name__ == "__main__":
    # print(timm.list_models())

    dataset_path = 'D:/WU/EUDA-main/'
    root = 'D:/WU/EUDA-main/Data/officehome/'
    # source = ['A.txt','C.txt','P.txt','R.txt'] #
    # target = ['A.txt','C.txt','P.txt','R.txt']
    source = ['amazon.txt', 'webcam.txt', 'dslr.txt']  #
    target = ['amazon.txt', 'webcam.txt', 'dslr.txt']
    # source = ['A.txt']
    # target = ['P.txt']
    train_batch = 32
    dataset = 'office'
    name = 'office'
    lambda_1 = [0.4]
    for lam in lambda_1:
        acc = []
        for src in source:
            for tgt in target:
                if src != tgt:
                    train_list = root + src
                    test_list = root + tgt
                    tmp, acc_visda = main(dataset_path, train_list, test_list, lambda_1=lam, src=src[0],tgt=tgt[0])
                    acc.append(tmp)

        acc = np.array(acc)
        # np.savetxt('./acc_all_prostate_C1toC4(lambda_3='+str(lam)+').csv', acc, fmt='%f', delimiter=',')
        # np.savetxt('./acc_all_visda17(lr=0.03).csv', acc, delimiter=',')
        print(acc, np.mean(acc))

