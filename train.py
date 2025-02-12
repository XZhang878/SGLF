import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
from torch import Tensor
import torch.nn.functional as F
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList
import datetime
from Beta_Mixture_mode import BetaMixture1D
from affinity_layer import Affinity, sinkhorn_rpm, one_hot, BCEFocalLoss
from multiheadattentation import MultiHeadAttention
from k_predict_net import Encoder
from sinkhorn_topk import soft_topk
import numpy as np
from os.path import join
import json
from basenet import *
from utils import *
from GCN import GCN
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy that's robust to label noise"""
    def __init__(self, *args, offset=0.1, **kwargs):
        self.offset = offset
        super(RobustCrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(torch.clamp(input + self.offset, max=1.), target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='sum') / input.shape[0]

def image_classification_test(loader, G, F, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    feature_out = G(inputs[j])
                    predict_out = F(feature_out)

                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False

                else:

                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                features = G(inputs)
                outputs = F(features)

                # _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False

                else:

                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    with open(join('/data2/yaoxiwen4/home/disk1/code_1/SGLF/data/MLRSNet2WHU/info.json'), 'r') as fp:
        info = json.load(fp)
    name_classes = np.array(info['label'], dtype=np.str)
    num_classes = np.int(info['classes'])
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    c = (predict == all_label).squeeze()
    for i in range(len(all_label)):
        _label = all_label[i]
        class_correct[int(_label)] += c[i].item()
        class_total[int(_label)] += 1
    for i in range(num_classes):
        # print('Accuracy of %5s : %2d %%' % (
        #     name_classes[i], 100 * class_correct[i] / class_total[i]))
        print('Accuracy of %5s : %5f %%' % (
            name_classes[i], 100 * class_correct[i] / class_total[i]))
    return accuracy


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=0, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=0) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=0)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    # net_config = config["network"]
    # base_network = net_config["name"](**net_config["params"])
    # base_network = base_network.cuda()
    option = 'resnet' + args.resnet
    num_layer = args.num_layer
    G = ResBottle(option)
    F = ResClassifier(num_classes=6, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
    F.apply(weights_init)
    G = G.cuda()
    F = F.cuda()
    GraphCN = GCN(nfeat=256, nhid=256)
    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([G.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(G.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    # parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    # optimizer_config = config["optimizer"]
    # optimizer = optimizer_config["type"](parameter_list, \
    #                                      **(optimizer_config["optim_params"]))
    optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F.parameters()), momentum=0.9, lr=args.lr,
                            weight_decay=0.0005)

    # param_lr = []
    # for param_group in optimizer.param_groups:
    #     param_lr.append(param_group["lr"])
    # schedule_param = optimizer_config["lr_param"]
    # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    #
    # gpus = config['gpu'].split(',')
    # if len(gpus) > 1:
    #     ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
    #     base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    # best_model = nn.Sequential(base_network)
    each_log = ""
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:

            G.train(False)
            F.train(False)

            temp_acc = image_classification_test(dset_loaders, G, F, test_10crop=prep_config["test_10crop"])
            if temp_acc > best_acc:
                best_acc = temp_acc
            log_str = "iter: {:05d}, precision: {:.5f}, transfer_loss:{:.4f}, classifier_loss:{:.4f}, total_loss:{:.4f}" \
                .format(i, temp_acc, transfer_loss.item(), classifier_loss.item(), total_loss.item())
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)

            config["out_file"].write(each_log)
            config["out_file"].flush()
            each_log = ""


        loss_params = config["loss"]
        ## train one iter
        G.train(True)
        F.train(True)
        ad_net.train(True)
        # optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source = G(inputs_source)
        outputs_source = F(features_source)
        features_target = G(inputs_target)
        outputs_target = F(features_target)

        features_source = GraphCN(features_source.cpu())
        features_target = GraphCN(features_target.cpu())
        features_source = features_source.cuda()
        features_target = features_target.cuda()
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        # labels = torch.cat((labels_source, labels_target_fake))
        entropy = loss.Entropy(softmax_out)
        transfer_loss = loss_params["trade_off"] * loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)

        classifier_loss = args.cls_weight * nn.CrossEntropyLoss()(outputs_source, labels_source)

        confidence, labels_target_fake = torch.max(nn.Softmax(dim=1)(outputs_target), 1)
        bmm_model = BetaMixture1D(max_iters=10)
        confidence = confidence.cpu()
        c = np.asarray(confidence.detach().numpy())
        c, c_min, c_max = bmm_model.outlier_remove(c)
        c = bmm_model.normalize(c, c_min, c_max)
        bmm_model.fit(c)
        bmm_model.create_lookup(1)  # 0: noisy, 1: clean
        # get posterior
        c = np.asarray(confidence.detach().numpy())
        c = bmm_model.normalize(c, c_min, c_max)
        p = bmm_model.look_lookup(c)
        p = torch.from_numpy(p)

        weight = torch.tensor(p, dtype=torch.float32).cuda()

        mask = weight.ge(0.5).float()


        cls_loss_t = RobustCrossEntropyLoss(ignore_index=-1, offset=args.epsilon)(outputs_target, labels_target_fake)

        loss_t_weight = args.self_train * ((cls_loss_t * mask).sum() / (mask.sum() + 1e-6))

        nodes_fs = features_source
        nodes_ft = features_target
        #

        affinity = Affinity(512)
        cross_domain_feature = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Cross Graph Interaction
        intra_domain_feature = MultiHeadAttention(256, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation
        intra_f_s = intra_domain_feature.forward(nodes_fs.cpu(), nodes_fs.cpu(), nodes_fs.cpu())[0]
        intra_f_t = intra_domain_feature.forward(nodes_ft.cpu(), nodes_ft.cpu(), nodes_ft.cpu())[0]

        cross_f_s = cross_domain_feature.forward(intra_f_t, intra_f_t, intra_f_s)[0]
        cross_f_t = cross_domain_feature.forward(intra_f_s, intra_f_s, intra_f_t)[0]

        prob = affinity(cross_f_s, cross_f_t)
        InstNorm_layer = nn.InstanceNorm2d(1)
        M = InstNorm_layer(prob[None, None, :, :])
        M = sinkhorn_rpm(M[:, 0, :, :], n_iters=20).squeeze().exp()

        encoder = Encoder()

        maxpool = nn.MaxPool1d(kernel_size=19)


        final_row = nn.Sequential(
            nn.Linear(256, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        final_col = nn.Sequential(
            nn.Linear(256, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        out_emb_s, out_emb_t = encoder(cross_f_s, cross_f_t, M)

        out_emb_s = out_emb_s.permute(0, 2, 1)
        out_emb_t = out_emb_t.permute(0, 2, 1)
        maxpool_out_1 = maxpool(out_emb_s).squeeze(-1)
        maxpool_out_2 = maxpool(out_emb_t).squeeze(-1)
        k_row = final_row(maxpool_out_1).squeeze(-1)
        k_col = final_col(maxpool_out_2).squeeze(-1)
        ks = (k_row + k_col) / 2

        M_k = M.reshape(1, M.size(0), M.size(1))
        _, ss = soft_topk(M_k, ks.view(-1) * 36)
        ss = ss.squeeze()


        one_hot_s = one_hot(labels_source.cpu(), num_class=6)

        one_hot_t = one_hot(labels_target_fake.cpu(), num_class=6)
        one_hot_s = torch.as_tensor(one_hot_s)
        one_hot_t = torch.as_tensor(one_hot_t)
        matching_target = torch.mm(one_hot_s, one_hot_t.t())
        TP_mask = (matching_target == 1).float()
        indx = (ss * TP_mask).max(-1)[1]
        TP_samples = ss[range(ss.size(0)), indx].view(-1, 1)
        TP_target = torch.full(TP_samples.shape, 1, dtype=torch.float, device=TP_samples.device).float()

        FP_samples = M[matching_target == 0].view(-1, 1)
        FP_target = torch.full(FP_samples.shape, 0, dtype=torch.float, device=FP_samples.device).float()
        matching_loss = BCEFocalLoss()
        TP_loss = matching_loss(TP_samples, TP_target.float()) / len(TP_samples)
        FP_loss = matching_loss(FP_samples, FP_target.float()) / torch.sum(FP_samples).detach()
        # print('FP: ', FP_loss, 'TP: ', TP_loss)
        ss = ss.cuda()
        R = torch.mm(ss, nodes_fs) - torch.mm(ss, nodes_ft)
        quadratic_loss = torch.nn.L1Loss(reduction='mean')
        loss_quadratic = quadratic_loss(R, R.new_zeros(R.size()))
        TP_loss = TP_loss.cuda()
        FP_loss = FP_loss.cuda()
        matching_loss = args.matching_weight * (TP_loss + FP_loss + loss_quadratic)
        matching_loss = matching_loss.cuda()

        # mdd_loss = loss.mdd_loss(
        #     features=features, labels=labels, left_weight=args.left_weight, right_weight=args.right_weight)

        # max_entropy_loss = args.entropic_weight * loss.EntropicConfusion(features)
        softmax_tgt = nn.Softmax(dim=1)(outputs_target)
        _, s_tgt, _ = torch.svd(softmax_tgt)
        if config["method"] == "BNM":
            method_loss = -torch.mean(s_tgt)
        elif config["method"] == "BFM":
            method_loss = -torch.sqrt(torch.sum(s_tgt * s_tgt) / s_tgt.shape[0])
        elif config["method"] == "ENT":
            method_loss = -torch.mean(torch.sum(softmax_tgt * torch.log(softmax_tgt + 1e-8), dim=1)) / torch.log(
                softmax_tgt.shape[1])
        elif config["method"] == "NO":
            method_loss = 0

        total_loss = transfer_loss + classifier_loss + matching_loss + loss_t_weight + args.lambda_method * method_loss
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        # log_str = "iter: {:05d},transfer_loss:{:.4f}, classifier_loss:{:.4f}, matching_loss:{:4f}," \
        #           "max_entropy_loss:{:.4f},total_loss:{:.4f}" \
        #     .format(i, transfer_loss.item(), classifier_loss.item(), matching_loss.item(),
        #             max_entropy_loss.item(), total_loss.item())
        # each_log += log_str + "\n"
        if i % config['print_num'] == 0:
            log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, pseudo: {:.5f}, matching: {:.5f}, method: {:.5f}".format(i,
              classifier_loss, transfer_loss, args.self_train * loss_t_weight, args.matching_weight * matching_loss, args.lambda_method * method_loss)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            if config['show']:
                print(log_str)

            if i % args.print_freq == 0:

                print("iter: {:05d},transfer_loss:{:.6f}, classifier_loss:{:.6f}, loss_t_weight:{:6f}, matching_loss:{:6f}, method_loss:{:.6f},total_loss:{:.6f}" \
                    .format(i, loss_params["trade_off"] * transfer_loss.item(),
                            args.cls_weight * classifier_loss.item(),
                            args.self_train * loss_t_weight, args.matching_weight * matching_loss,
                            args.lambda_method * method_loss, total_loss))

        if i % config["snapshot_interval"] == 0:
               torch.save(G, osp.join(config["output_path"], "iter_{:05d}_G_model.pth.tar".format(i)))
               torch.save(F, osp.join(config["output_path"], "iter_{:05d}_F_model.pth.tar".format(i)))

    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_num', type=int, default=100, help="print num ")
    parser.add_argument('--method', type=str, default='BNM', choices=['BNM', 'BFM', 'ENT', 'NO'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='4', help="device id to run")
    parser.add_argument('--resnet', type=str, default='50', metavar='B', help='which resnet 18,50,101,152,200')
    parser.add_argument('--num_layer', type=int, default=2, metavar='K', help='how many layers for classifier')
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='/data2/yaoxiwen4/home/disk1/code_1/SGLF/data/MLRSNet2WHU/source_MLRSNet.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='/data2/yaoxiwen4/home/disk1/code_1/SGLF/data_rs/MLRSNet2WHU/target_WHU-SAR.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=500, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='/data2/yaoxiwen4/home/disk1/code_1/SGLF/MLRSNet2WHU',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument("--matching_weight", type=float, default=1.0)
    parser.add_argument("--log_name", type=str, default="MLRSNet2WHU")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--use_seed", type=bool, default=False)
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--torch_cuda_seed", type=int, default=1)
    parser.add_argument("--left_weight", type=float, default=1)
    parser.add_argument("--right_weight", type=float, default=1)
    parser.add_argument("--cls_weight", type=float, default=1)
    parser.add_argument("--epoch", type=int, default=40000)
    parser.add_argument('--lambda_method', type=float, default=0.1, help="parameter for method")
    parser.add_argument('--self_train', type=float, default=0.01)
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--epsilon', default=0.01, type=float,
                        help='epsilon hyper-parameter in Robust Cross Entropy')
    parser.add_argument('--show', type=bool, default=False, help="whether show the loss functions")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if (args.use_seed):
        torch.manual_seed(args.torch_seed)
        torch.cuda.manual_seed(args.torch_cuda_seed)
        torch.cuda.manual_seed_all(args.torch_cuda_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = {}
    config["left_weight"] = args.left_weight
    config["right_weight"] = args.right_weight
    config['torch_seed'] = torch.initial_seed()
    config['torch_cuda_seed'] = torch.cuda.initial_seed()
    config["mdd_weight"] = args.matching_weight
    config["output_path"] = args.output_dir
    config['method'] = args.method
    config["print_num"] = args.print_num
    config["show"] = args.show

    # config["entropic_weight"] = args.entropic_weight
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.epoch
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["log_output_path"] = args.output_dir + "/log/"
    config["model_output_path"] = args.output_dir + "/model/"
    config['log_name'] = args.log_name
    if not osp.exists(config["log_output_path"]):
        os.system('mkdir -p ' + config["log_output_path"])
    config["out_file"] = open(
        osp.join(config["log_output_path"], args.log_name + "_{}.txt".format(str(datetime.datetime.utcnow()))), "w")
    if not osp.exists(config["log_output_path"]):
        os.mkdir(config["log_output_path"])
    if not osp.exists(config["model_output_path"]):
        os.mkdir(config["model_output_path"])

    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 36}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 4}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters 0.001 default
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters 0.0003 default
        config["network"]["params"]["class_num"] = 6
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 6
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 6
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 6
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')


    config["out_file"].write(str(config) + "\n")
    config["out_file"].flush()
    train(config)
