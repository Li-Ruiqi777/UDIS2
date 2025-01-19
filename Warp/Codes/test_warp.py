"""
在UDIS-D数据集上进行测试并保存结果图片和mask
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, Network
from dataset import *
import os
import cv2

import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

# last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = "E:/DeepLearning/7_Stitch/UDIS2/Warp/model"

def test(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    # nl: set num_workers = the number of cpus
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        drop_last=False,
    )

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    model_path = "E:/DeepLearning/7_Stitch/UDIS2/Warp/model/epoch100_model.pth"
    checkpoint = torch.load(model_path)

    net.load_state_dict(checkpoint["model"])
    print("load model from {}!".format(model_path))

    path_ave_fusion = os.path.join(args.test_path , "ave_fusion/")
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)

    path_warp1 = os.path.join(args.test_path , "warp1/")
    if not os.path.exists(path_warp1):
        os.makedirs(path_warp1)

    path_warp2 = os.path.join(args.test_path , "warp2/")
    if not os.path.exists(path_warp2):
        os.makedirs(path_warp2)

    path_mask1 = os.path.join(args.test_path , "mask1/")
    if not os.path.exists(path_mask1):
        os.makedirs(path_mask1)

    path_mask2 = os.path.join(args.test_path , "mask2/")
    if not os.path.exists(path_mask2):
        os.makedirs(path_mask2)

    net.eval()
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)

            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']

            final_warp1 = (
                ((final_warp1[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
            )
            final_warp2 = (
                ((final_warp2[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
            )
            final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
            final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
            final_mesh1 = final_mesh1[0].cpu().detach().numpy()
            final_mesh2 = final_mesh2[0].cpu().detach().numpy()

            # 保存warp的结果和mask
            path = path_warp1 + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, final_warp1)

            path = path_warp2 + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, final_warp2)

            path = path_mask1 + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, final_warp1_mask * 255)

            path = path_mask2 + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, final_warp2_mask * 255)

            # 加权融合ref和tar的warp结果
            ave_fusion = (final_warp1 * (final_warp1 / (final_warp1 + final_warp2 + 1e-6)))+(final_warp2 * (final_warp2 / (final_warp1 + final_warp2 + 1e-6)))

            path = path_ave_fusion + str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(path, ave_fusion)

            print("i = {}".format(i + 1))
            torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--test_path",
        type=str,
        default="E:/DeepLearning/0_DataSets/007-UDIS-D/testing/testing",
    )

    print("<==================== Loading data ===================>\n")

    args = parser.parse_args()
    print(args)
    test(args)
