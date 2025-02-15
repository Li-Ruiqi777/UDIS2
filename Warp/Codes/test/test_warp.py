import cv2
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network import *
from utils.ImageSaver import ImageSaver

ckpt_path = "E:/DeepLearning/7_Stitch/UDIS2/Warp/model/epoch100_model.pth"
save_path = 'E:/DeepLearning/7_Stitch/UDIS2/Warp/results/test_warp'

@torch.no_grad()
def test_get_batch_outputs_for_train():
    model = UDIS2()
    model = model.cuda()

    # 加载权重
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 加载输入
    ref_img = cv2.imread("E:/DeepLearning/0_DataSets/007-UDIS-D/testing/testing/input1/000001.jpg")
    target_img = cv2.imread("E:/DeepLearning/0_DataSets/007-UDIS-D/testing/testing/input2/000001.jpg")
    # 归一化
    ref_img = ref_img.astype(np.float32) / 127.5 - 1.0
    target_img = target_img.astype(np.float32) / 127.5 - 1.0
    
    ref_img = np.transpose(ref_img, (2, 0, 1))
    target_img = np.transpose(target_img, (2, 0, 1))
    
    target_tensor = torch.from_numpy(target_img).unsqueeze(0).cuda()
    ref_tensor = torch.from_numpy(ref_img).unsqueeze(0).cuda()

    # 推理
    batch_output = get_batch_outputs_for_train(model, ref_tensor, target_tensor, False)
    
    # output_H = batch_output['output_H']
    # output_H_inv = batch_output['output_H_inv']
    # warp_mesh = batch_output['warp_mesh']
    # warp_mesh_mask = batch_output['warp_mesh_mask']
    # mesh1 = batch_output['mesh1']
    # mesh2 = batch_output['mesh2']
    # overlap = batch_output['overlap']
    
    # 保存结果
    image_saver = ImageSaver(save_path)

    warped_target = batch_output['output_H'][:, 0:3, :, :].cpu().numpy().transpose(0, 2, 3, 1)
    H_warped_mask = batch_output['output_H'][:, 3:6, :, :].cpu().numpy().transpose(0, 2, 3, 1)

    warped_reference = batch_output['output_H_inv'][:, 0:3, :, :].cpu().numpy().transpose(0, 2, 3, 1)
    H_inv_warped_mask = batch_output['output_H_inv'][:, 3:6, :, :].cpu().numpy().transpose(0, 2, 3, 1)

    tps_warped_target = batch_output['warp_mesh'].cpu().numpy().transpose(0, 2, 3, 1)
    tps_warped_mask = batch_output['warp_mesh_mask'].cpu().numpy().transpose(0, 2, 3, 1)

    image_saver.add_image('warped_target', de_normalize(warped_target[0]))
    image_saver.add_image('H_warped_mask', de_normalize(H_warped_mask[0]))
    image_saver.add_image('warped_reference', de_normalize(warped_reference[0]))
    image_saver.add_image('H_inv_warped_mask', de_normalize(H_inv_warped_mask[0]))
    image_saver.add_image('tps_warped_target', de_normalize(tps_warped_target[0]))
    image_saver.add_image('tps_warped_mask', de_normalize(tps_warped_mask[0]))

    image_saver.flush()

def de_normalize(img):
    '''
    将图像数据从[-1, 1]归一化到[0, 255]
    还原归一化
    '''
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    return img

    
if __name__ == '__main__':
    test_get_batch_outputs_for_train()
    