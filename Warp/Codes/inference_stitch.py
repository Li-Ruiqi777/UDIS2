"""
在UDIS-D数据集上进行测试并保存结果图片和mask
"""
import torch
from torch.utils.data import DataLoader
import argparse

from network import get_batch_outputs_for_stitch, UDIS2
from utils.ImageSaver import ImageSaver
from dataset import *
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

@torch.no_grad()
def test_stitch(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_dataset = TestDataset(data_path=args.test_dataset_path)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    model = UDIS2().to(device)
    model.eval()

    # 加载权重
    check_point = torch.load(args.ckpt_path)
    logger.info(f"load model from {args.ckpt_path}!")
    model.load_state_dict(check_point["model"])

    image_saver = ImageSaver(args.save_path)

    for idx, batch_value in enumerate(test_dataloader):

        inpu1_tesnor = batch_value[0].float().to(device)
        inpu2_tesnor = batch_value[1].float().to(device)

        batch_outputs = get_batch_outputs_for_stitch(model, inpu1_tesnor, inpu2_tesnor)
        save_stitch_result(batch_outputs, image_saver, idx)

        logger.info(f"save stitch result for idx = {idx + 1}")

def save_stitch_result(batch_outputs, image_saver, idx):
    translated_reference = batch_outputs['translated_reference'].cpu().numpy().transpose(0, 2, 3, 1)
    tps_warped_target = batch_outputs['tps_warped_target'].cpu().numpy().transpose(0, 2, 3, 1)
    translated_reference = de_normalize(translated_reference)
    tps_warped_target = de_normalize(tps_warped_target)

    # 融合结果
    weight_ref = translated_reference / (translated_reference + tps_warped_target + 1e-10)
    weight_warp = 1.0 - weight_ref
    np.clip(weight_ref, 0, 1, weight_ref)
    np.clip(weight_warp, 0, 1, weight_warp)
    ave_fusion = weight_ref * translated_reference + weight_warp * tps_warped_target
    ave_fusion = np.clip(ave_fusion, 0, 255).astype(np.uint8)
    
    # 结果放在一个文件夹
    image_saver.add_image(f'{str(idx + 1).zfill(6)}_warped', tps_warped_target[0])
    image_saver.add_image(f'{str(idx + 1).zfill(6)}_translated', translated_reference[0])
    image_saver.add_image(f'{str(idx + 1).zfill(6)}_fusion', ave_fusion[0])

    image_saver.flush()

def de_normalize(img):
    '''
    将图像数据从[-1, 1]归一化到[0, 255]
    还原归一化
    '''
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default='E:/DeepLearning/7_Stitch/UDIS2/Warp/model/epoch100_model.pth')
    parser.add_argument('--save_path', type=str, default='E:/DeepLearning/7_Stitch/UDIS2/Warp/results/stitch')
    parser.add_argument("--test_dataset_path",type=str, default="E:/DeepLearning/0_DataSets/007-UDIS-D-subset/test")
    args = parser.parse_args()

    test_stitch(args)
