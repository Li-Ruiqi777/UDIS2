import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import ssl

import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
from utils.misc import *
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

class UDIS2(nn.Module):
    def __init__(self):
        super(UDIS2, self).__init__()

        # H的回归网络
        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )

        # TPS的回归网络
        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ssl._create_default_https_context = ssl._create_unverified_context
        resnet50_model = models.resnet.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.feature_extractor_stage1, self.feature_extractor_stage2 = self.get_res50_FeatureMap(resnet50_model)

    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)

        feature_extractor_stage1 = nn.Sequential(*layers_list)

        feature_extractor_stage2 = nn.Sequential(resnet50_model.layer3)

        # layers_list.append(resnet50_model.layer3)

        return feature_extractor_stage1, feature_extractor_stage2

    # 返回H中4点的偏移以及TPS中各控制点的偏移
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)  # refrence的1/8
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)  # refrence的1/16

        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)  # target的1/8
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)  # target的1/16

        ######### stage 1
        correlation_32 = self.CCL(feature_1_32, feature_2_32)
        temp_1 = self.regressNet1_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)  # 计算4-pt的偏移量

        src_p = torch.tensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1  # 将偏移施加到点上
        H = torch_DLT.tensor_DLT(src_p / 8, dst_p / 8)  # 由DLT计算H
        
        # 这里之所以/8是因为输出feature map的尺寸是原图的1/8
        M_tensor = torch.tensor(
            [
                [img_w / 8 / 2.0, 0.0, img_w / 8 / 2.0],
                [0.0, img_h / 8 / 2.0, img_h / 8 / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        # 将全局的单应性变换H施加到target上
        warp_feature_2_64 = torch_homo_transform.transformer(
            feature_2_64, H_mat, (int(img_h / 8), int(img_w / 8))
        )

        ######### stage 2
        correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)  # 计算TPS中各控制点的偏移

        return offset_1, offset_2

    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches

    # Context Correlation Layer
    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        # print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters = patches.reshape(
            (
                patches.size()[0],
                -1,
                patches.size()[3],
                patches.size()[4],
                patches.size()[5],
            )
        )

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(
                norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1
            )
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        # print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol * softmax_scale, 1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h - 1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w - 1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel - 1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol * (c_one // w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol * (c_one % w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        # print(flow.size())

        return feature_flow

def get_batch_outputs_for_train(net, ref_tensor, target_tensor, is_training=True):
    """
    用于训练和指标计算, 计算一个batch的输出, 输出图像的大小和输入图像一样

    1.用网络预测H的4pt和TPS的控制点的偏移
    2.计算mesh(的控制点)在网络预测的H和TPS motion后的坐标
    3.将target及其mask通过单应性矩阵变到reference坐标系下(同名像素点的在2图中一样)
    4.将reference及其mask通过单应性矩阵变到target坐标系下
    5.将target及其mask通过TPS变到reference坐标系下
    6.使用计算重叠区域,并用一个二值图表示重叠区和非重叠区----?有吗
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(ref_tensor, target_tensor)
        H_motion, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion, mesh_motion = net(ref_tensor, target_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)

    src_p = torch.tensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion

    H_mat = torch_DLT.tensor_DLT(src_p, dst_p)
    H_inv_mat = torch.inverse(H_mat)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H_mat, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    normalize_mat = torch.tensor(
        [
            [2.0/img_w, 0.0, -1.0],
            [0.0, 2.0/img_h, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    denormalize_mat = torch.tensor(
        [
            [img_w/2.0, 0.0, img_w/2.0],
            [0.0, img_h/2.0, img_h/2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    mask = torch.ones_like(target_tensor)

    if torch.cuda.is_available():
        normalize_mat = normalize_mat.cuda()
        denormalize_mat = denormalize_mat.cuda()
        mask = mask.cuda()

    denormalize_mat = denormalize_mat.unsqueeze(0).expand(batch_size, -1, -1)
    normalize_mat = normalize_mat.unsqueeze(0).expand(batch_size, -1, -1)

    H_mat = torch.matmul(torch.matmul(normalize_mat, H_mat), denormalize_mat)
    H_inv_mat = torch.matmul(torch.matmul(normalize_mat, H_inv_mat), denormalize_mat)   

    temp = torch.cat((target_tensor, mask), 1)
    warped_target_and_mask = torch_homo_transform.transformer(temp, H_mat, (img_h, img_w))

    temp = torch.cat((ref_tensor, mask), 1)
    warped_reference_and_mask = torch_homo_transform.transformer(temp, H_inv_mat, (img_h, img_w))

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    temp = torch.cat((target_tensor, mask), 1)
    tps_warped_target_and_mask = torch_tps_transform.transformer(temp, norm_mesh, norm_rigid_mesh, (img_h, img_w))
    tps_warped_target = tps_warped_target_and_mask[:, 0:3, ...]
    tps_warped_mask = tps_warped_target_and_mask[:, 3:6, ...]

    # 计算重叠区域以施加shape-preserving约束
    overlap = torch_tps_transform.transformer(tps_warped_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = (
        overlap.permute(0, 2, 3, 1)
        .unfold(1, int(img_h / grid_h), int(img_h / grid_h))
        .unfold(2, int(img_w / grid_w), int(img_w / grid_w))
    )
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap < 0.9, overlap_one, overlap_zero)

    batch_outputs = {
        #[N, 6, 512, 512]
        'warped_target_and_mask':warped_target_and_mask,
        #[N, 6, 512, 512]
        'warped_reference_and_mask':warped_reference_and_mask,
        #[N, 3, 512, 512]
        'tps_warped_target':tps_warped_target,
        #[N, 3, 512, 512]
        'tps_warped_mask':tps_warped_mask,
        #[N, 13, 13, 2]
        'rigid_mesh':rigid_mesh,
        #[N, 13, 13, 2]
        'mesh':mesh,
        #[N, 12, 12]
        'overlap':overlap, # 在target系下的重叠区,0:重叠,1:非重叠
    }

    return batch_outputs

def get_batch_outputs_for_stitch(net, ref_tensor, target_tensor):
    """
    用于计算拼接时一个batch的输出, 输出图像的大小和最终拼接结果图像的大小一致
    
    1.用网络预测H的4pt和TPS的控制点的偏移
    2.根据变换后控制点的坐标最值,确定stitched img的h,w大小
    3.对reference进行平移变换,将其变到stitched img的坐标系下
    4.对target进行TPS变换,将其变到stitched img的坐标系下
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    resized_ref_tensor = resize_512(ref_tensor)
    resized_target_tensor = resize_512(target_tensor)
    H_motion, mesh_motion = net(resized_ref_tensor, resized_target_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    # 由于H是在(512,512)分辨率下训练的,这里先除以512再乘以当前的h,w是为了适应不同的分辨率
    H_motion = torch.stack(
        [H_motion[..., 0] * img_w / 512, H_motion[..., 1] * img_h / 512], 2
    )
    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)
    mesh_motion = torch.stack(
        [mesh_motion[..., 0] * img_w / 512, mesh_motion[..., 1] * img_h / 512], 3
    )

    src_p = torch.tensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion

    H_mat = torch_DLT.tensor_DLT(src_p, dst_p)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H_mat, rigid_mesh)  # 使用H对网格上各控制点施加一个初始的偏移
    mesh = ini_mesh + mesh_motion

    # 根据控制点的坐标最值,确定stitched img的h,w大小
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    # stitched img的h,w大小
    out_width = width_max - width_min
    out_height = height_max - height_min

    # print(f"x_max: {width_max}, x_min: {width_min}, y_max: {height_max}, y_min: {height_min}")
    # print(f"output_width: {out_width}, output_height: {out_height}")

    normalize_mat = torch.tensor(
        [
            [2.0/img_w, 0.0, -1.0],
            [0.0, 2.0/img_h, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    denormalize_mat = torch.tensor(
        [
            [out_width / 2.0, 0.0, out_width / 2.0],
            [0.0, out_height / 2.0, out_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    reference_translation_mat = torch.tensor(
        [
            [1.0, 0.0, width_min],
            [0.0, 1.0, height_min],
            [0.0, 0.0, 1.0]
         ]
    )

    mask = torch.ones_like(target_tensor)

    if torch.cuda.is_available():
        denormalize_mat = denormalize_mat.cuda()
        normalize_mat = normalize_mat.cuda()
        reference_translation_mat = reference_translation_mat.cuda()
        mask = mask.cuda()        

    reference_translation_mat = torch.matmul(torch.matmul(normalize_mat, reference_translation_mat), denormalize_mat).unsqueeze(0)
    
    translated_reference_and_mask = torch_homo_transform.transformer(
        torch.cat((ref_tensor + 1, mask), 1),
        reference_translation_mat,
        (out_height.int(), out_width.int()),
    )

    torch.cuda.empty_cache()

    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    
    tps_warped_target_and_mask = torch_tps_transform.transformer(
        torch.cat([target_tensor + 1, mask], 1),
        norm_mesh,
        norm_rigid_mesh,
        (out_height.int(), out_width.int()),
    )

    out_dict = {}
    out_dict.update(
        #[N, 3, output_height, output_width]
        final_warp1=translated_reference_and_mask[:, 0:3, ...] - 1, #将reference平移变换到stitched image的坐标系下
        #[N, 3, output_height, output_width]
        final_warp1_mask=translated_reference_and_mask[:, 3:6, ...],#将reference平移变换到stitched image的坐标系下的mask
        #[N, 3, output_height, output_width]
        final_warp2=tps_warped_target_and_mask[:, 0:3, ...] - 1,  #将target进行TPS变换到stitched image的坐标系下
        #[N, 3, output_height, output_width]
        final_warp2_mask=tps_warped_target_and_mask[:, 3:6, ...], #将target进行TPS变换到stitched image的坐标系下的mask
        mesh1=rigid_mesh,
        mesh2=mesh_trans,
    )

    batch_outputs = {
        #[N, 3, output_height, output_width]
        'translated_reference': translated_reference_and_mask[:, 0:3, ...] - 1,
        #[N, 3, output_height, output_width]
        'translated_mask': translated_reference_and_mask[:, 3:6, ...],
        #[N, 3, output_height, output_width]
        'tps_warped_target': tps_warped_target_and_mask[:, 0:3, ...] - 1,
        #[N, 3, output_height, output_width]
        'tps_warped_mask': tps_warped_target_and_mask[:, 3:6, ...],
        'rigid_mesh': rigid_mesh,
        'mesh_trans': mesh_trans,
    }

    return batch_outputs

def get_batch_outputs_for_ft(net, input1_tensor, input2_tensor):
    """
    计算在线Fine tuning中的初步配准结果
    与`get_batch_outputs_for_stitch`的区别:此函数没有改变将warp后的图片的(h,w).导致warp后一部分图像的缺失

    1.用网络预测H的4pt和TPS的控制点的偏移
    2.计算mesh(的控制点)在网络预测的H和TPS motion后的坐标
    3.带入(2)的数据对target进行TPS变换
    """
    batch_size, _, img_h, img_w = input1_tensor.size()

    H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    # H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)
    # mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    # 下面这一堆mesh的计算是用于TPS的
    # 首先生成原始均匀分布的控制点(rigid_mesh),然后根据H变换后得到的mesh_motion进行偏移得到mesh
    # 然后对rigid_mesh和mesh进行归一化,并进行TPS进行变换
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps = torch_tps_transform.transformer(
        torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w)
    )
    warp_mesh = output_tps[:, 0:3, ...]
    warp_mesh_mask = output_tps[:, 3:6, ...]

    out_dict = {}
    out_dict.update(
        warp_mesh=warp_mesh,           #对target图像进行TPS变换后的结果
        warp_mesh_mask=warp_mesh_mask, #对mask进行TPS变换后的结果
        rigid_mesh=rigid_mesh,
        mesh=mesh,
    )

    return out_dict

# 用于在线迭代算法中的后续Fine tuning
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    """
    1.根据变换后控制点的坐标最值,确定stitched img的h,w大小
    2.对reference进行单应性变换(平移),将其变到stitched img的坐标系下
    3.对target进行TPS变换,将其变到stitched img的坐标系下
    4.对2张配准后的图片进行average fusion
    """
    batch_size, _, img_h, img_w = input1_tensor.size()

    # 根据实际的img size通过缩放调整控制点的坐标
    rigid_mesh = torch.stack(
        [rigid_mesh[..., 0] * img_w / 512, rigid_mesh[..., 1] * img_h / 512], 3
    )
    mesh = torch.stack([mesh[..., 0] * img_w / 512, mesh[..., 1] * img_h / 512], 3)

    # 根据控制点的坐标最值,确定stitched img的h,w大小
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)

    # 将reference变到stitched image的坐标系下(可以看成平移操作)
    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[
        :,
        :,
        int(torch.abs(height_min)) : int(torch.abs(height_min)) + img_h, #warp1的在新坐标系下的范围[abs(height_min), abs(height_min) + img_h]
        int(torch.abs(width_min)) : int(torch.abs(width_min)) + img_w,
    ] = (input1_tensor + 1) * 127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[
        :,
        :,
        int(torch.abs(height_min)) : int(torch.abs(height_min)) + img_h,
        int(torch.abs(width_min)) : int(torch.abs(width_min)) + img_w,
    ] = 255

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    # 用TPS变换将target变到stitched image的坐标系下
    stitch_tps_out = torch_tps_transform.transformer(
        torch.cat([input2_tensor + 1, mask], 1),
        norm_mesh,
        norm_rigid_mesh,
        (out_height.int(), out_width.int()),
    )
    warp2 = stitch_tps_out[:, 0:3, :, :] * 127.5
    mask2 = stitch_tps_out[:, 3:6, :, :] * 255

    # average fustion
    stitched = warp1 * (warp1 / (warp1 + warp2 + 1e-6)) + warp2 * (
        warp2 / (warp1 + warp2 + 1e-6)
    )

    stitched_mesh = draw_mesh_on_warp(
        stitched[0].cpu().detach().numpy().transpose(1, 2, 0),
        mesh_trans[0].cpu().detach().numpy(),
    )

    out_dict = {}
    out_dict.update(
        warp1=warp1,
        mask1=mask1,
        warp2=warp2,
        mask2=mask2,
        stitched=stitched,
        stitched_mesh=stitched_mesh,
    )

    return out_dict
