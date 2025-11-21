# -*- coding:utf-8 -*-
import os

import sys
import json
from multiprocessing import Process, Queue,Manager
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model_process.Data_Load import *
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from utils.random_seed import setup_seed
from utils.metrics import MultiTaskLossWrapper
"""
加载模型
"""
from utils.model.AP_Net import AP_Net
from utils.model.Res_FAMLP import Res_MLP
from utils.model.RSE_Branch import RSE as dynamic_model
from utils.model.static_model import MLP, Conv1D
from utils.other_models.densenet import densenet121 as densenet
from utils.other_models.shufflenet import shufflenet_v2_x1_0 as shufflenet
from utils.other_models.Resnet_row import resnet50 as resnet
from utils.other_models.swin import swin_tiny_patch4_window7_224 as swin
from utils.other_models.VGG import vgg
from utils.other_models.VIT import satellite_VIT as vit
from utils.model.Res_GCMFE import GCMFE as CAT
"""
使用的数据加载
1.分两种情况进行，只需在yaml进行修改
"""
def Load_Data(train_inf,val_inf,Data_root_path,train_batch_size=64,val_batch_size=64,num_threads=4,seq_number=4):
    train_eii = DataSource(batch_size=train_batch_size, Datainf=train_inf, Data_root_path=Data_root_path,seq_number=seq_number)
    train_pipe = SourcePipeline(batch_size=train_batch_size, num_threads=num_threads, device_id=0, external_data=train_eii,
                                modeltype='train')
    train_iter = CustomDALIGenericIterator(len(train_eii) / train_batch_size, pipelines=[train_pipe],
                                           output_map=['Satellite','seg_inf', 'road_inf','risk_number','risk_rush'],
                                           last_batch_padded=False,
                                           size=len(train_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)
    val_eii = DataSource(batch_size=val_batch_size, Datainf=val_inf, Data_root_path=Data_root_path,seq_number=seq_number)
    val_pipe = SourcePipeline(batch_size=val_batch_size, num_threads=num_threads, device_id=0, external_data=val_eii,
                                modeltype='val')
    val_iter = CustomDALIGenericIterator(len(val_eii) / val_batch_size, pipelines=[val_pipe],
                                           output_map=['Satellite','seg_inf', 'road_inf','risk_number','risk_rush'],
                                           last_batch_padded=False,
                                           size=len(val_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)

    train_loader = train_iter
    val_loader = val_iter
    return train_loader,val_loader

def train(config_list,gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    for cfg in config_list:
        torch.cuda.empty_cache()
        Data_root_path = 'Dataset'
        Dataset_path = '/home/group/Group_Data'
        train_inf = pd.read_csv(os.path.join(Data_root_path, 'train.csv'))
        val_inf = pd.read_csv(os.path.join(Data_root_path, 'val.csv'))
        config_name = cfg['config_path'].split('/')[1].replace('.yaml', '')
        config_path = cfg['config_path']
        setup_seed(cfg['seed'])
        if config_name == 'MLP_ablation':
            if cfg['model']['name'] == 'FAMLP':
                model = Res_MLP(Out_flag=cfg['model']['Out_flag']).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'Conv1D':
                model = Conv1D().cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'MLP':
                model = MLP().cuda()
                experiment_name = str(cfg['model']['name'])

        elif config_name == 'Dynamic_ablation':
            if cfg['model']['name'] == 'RSE':
                model = dynamic_model(time_len=cfg['time_len'], Out_flag=True, CGAF_Flag=cfg['model']['CGAF_Flag']).cuda()
                experiment_name = 'RSE' + '_' + str(cfg['model']['CGAF_Flag'])
            if cfg['model']['name'] == 'Concat':
                model = dynamic_model(time_len=cfg['time_len'], Out_flag=True, CGAF_Flag=cfg['model']['CGAF_Flag'],
                                      fuse_flag='Concat').cuda()
                experiment_name = str(cfg['model']['name'])
            if cfg['model']['name'] == 'Add':
                model = dynamic_model(time_len=cfg['time_len'], Out_flag=True, CGAF_Flag=cfg['model']['CGAF_Flag'],
                                      fuse_flag='Add').cuda()
                experiment_name = str(cfg['model']['name'])
            if cfg['model']['name'] == 'LSTM':
                model = dynamic_model(time_len=cfg['time_len'], Out_flag=True, CGAF_Flag=cfg['model']['CGAF_Flag'],
                                      fuse_flag='LSTM').cuda()
                experiment_name = str(cfg['model']['name'])

        elif config_name == 'CV_ablation':
            if cfg['model']['name'] == 'densenet':
                model = densenet(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'shufflenet':
                model = shufflenet(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'resnet':
                model = resnet(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'swin':
                model = swin(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'vgg':
                model = vgg(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'vgg':
                model = vgg(num_classes=2).cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'vit':
                model = vit().cuda()
                experiment_name = str(cfg['model']['name'])
            elif cfg['model']['name'] == 'cat':
                model = CAT(include_top=True).cuda()
                experiment_name = str(cfg['model']['name'])

        elif config_name == 'HFAF_ablation':
            if cfg['model']['name'] == 'AP_Net':
                model = AP_Net(fuse_flag=cfg['model']['fuse_flag']).cuda()
                experiment_name = 'AP_Net_' + cfg['model']['fuse_flag']

        elif config_name == 'sequences_ablation':
            model = AP_Net(time_len=cfg['time_len']).cuda()
            experiment_name = 'AP_Net_' + str(cfg['time_len'])

        elif config_name == 'branch_ablation':
            if cfg['model']['name'] == 'AP_Net':
                model = AP_Net(fuse_flag=cfg['model']['fuse_flag']).cuda()
                experiment_name = 'AP_Net_' + cfg['model']['fuse_flag']

        elif config_name == 'guide_ablation':
            if cfg['model']['name'] == 'AP_Net':
                model = AP_Net(Guide=cfg['model']['Guide']).cuda()
                experiment_name = 'AP_Net_' + cfg['model']['Guide']
        """
        定义损失函数、模型参数
        """
        ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
        if cfg['optimizer'] == 'adamw':
            try:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(cfg['momentum'], 0.99),
                                        eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
            except:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),
                                        weight_decay=float(cfg['weight_decay']))
        elif cfg['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']),
                                   weight_decay=float(cfg['weight_decay']))
        if cfg['scheduler_flag']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg['train_epoch'],
                                                                   eta_min=cfg['eta_min'])
        best_Risk_map_mse = 2000.0
        model_path = os.path.join('model_pth', config_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_weight_path = os.path.join(model_path, experiment_name + '.pth')
        """
        数据加载
        """
        train_loader, val_loader = Load_Data(train_inf=train_inf, val_inf=val_inf, Data_root_path=Dataset_path,
                                             train_batch_size=cfg['train_batch_size'],
                                             val_batch_size=cfg['val_batch_size'], num_threads=8,
                                             seq_number=cfg['time_len'])
        train_numbers, val_numbers = len(train_inf), len(val_inf)
        """
        训练
        """
        writer = SummaryWriter("exp/" + config_path.split('/')[1].replace('.yaml', '') + '/' + experiment_name,
                               flush_secs=60)
        print('*********************{}开始训练*********************'.format(experiment_name))
        best_accident_acc = 0
        for epoch in range(cfg['train_epoch']):
            sum_train_loss = 0.0
            train_Risk_number_mse = 0.0
            train_Risk_rush_mse = 0.0
            train_Risk_rush_acc = 0.0
            train_rush_loss = 0.0

            model.train()
            train_bar = tqdm(train_loader, total=int(train_loader.__len__()), file=sys.stdout, ncols=200, position=0)
            for step, batch in enumerate(train_bar):
                """
                每次开始前将梯度清零
                """
                optimizer.zero_grad()
                Satellite_img, seg_inf, road_inf, risk_number_label, risk_rush_label = batch['Satellite'], batch['seg_inf'], \
                                                                                       batch['road_inf'], batch[
                                                                                           'risk_number'], batch[
                                                                                           'risk_rush']
                if config_path.split('/')[1].replace('.yaml', '') == "MLP_ablation":
                    risk_rush_out = model(road_inf)
                elif config_path.split('/')[1].replace('.yaml', '') == "CV_ablation":
                    risk_rush_out = model(Satellite_img)
                elif config_path.split('/')[1].replace('.yaml', '') == "Dynamic_ablation":
                    risk_rush_out = model(seg_inf)
                elif config_path.split('/')[1].replace('.yaml', '') == "HFAF_ablation":
                    risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                elif config_path.split('/')[1].replace('.yaml', '') == "sequences_ablation":
                    risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                elif config_path.split('/')[1].replace('.yaml', '') == "branch_ablation":
                    risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                elif config_path.split('/')[1].replace('.yaml', '') == 'guide_ablation':
                    risk_rush_out = model(Satellite_img, seg_inf, road_inf)

                ce_loss = ce_criterion(risk_rush_out, risk_rush_label.long())
                loss = ce_loss

                loss.backward()
                optimizer.step()
                train_predict_accident = torch.max(risk_rush_out, dim=1)[1]
                train_Risk_rush_acc += torch.eq(train_predict_accident, risk_rush_label).sum().item()
                """
                计算训练期间的指标值
                """
                sum_train_loss = sum_train_loss + loss.item()
                train_bar.desc = '训练阶段==> Loss:{:.3f}'.format(loss.item())
            if cfg['scheduler_flag']:
                scheduler.step()
            epoch_times = step + 1
            print(' [Train epoch {}] 训练阶段平均指标======>Loss:{:.3f} Acc:{:.3f}'.format(epoch + 1,
                                                                                   sum_train_loss / epoch_times,
                                                                                   train_Risk_rush_acc / train_numbers))
            # writer.add_scalars('训练指标', {"总风险Loss": round(sum_train_loss / epoch_times, 4),
            #                             "数量mse": round(train_Risk_number_mse / epoch_times, 4),
            #                             "高峰期loss": round(train_rush_loss / epoch_times, 4),
            #                             "高峰期acc": round(train_Risk_rush_acc / train_numbers, 4)}, epoch + 1)
            # 绘制总风险Loss
            writer.add_scalar('训练指标/总风险Loss', round(sum_train_loss / epoch_times, 4), epoch + 1)
            # 绘制高峰期mse
            writer.add_scalar('训练指标/高峰期acc', round(train_Risk_rush_acc / train_numbers, 4), epoch + 1)

            """
            进入评估阶段
            """
            if (epoch + 1) % cfg['train_val_times'] == 0:
                sum_val_loss = 0.0
                val_Risk_number_mse = 0.0
                val_Risk_rush_mse = 0.0
                val_Risk_rush_acc = 0.0
                val_rush_loss = 0.0
                model.eval()
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout, ncols=200, position=0)
                    output_map = []
                    label_map = []
                    for step, batch in enumerate(val_bar):
                        optimizer.zero_grad()
                        Satellite_img, seg_inf, road_inf, risk_number_label, risk_rush_label = batch['Satellite'], batch[
                            'seg_inf'], batch['road_inf'], batch['risk_number'], batch['risk_rush']
                        if config_path.split('/')[1].replace('.yaml', '') == "MLP_ablation":
                            risk_rush_out = model(road_inf)
                        elif config_path.split('/')[1].replace('.yaml', '') == "CV_ablation":
                            risk_rush_out = model(Satellite_img)
                        elif config_path.split('/')[1].replace('.yaml', '') == "Dynamic_ablation":
                            risk_rush_out = model(seg_inf)
                        elif config_path.split('/')[1].replace('.yaml', '') == "HFAF_ablation":
                            risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                        elif config_path.split('/')[1].replace('.yaml', '') == "sequences_ablation":
                            risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                        elif config_path.split('/')[1].replace('.yaml', '') == "branch_ablation":
                            risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                        elif config_path.split('/')[1].replace('.yaml', '') == 'guide_ablation':
                            risk_rush_out = model(Satellite_img, seg_inf, road_inf)
                        ce_loss = ce_criterion(risk_rush_out, risk_rush_label.long())
                        loss = ce_loss
                        """
                        计算训练期间的指标值
                        """
                        sum_val_loss = sum_val_loss + loss.item()
                        val_predict_accident = torch.max(risk_rush_out, dim=1)[1]
                        val_Risk_rush_acc += torch.eq(val_predict_accident, risk_rush_label).sum().item()

                    epoch_times = step + 1
                    print(' [Train epoch {}] 验证阶段平均指标======>Loss:{:.3f} Acc:{:.3f}'.format(epoch + 1,
                                                                                           sum_val_loss / epoch_times,
                                                                                           val_Risk_rush_acc / val_numbers))
                    # writer.add_scalars('验证指标', {"总风险Loss": round(sum_val_loss / epoch_times, 3),
                    #                             "数量mse": round(val_Risk_number_mse / epoch_times, 3),
                    #                             "高峰期mse": round(val_Risk_rush_mse / epoch_times, 3), }, epoch + 1)

                    writer.add_scalar('验证指标/loss', round(sum_val_loss / epoch_times, 4), epoch + 1)
                    # 绘制数量mse
                    # 绘制高峰期loss
                    writer.add_scalar('验证指标/Acc', round(val_Risk_rush_acc / val_numbers, 4), epoch + 1)

                    #

                    if (best_accident_acc <= val_Risk_rush_acc / val_numbers):
                        best_accident_acc = val_Risk_rush_acc / val_numbers
                        torch.save(model.state_dict(), model_weight_path)

if __name__ == '__main__':
    Data_root_path = 'Dataset'
    Dataset_path = '/home/group/Group_Data'
    train_inf = pd.read_csv(os.path.join(Data_root_path, 'train.csv'))
    val_inf = pd.read_csv(os.path.join(Data_root_path, 'val.csv'))
    """
    循环进行对config文件进行训练
    """
    train_config_path_list = ['configs/guide_ablation.yaml','configs/branch_ablation.yaml','configs/sequences_ablation.yaml']
    train_config_list = []
    for train_config_path in train_config_path_list:
        with open(train_config_path, 'r') as file:
            config = yaml.safe_load(file)
        for cfg in config:
            cfg['config_path'] = train_config_path
            train_config_list.append(cfg)

    gpu_list = ['0','1','2']
    # gpu_list = ['0']
    train_config_list = np.array_split(train_config_list, len(gpu_list))
    process_list = []
    for i in range(len(train_config_list)):
        p = Process(target=train, args=(train_config_list[i],gpu_list[i],))  # 粗略爬取
        process_list.append(p)
    for i in process_list:
        i.start()
    for i in process_list:
        i.join()







