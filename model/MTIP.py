import argparse

import yaml

import torch
import torch.nn as nn

from model.module.PedTempGCN import PedGraphConvolution
from model.polar_predict import predictor

class MTIP(nn.Module):
    def __init__(self, cfg):
        super(MTIP, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.device = cfg.device

        self.MLP = predictor()

        # kps branch
        self.gcn1 = PedGraphConvolution(cfg, cfg.kps_ch, cfg.embedding_ch)
        if cfg.seg:
            self.seg_conv = nn.Sequential(
                nn.MaxPool2d(10),
                nn.Conv2d(1, cfg.embedding_ch, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(cfg.embedding_ch),
                nn.AdaptiveAvgPool2d(1),
                nn.ReLU()
            )
        # self.gcn2 = PedGraphConvolution(cfg, 32, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # sequence embedding
        # self.gru = nn.GRU(1088, 1024, 1, batch_first=True)
        self.kps_fc_out = nn.Linear(cfg.embedding_ch, 2)

        # speed branch
        if cfg.speed:
            self.speed_conv = nn.Sequential(
                    nn.Conv1d(1, cfg.embedding_ch, 3, bias=False),
                    nn.BatchNorm1d(cfg.embedding_ch),
                    nn.AdaptiveAvgPool1d(1),
                    nn.ReLU()
                    )
            self.speed_fc_out = nn.Linear(cfg.embedding_ch, 2)

        # trajectory branch
        self.center_conv = nn.Sequential(
            nn.Conv1d(2, cfg.embedding_ch, 3, bias=False),
            nn.BatchNorm1d(cfg.embedding_ch),
            nn.ReLU(),
        )
        self.center_gru = nn.GRU(cfg.embedding_ch, cfg.embedding_ch, 1, batch_first=True)
        if cfg.center:
            self.center_fc_out = nn.Linear(cfg.embedding_ch, 2)

        # area branch
        if cfg.area:
            self.area_conv = nn.Sequential(
                nn.Conv1d(1, cfg.embedding_ch, 3, bias=False),
                nn.BatchNorm1d(cfg.embedding_ch),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU()
            )
            self.area_fc_out = nn.Linear(cfg.embedding_ch, 2)

        if cfg.dist:
            self.dist_conv = nn.Sequential(
                nn.Conv1d(6, cfg.embedding_ch, 3, bias=False),
                nn.BatchNorm1d(cfg.embedding_ch),
                nn.AdaptiveAvgPool1d(1),
                nn.ReLU()
            )
            self.dist_fc_out = nn.Linear(cfg.embedding_ch, 2)

        self.wh_conv = nn.Sequential(
            nn.Conv1d(2, cfg.embedding_ch, 3, bias=False),
            nn.BatchNorm1d(cfg.embedding_ch),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU()
        )

        # classification branch
        self.dropout = nn.Dropout(0.25)
        self.fc_scale = nn.Sequential(
            nn.Linear(cfg.embedding_ch, cfg.hidden_ch),
            nn.ReLU(),
            nn.Linear(cfg.hidden_ch, cfg.embedding_ch)
        )
        self.fc_out = nn.Linear(cfg.embedding_ch, 2)
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, kps, speed, center, area, seg, dist, angle, wh):
        y_all = torch.tensor([], dtype=torch.float32).cuda()
        gcn_res = self.gcn1(kps) # N T V C
        res = self.gap(gcn_res.view(self.batch_size, self.cfg.embedding_ch, -1, 17)).squeeze(-1).squeeze(-1)
        # res = torch.ones(self.batch_size, 64).to(self.device)
        if self.cfg.seg:
            # print('seg')
            seg_emb = self.seg_conv(seg).squeeze(-1).squeeze(-1)
            res = res * seg_emb

        if self.cfg.speed:
            # print('speed')
            speed_emb = self.speed_conv(speed.permute(0, 2, 1)).squeeze(-1)
            res = res * speed_emb

        center_c = torch.cat((torch.zeros(self.batch_size, 1, 3).to(self.cfg.device), center[:, 1:, :] - center[:, 0:-1, :]), dim=1)
        # center_c = center_c[:, :, 0:2].clone()
        center_emb = self.center_conv(center_c[:, :, 0:2].permute(0, 2, 1))  # .squeeze(-1)
        center_out, center_state = self.center_gru(center_emb.permute(0, 2, 1))

        if self.cfg.center:
            # print('center')
            center_emb = center_state.permute(1, 0, 2).squeeze(1)
            res = res * center_emb

        if self.cfg.area:
            # print('area')
            area_emb = self.area_conv(area.permute(0, 2, 1)).squeeze(-1)
            res = res * area_emb

        if self.cfg.seg:
            # print('seg')
            seg_emb = self.seg_conv(seg).squeeze(-1).squeeze(-1)
            res = res * seg_emb

        if self.cfg.dist:
            # print('dist')
            feature = self.MLP(dist.permute(0, 2, 1))
            dist_emb = torch.cat((dist.permute(0, 2, 1), feature), dim=-1)
            dist_emb = self.dist_conv(dist_emb).squeeze(-1)
            res = res * dist_emb


        wh_emb = self.wh_conv(wh.permute(0, 2, 1).float()).squeeze(-1)
        res = res * wh_emb

        # res = self.dropout(res)
        return [self.softmax(self.fc_out(self.relu(res)))]

    def test_loss(self, y, y_true):
        L_INTENT = nn.CrossEntropyLoss()(y, y_true)
        return L_INTENT, 0

    def train_loss(self, y, y_true):
        L_INTENT = nn.CrossEntropyLoss()(y, y_true)
        return L_INTENT, 0


if __name__ == '__main__':
    cfg = argparse.Namespace(**yaml.load(open('../config/pie_config.yaml', 'r'), Loader=yaml.FullLoader))
    model = MTIP(cfg).to(cfg.device)
    # bbx = torch.randn(cfg.batch_size, 16, 4).to(cfg.device)
    trajectory = torch.randn(cfg.batch_size, 16, 3).to(cfg.device)
    kps = torch.randn(cfg.batch_size, 16, 17, 3).to(cfg.device)
    speed = torch.randn(cfg.batch_size, 16, 1).to(cfg.device)
    area = torch.randn(cfg.batch_size, 16, 1).to(cfg.device)
    dist = torch.randn(cfg.batch_size, 16, 6).to(cfg.device)
    angle = torch.randn(cfg.batch_size, 16, 1).to(cfg.device)
    wh = torch.randn(cfg.batch_size, 16, 2).to(cfg.device)
    seg = torch.randn(cfg.batch_size, 16, 1).to(cfg.device)
    # thop
    from thop import profile

    flops, params = profile(model, inputs=(kps,speed,trajectory, area, seg, dist, angle, wh))
    print(flops / 1e9, params / 1e6)
