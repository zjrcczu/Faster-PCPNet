import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

from model.MTIP import MTIP

from torch.utils.data import DataLoader
from utils.pie_dataloader import dataset

from torch.utils.tensorboard import SummaryWriter

import warnings

warnings.filterwarnings('ignore')


def get_config():
    parser = argparse.ArgumentParser(description='MTIP args')
    parser.add_argument('--config', type=str, default='config/pie_config.yaml', help='config path')
    args = parser.parse_args()
    configs = argparse.Namespace(**yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader))
    return configs


def get_dataloader(cfg):
    train_dataset = dataset(cfg, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                                  drop_last=True)
    val_dataset = dataset(cfg, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                drop_last=True)
    test_dataset = dataset(cfg, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                                 drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader


def train(model, train_dataloader, optimizer, cfg, epoch):
    model.train()
    obs_time = cfg.observe_time
    train_loss = np.array([])
    crossing_pred = np.array([])
    crossing_true = np.array([])

    for i, batch in enumerate(train_dataloader):
        joint, speed, trajectory, area, seg, dist, angle, wh, crossing = batch

        past_trajectory = trajectory[:, 0:obs_time, :].clone()

        crossing = crossing.to(cfg.device)
        crossing_y_onehot = torch.FloatTensor(crossing.shape[0], 2).to(crossing.device).zero_()
        crossing_y_onehot.scatter_(1, crossing.long(), 1)

        optimizer.zero_grad()
        output = model(joint.to(cfg.device), speed.to(cfg.device), past_trajectory.to(cfg.device), area.to(cfg.device), seg.to(cfg.device), dist.to(cfg.device), angle.to(cfg.device), wh.to(cfg.device))
        loss, rate = model.train_loss(output[-1], crossing_y_onehot)
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 10 == 0:
            print('epoch: {}, batch: {}, loss: {}'.format(epoch, i, loss.item()))
        train_loss = np.append(train_loss, loss.cpu().detach().numpy())
        crossing_true = np.append(crossing_true, crossing.cpu().detach().numpy())
        crossing_pred = np.append(crossing_pred, output[-1].argmax(dim=1).cpu().detach().numpy())

    return train_loss, crossing_true, crossing_pred


def val(model, val_dataloader, cfg):
    model.eval()
    obs_time = cfg.observe_time
    val_loss = np.array([])
    crossing_pred = np.array([])
    crossing_true = np.array([])

    for i, batch in enumerate(val_dataloader):
        joint, speed, trajectory, area, seg, dist, angle, wh, crossing = batch

        crossing = crossing.to(cfg.device)
        crossing_y_onehot = torch.FloatTensor(crossing.shape[0], 2).to(crossing.device).zero_()
        crossing_y_onehot.scatter_(1, crossing.long(), 1)

        past_trajectory = trajectory[:, 0:obs_time, :].clone()

        output = model(joint.to(cfg.device), speed.to(cfg.device), past_trajectory.to(cfg.device), area.to(cfg.device), seg.to(cfg.device), dist.to(cfg.device), angle.to(cfg.device), wh.to(cfg.device))
        loss, rate = model.test_loss(output[-1], crossing_y_onehot)

        val_loss = np.append(val_loss, loss.cpu().detach().numpy())
        crossing_true = np.append(crossing_true, crossing.cpu().detach().numpy())
        crossing_pred = np.append(crossing_pred, output[-1].argmax(dim=1).cpu().detach().numpy())

    return val_loss, crossing_true, crossing_pred


def test(model, test_dataloader, cfg):
    model.eval()
    obs_time = cfg.observe_time
    test_loss = np.array([])
    crossing_pred = np.array([])
    crossing_true = np.array([])

    for i, batch in enumerate(test_dataloader):
        joint, speed, trajectory, area, seg, dist, angle, wh, crossing = batch

        crossing = crossing.to(cfg.device)
        crossing_y_onehot = torch.FloatTensor(crossing.shape[0], 2).to(crossing.device).zero_()
        crossing_y_onehot.scatter_(1, crossing.long(), 1)

        past_trajectory = trajectory[:, 0:obs_time, :].clone()

        output = model(joint.to(cfg.device), speed.to(cfg.device), past_trajectory.to(cfg.device), area.to(cfg.device), seg.to(cfg.device), dist.to(cfg.device), angle.to(cfg.device), wh.to(cfg.device))
        loss, rate = model.test_loss(output[-1], crossing_y_onehot)

        test_loss = np.append(test_loss, loss.cpu().detach().numpy())
        crossing_true = np.append(crossing_true, crossing.cpu().detach().numpy())
        crossing_pred = np.append(crossing_pred, output[-1].argmax(dim=1).cpu().detach().numpy())

    return test_loss, crossing_true, crossing_pred


def main(cfg, train_dataloader, val_dataloader, test_dataloader):
    print(cfg)
    model = MTIP(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=cfg.lr_min)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    with open(os.path.join(cfg.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    best_acc = 0
    for epoch in range(cfg.epoch):
        train_loss, crossing_true, crossing_pred = train(model, train_dataloader, optimizer, cfg, epoch)
        scheduler.step()
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        writer.add_scalar('train_loss', train_loss.mean(), epoch)

        with torch.no_grad():
            val_loss, crossing_true, crossing_pred = val(model, val_dataloader, cfg)
            writer.add_scalar('val_loss', val_loss.mean(), epoch)
            writer.add_scalar('val_crossing_acc', accuracy_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('val_crossing_f1', f1_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('val_crossing_auc', roc_auc_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('val_crossing_recall', recall_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('val_crossing_precision', precision_score(crossing_true, crossing_pred), epoch)

            if accuracy_score(crossing_true, crossing_pred) > best_acc:
                best_acc = accuracy_score(crossing_true, crossing_pred)
                torch.save(model.state_dict(), os.path.join(cfg.log_dir, 'best_model.pth'))

            test_loss, crossing_true, crossing_pred = test(model, test_dataloader, cfg)
            writer.add_scalar('test_loss', test_loss.mean(), epoch)
            writer.add_scalar('test_crossing_acc', accuracy_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('test_crossing_f1', f1_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('test_crossing_auc', roc_auc_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('test_crossing_recall', recall_score(crossing_true, crossing_pred), epoch)
            writer.add_scalar('test_crossing_precision', precision_score(crossing_true, crossing_pred), epoch)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg = get_config()
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(cfg)

    main(cfg, train_dataloader, val_dataloader, test_dataloader)
