import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator
from utils import ImageDataset, weights_init_normal

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(g_a, g_b, d_a, d_b, data_loader, g_optimizer, da_optimizer, db_optimizer):
    g_a.train()
    g_b.train()
    d_a.train()
    d_b.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for real_a, real_b in train_bar:
        real_a, real_b = real_a.cuda(), real_b.cuda()

        # Generators #
        g_optimizer.zero_grad()

        fake_b = G_A(real_a)
        fake_a = G_B(real_b)
        pred_fake_b = D_B(fake_b)
        pred_fake_a = D_A(fake_a)

        # adversarial loss
        target_fake_a = torch.ones(pred_fake_a.size(), device=pred_fake_a.device)
        target_fake_b = torch.ones(pred_fake_b.size(), device=pred_fake_b.device)
        adversarial_loss = criterion_adversarial(pred_fake_b, target_fake_b) + criterion_adversarial(pred_fake_a,
                                                                                                     target_fake_a)
        # cycle loss
        cycle_loss = criterion_cycle(G_B(fake_b), real_a) + criterion_cycle(G_A(fake_a), real_b)
        # identity loss
        identity_loss = criterion_identity(G_B(real_a), real_a) + criterion_identity(G_A(real_b), real_b)

        loss = adversarial_loss + 10 * cycle_loss + 5 * identity_loss
        loss.backward()
        g_optimizer.step()

        # Discriminator A #
        da_optimizer.zero_grad()

        da_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# val for one epoch
def val(net, data_loader):
    net.eval()
    with torch.no_grad():
        for a, _ in tqdm(data_loader, desc='Generating images', dynamic_ncols=True):
            fake_a = net(a.cuda())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Model')
    # common args
    parser.add_argument('--data_root', default='monet2photo', type=str, help='Dataset root path')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs over the data to train')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--decay', default=100, type=int, help='Epoch to start linearly decaying lr to 0')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    # args parse
    args = parser.parse_args()
    data_root, batch_size, epochs, lr = args.data_root, args.batch_size, args.epochs, args.lr
    decay, save_root = args.decay, args.save_root

    # data prepare
    train_data = ImageDataset(data_root, 'train')
    test_data = ImageDataset(data_root, 'test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model setup
    G_A = Generator(3, 3).cuda()
    G_B = Generator(3, 3).cuda()
    D_A = Discriminator(3).cuda()
    D_B = Discriminator(3).cuda()
    G_A.apply(weights_init_normal)
    G_B.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # optimizer setup
    optimizer_G = Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_DA = Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_DB = Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))
    lr_scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda eiter: 1.0 - max(0, eiter - decay) / float(decay))
    lr_scheduler_DA = LambdaLR(optimizer_DA, lr_lambda=lambda eiter: 1.0 - max(0, eiter - decay) / float(decay))
    lr_scheduler_DB = LambdaLR(optimizer_DB, lr_lambda=lambda eiter: 1.0 - max(0, eiter - decay) / float(decay))

    # loss setup
    criterion_adversarial = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # training loop
    results = {'train_g_loss': [], 'train_da_loss': [], 'train_db_loss': []}
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for epoch in range(1, epochs + 1):
        g_loss, da_loss, db_loss = train(G_A, G_B, D_A, D_B, train_loader, optimizer_G, optimizer_DA, optimizer_DB)
        results['train_g_loss'].append(g_loss)
        results['train_da_loss'].append(da_loss)
        results['train_db_loss'].append(db_loss)
        val(G_A, G_B, test_loader)
        lr_scheduler_G.step()
        lr_scheduler_DA.step()
        lr_scheduler_DB.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/results.csv'.format(save_root), index_label='epoch')
        torch.save(G_A.state_dict(), '{}/GA.pth'.format(save_root))
        torch.save(G_B.state_dict(), '{}/GB.pth'.format(save_root))
        torch.save(D_A.state_dict(), '{}/DA.pth'.format(save_root))
        torch.save(D_B.state_dict(), '{}/DB.pth'.format(save_root))
