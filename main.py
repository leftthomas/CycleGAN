import argparse
import itertools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from model import Generator, Discriminator
from utils import ImageDataset, weights_init_normal

# for reproducibility
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = False


# train for one epoch
def train(translator_a, translator_b, discriminator_a, discriminator_b, data_loader, g_optimizer, d_optimizer):
    translator_a.train()
    translator_b.train()
    discriminator_a.train()
    discriminator_b.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, dynamic_ncols=True)
    for a, b in train_bar:
        a, b = a.cuda(), b.cuda()

        # Generators #
        g_optimizer.zero_grad()

        fake_b = G_A(a)
        fake_a = G_B(b)
        pred_b = D_B(fake_b)
        pred_a = D_A(fake_a)

        # adversarial loss
        target_a = torch.ones(pred_a.size(), device=pred_a.device)
        target_b = torch.ones(pred_b.size(), device=pred_b.device)
        adversarial_loss = criterion_adversarial(pred_b, target_b) + criterion_adversarial(pred_a, target_a)
        # cycle loss
        cycle_loss = criterion_cycle(G_B(fake_b), a) + criterion_cycle(G_A(fake_a), b)
        # identity loss
        identity_loss = criterion_identity(G_B(a), a) + criterion_identity(G_A(b), b)

        loss = adversarial_loss + 10 * cycle_loss + 5 * identity_loss
        loss.backward()
        g_optimizer.step()

        # Discriminator A #
        d_optimizer.zero_grad()

        d_optimizer.step()

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
    optimizer_D = Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=lr / 2, betas=(0.5, 0.999))
    lr_scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda eiter: 1.0 - max(0, eiter - decay) / float(decay))
    lr_scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda eiter: 1.0 - max(0, eiter - decay) / float(decay))

    # loss setup
    criterion_adversarial = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # training loop
    results = {'train_loss': []}
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for epoch in range(1, epochs + 1):
        train_loss = train(G_A, G_B, D_A, D_B, train_loader, optimizer_G, optimizer_D)
        results['train_loss'].append(train_loss)
        val(G_A, G_B, test_loader)
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/results.csv'.format(save_root), index_label='epoch')
        torch.save(backbone.state_dict(), '{}/{}_model.pth'.format(save_root, save_name_pre))
