import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import time
from dataset import Dataset, DataLoader
# from model import Network
from model import Network
from scripts import save_checkpoint, VGGLoss, EdgeLoss
import cv2
from torch.utils.tensorboard import SummaryWriter
import lpips
from visualization import board_add_image, board_add_images, save_images
import pytorch_ssim
import torch
from torch.autograd import Variable

torch.manual_seed(0)


def Parse_7_to_1(parse):
    # 显示parse
    b, c, h, w = parse.shape
    parse_show = parse.cpu().detach().numpy()
    parse_show = parse_show.reshape(b, c, -1).transpose(0,2,1)
    res = [np.argmax(item, axis=1) for item in parse_show]
    parse_show = np.array(res).reshape(b, h, w)
    parse_show = torch.from_numpy(parse_show.astype('uint8')).unsqueeze(1)
    return parse_show    # [0,6]

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="LTF_ResNet34_1l1_2vgg_0.4edge")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument("--dataroot", default="../data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument("--display_count_loss", type=int, default=50)
    parser.add_argument("--display_count_img", type=int, default=500)
    parser.add_argument("--save_count", type=int, default=20000)
    parser.add_argument("--keep_step", type=int, default=40000)
    parser.add_argument("--decay_step", type=int, default=40000)
    parser.add_argument("--shuffle", type=bool, default=True, help='shuffle input data')
    parser.add_argument("--warp_cloth_path", default="../MCW/result/MCW_ResNet34_1l1_8lpips_0.1vt_lr0.0001_b4_e60000_train/warp-cloth/")
    parser.add_argument("--parse_t_path", default="../HPE/result/HPE_SE_3333_lr0.0001_b4_e80000_train/parse")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_step", type=int, default=0)
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"
    opt.name = opt.name + "_lr" + str(opt.lr) + "_b" + str(opt.batch_size) + "_e" + str(opt.keep_step + opt.decay_step)
    return opt

def train_network(opt, train_loader, model, board, is_train=True):
    model.cuda()

    if is_train:
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")

        model.train()

        criterionL1 = nn.L1Loss()
        criterionVGG = lpips.LPIPS(net='vgg').cuda()
        criterionEdge = EdgeLoss()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - max(0, opt.resume_step + step - opt.keep_step) / float(opt.decay_step + 1))

        all_step = opt.keep_step + opt.decay_step
    else:
        save_dir = os.path.join('result', opt.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try_on_coarse_dir = os.path.join(save_dir, 'try_on_coarse')
        if not os.path.exists(try_on_coarse_dir):
            os.makedirs(try_on_coarse_dir)
        try_on_fine_dir = os.path.join(save_dir, 'try_on_fine')
        if not os.path.exists(try_on_fine_dir):
            os.makedirs(try_on_fine_dir)
            show_dir = os.path.join(save_dir, 'show')
        if not os.path.exists(show_dir):
            os.makedirs(show_dir)

        num_data = len(os.listdir("../data/" + opt.datamode + "/cloth"))
        all_step = (num_data // opt.batch_size) + 1

    for step in range(opt.resume_step, all_step):
        inputs = train_loader.next_batch()
        
        c_name = inputs['c_name']                                  # list
        im_name = inputs['im_name']                                # list
        image = inputs['image'].cuda()                             # [b, 3, 256, 192]
        limbs = inputs['limbs'].cuda()                             # [b, 192, 32, 24]
        warp_cloth = inputs['warp_cloth'].cuda()                   # [b, 3, 256, 192]
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_t = inputs['parse7_t'].cuda()                       # [b, 7, 256, 192]
        image_occ = inputs['image_occ'].cuda()                     # [b, 3, 256, 192]

        try_on_coarse, try_on_fine = model(limbs, warp_cloth, pose_map18, parse7_t, image_occ)

        if is_train:
            visuals = [[warp_cloth, Parse_7_to_1(parse7_t), image_occ],
                        [try_on_coarse, try_on_fine, image]]

            # loss
            l1_coarse = criterionL1(try_on_coarse, image) * 1.0
            vgg_coarse = criterionVGG(try_on_coarse, image).mean() * 2.0
            edge_coarse = criterionEdge(try_on_coarse, image) * 0.4

            l1_fine = criterionL1(try_on_fine, image) * 1.0
            vgg_fine = criterionVGG(try_on_fine, image).mean() * 2.0
            edge_fine = criterionEdge(try_on_fine, image) * 0.4

            loss_coarse = l1_coarse + vgg_coarse + edge_coarse 
            loss_fine = l1_fine + vgg_fine + edge_fine
            loss = loss_coarse + loss_fine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if ((step+1) % opt.display_count_loss == 0) or ((step+1)<2):
                board.add_scalar('l1_coarse', l1_coarse.item(), step+1)
                board.add_scalar('vgg_coarse', vgg_coarse.item(), step+1)
                board.add_scalar('edge_coarse', edge_coarse.item(), step+1)
                board.add_scalar('loss_coarse', loss_coarse.item(), step+1)
                
                board.add_scalar('l1_fine', l1_fine.item(), step+1)
                board.add_scalar('vgg_fine', vgg_fine.item(), step+1)
                board.add_scalar('edge_fine', edge_fine.item(), step+1)
                board.add_scalar('loss_fine', loss_fine.item(), step+1)
                board.add_scalar('loss', loss.item(), step+1)

                print('step: %6d, l1_c: %4f, vgg_c: %4f, edge_c: %4f, loss_c: %4f, ' % (step+1, l1_coarse.item(), vgg_coarse.item(), edge_coarse.item(), loss_coarse.item()), flush=True)
                print('%12d, l1_f: %4f, vgg_f: %4f, edge_f: %4f, loss_f: %4f, ' % (step+1, l1_fine.item(), vgg_fine.item(), edge_fine.item(), loss_fine.item()), flush=True)
                print('%12d, loss: %4f' % (step+1, loss.item()), flush=True)

                # 训练信息存储
                ckpt_path = os.path.join(opt.checkpoint_dir, opt.name)
                if not os.path.exists(ckpt_path):
                    os.mkdir(ckpt_path)
                time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                content_info1 = 'step: %6d, l1_c: %4f, vgg_c: %4f, edge_c: %4f, loss_c: %4f, ' % (step+1, l1_coarse.item(), vgg_coarse.item(), edge_coarse.item(), loss_coarse.item())
                content_info2 = '%12d, l1_f: %4f, vgg_f: %4f, edge_f: %4f, loss_f: %4f, ' % (step+1, l1_fine.item(), vgg_fine.item(), edge_fine.item(), loss_fine.item())
                content_info3 = '%12d, loss: %4f' % (step+1, loss.item())

                info = time_info + '\n' + content_info1 + '\n' + content_info2 + '\n' + content_info3 + '\n'
                info_path = os.path.join(opt.checkpoint_dir, opt.name, 'loss_info.txt')
                with open(info_path, "a") as f:
                    f.write(info)

            if (step+1) in [100, 200, 300, 400, 500]:
                board_add_images(board, 'combine', visuals, step+1)

            if (step+1) % opt.display_count_img == 0:
                board_add_images(board, 'combine', visuals, step+1)

            if (step+1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

        else:
            save_images(try_on_coarse, im_name, try_on_coarse_dir)
            save_images(try_on_fine, im_name, try_on_fine_dir)
            show = torch.cat((warp_cloth, try_on_coarse, try_on_fine, image), axis=3)
            save_images(show, im_name, show_dir)

            print('step: %d' % (step+1), flush=True)


if __name__ == "__main__":
    opt = get_opt()

    train_dataset = Dataset(opt)
    train_loader = DataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    model = Network().cuda()
    train_network(opt, train_loader, model, board, is_train=True)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'final.pth'))

    # -------------------------------------
    opt_test = opt
    opt_test.datamode = "test"
    opt_test.data_list = "test_pairs.txt"
    opt.warp_cloth_path = opt.warp_cloth_path.replace("_train", "_test")
    opt.parse_t_path = opt.parse_t_path.replace("_train", "_test")
    test_dataset = Dataset(opt_test)
    test_loader = DataLoader(opt_test, test_dataset)
    with torch.no_grad():
        train_network(opt_test, test_loader, model, board, is_train=False)
