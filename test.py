import torch
import argparse
import os
from dataset import Dataset, DataLoader
from model import PLVTON
from PIL import Image
from tqdm import tqdm

torch.manual_seed(0)

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = img_tensor.clone() * 255
        tensor = tensor.cpu().clamp(0,255)            
        array = tensor.numpy().astype('uint8')
        array = array.swapaxes(0, 1).swapaxes(1, 2)
        Image.fromarray(array).save(os.path.join(save_dir, img_name), quality=95)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test_new2")
    parser.add_argument("--model_path", default="./checkpoints/")
    parser.add_argument("--datamode", default="test")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--shuffle", type=bool, default=True, help='shuffle input data')
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"
    return opt


def train_network(opt, train_loader, model):
    model.cuda()

    save_dir = os.path.join('result', opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    show_dir = os.path.join('result', "show")
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    num_data = len(os.listdir("./data/test/cloth"))
    step = (num_data // opt.batch_size) + 1

    for step in tqdm(range(step)):
        inputs = train_loader.next_batch()

        im_name = inputs['im_name']                                # list
        cloth = inputs['cloth'].cuda()                             # [b, 3, 256, 192]
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_occ = inputs['parse7_occ'].cuda()                   # [b, 7, 256, 192]
        img_occ = inputs['img_occ'].cuda()                         # [b, 3, 256, 192]
        limbs = inputs['limbs'].cuda()                             # [b, 3, 256, 192]
        image = inputs['image'].cuda()                             # [b, 3, 256, 192]

        try_on = model(cloth, pose_map18, parse7_occ, img_occ, limbs)
        save_images(try_on, im_name, save_dir)

if __name__ == "__main__":
    opt = get_opt()

    print("====================== create model ======================")
    model = torch.nn.DataParallel(PLVTON(opt)).cuda()

    print("====================== get data ======================")
    test_dataset = Dataset(opt)
    test_loader = DataLoader(opt, test_dataset)
    print("dataset size:", test_dataset.__len__())
    print("====================== generate result ======================")
    with torch.no_grad():
        train_network(opt, test_loader, model)
