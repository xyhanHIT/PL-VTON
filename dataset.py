import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
import os.path as osp
import numpy as np
import json
import cv2
import argparse
from skimage import transform

def ParseOneHot(x, num_class=None):
    h, w = x.shape
    x = x.reshape(-1)
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((x.shape[0], num_class))
    ohx[range(x.shape[0]), x] = 1
    ohx = ohx.reshape(h,w, ohx.shape[1])
    return ohx.transpose(2,0,1)

def mask_parse(parse):
    hand = np.where((parse==4) | (parse==5))
    cloth = np.where(parse==3)
    vertical_high = max(cloth[0])
    vertical_low = min(cloth[0])
    level_high1 = max(cloth[1])
    level_low1 = min(cloth[1])
    try:
        level_high2 = max(hand[1])
        level_low2 = min(hand[1])
        if level_high1 < level_high2:
            level_high = level_high2
        else:
            level_high = level_high1
        if level_low1 < level_low2:
            level_low = level_low1
        else:
            level_low = level_low2
    except:
        level_high = level_high1
        level_low = level_low1
    mask = np.zeros((vertical_high-vertical_low, level_high-level_low))
    mask_level1 = np.ones((vertical_high-vertical_low, level_low))
    mask_level2 = np.ones((vertical_high-vertical_low, parse.shape[1]-level_high))
    mask = np.concatenate((mask_level1, mask, mask_level2), axis=1)
    mask_vertical1 = np.ones((vertical_low, parse.shape[1]))
    mask_vertical2 = np.ones((parse.shape[0]-vertical_high, parse.shape[1]))
    mask = np.concatenate((mask_vertical1, mask, mask_vertical2), axis=0)
    if parse.ndim == 3:
        mask = np.repeat(mask, 3, 2)
    other_mask = (parse==1) + (parse==2) + (parse==6)
    mask = 1 - (1-mask) * (1-other_mask)
    res = parse * mask
    color = (1-mask) * 3
    res = res + color
    return vertical_high- vertical_low, mask, res

def mask_image(mask, image):
    mask = np.expand_dims(mask, 2)
    mask = np.repeat(mask, 3, 2)
    res = image * mask
    return res

def get_center(parse):
    parse_cloth = np.where(parse==3)
    parse_vertical_high = max(parse_cloth[0])
    parse_vertical_low = min(parse_cloth[0])
    parse_level_high = max(parse_cloth[1])
    parse_level_low = min(parse_cloth[1])
    parse_x = (parse_level_high + parse_level_low) / 2
    parse_y = (parse_vertical_high + parse_vertical_low) / 2
    return parse_x, parse_y

def get_pre_cloth(cloth, cloth_mask, parse, parse_cloth_high):
    cloth_region = np.where(cloth_mask==1)
    try:
        cloth_high = max(cloth_region[0]) - min(cloth_region[0])
        for i in [1.3,1.2,1.1]:
            if parse_cloth_high * i < 256:
                break
        scale = parse_cloth_high * i / cloth_high
        if scale > 1:
            scale = 1
    except:
        scale = 1

    res = transform.rescale(cloth, scale=scale, anti_aliasing=True, multichannel=True, preserve_range=True)
    res = cv2.copyMakeBorder(res, (cloth.shape[0]-res.shape[0])//2, (cloth.shape[0]-res.shape[0])//2, 
                                (cloth.shape[1]-res.shape[1])//2, (cloth.shape[1]-res.shape[1])//2, 
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
    res = cv2.resize(res, (parse.shape[1], parse.shape[0]), interpolation=cv2.INTER_AREA) 
    res_mask = transform.rescale(cloth_mask, scale=scale, preserve_range=True)
    res_mask = cv2.copyMakeBorder(res_mask, (cloth.shape[0]-res_mask.shape[0])//2, (cloth.shape[0]-res_mask.shape[0])//2, 
                                (cloth.shape[1]-res_mask.shape[1])//2, (cloth.shape[1]-res_mask.shape[1])//2, 
                                cv2.BORDER_CONSTANT, value=0)
    res_mask = cv2.resize(res_mask, (parse.shape[1], parse.shape[0]), interpolation=cv2.INTER_AREA).astype(np.uint8)

    parse_x, parse_y = get_center(parse)
    cloth_x = cloth.shape[1]/2
    cloth_y = cloth.shape[0]/2

    x = parse_x - cloth_x 
    y = parse_y - cloth_y
    cloth_xy = np.where(res_mask==np.max(res_mask))

    cloth_top = min(cloth_xy[0])
    cloth_bottom = max(cloth_xy[0])
    cloth_left = min(cloth_xy[1])
    cloth_right = max(cloth_xy[1])
    if x < 0:
        x = -(min(-x, cloth_left))
    else:
        x = min(x, cloth_right)
    if y < 0:
        y = -min(-y, cloth_top)
    else:
        y = min(y, cloth_bottom)

    M = np.float32([[1,0,x],[0,1,y]])
    res = cv2.warpAffine(res, M, (cloth.shape[1],cloth.shape[0]), borderValue=[255,255,255])
    res_mask = cv2.warpAffine(res_mask, M, (cloth.shape[1],cloth.shape[0]), borderValue=[0])
    res = res.astype('uint8')/255
    res_mask = res_mask.astype('uint8')/255
    return res.astype(np.float32), res_mask.astype(np.float32)

def ParseFine(parse):
    parse_background = (parse==0)
    parse_hair = (parse==2)
    parse_cloth1 = (parse==5)
    parse_cloth2 = (parse==6)
    parse_cloth3 = (parse==7)
    parse_low_cloth1 = (parse==8)
    parse_low_cloth2 = (parse==9)
    parse_cloth4 = (parse==10)
    parse_cloth5 = (parse==11)
    parse_low_cloth3 = (parse==12)
    parse_face = (parse==13)
    parse_left_hand = (parse==14)
    parse_right_hand = (parse==15)
    parse_leg1 = (parse==16)    
    parse_leg2 = (parse==17)   
    parse_shoe1 = (parse==18)    
    parse_shoe2 = (parse==19) 

    # -------------
    # 0：background
    # 1：hair
    # 2：face
    # 3：cloth / mask
    # 4：left arm
    # 5：right arm
    # 6：other parts
    # -------------
    parse = parse_background * 0 + \
        parse_hair * 1 + \
        parse_face * 2 + \
        (parse_cloth1 + parse_cloth2 + parse_cloth3 + parse_cloth4 + parse_cloth5) * 3 + \
        parse_left_hand * 4 + \
        parse_right_hand * 5 + \
        (parse_low_cloth1 + parse_low_cloth2 + parse_low_cloth3 + parse_leg1 + parse_leg2 + parse_shoe1 + parse_shoe2) * 6 
    
    return parse.astype("uint8")

# pose_map18
def get_pose_map18(im_name, data_path, fine_height, fine_width, radius, transform):
    pose_name = im_name.replace('.jpg', '_keypoints.json')
    with open(osp.join(data_path, 'pose', pose_name), 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1,3))
    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, fine_height, fine_width)
    r = radius
    im_pose = Image.new('L', (fine_width, fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (fine_width, fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]
    pose_map18 = pose_map.numpy()*0.5 + 0.5
    return pose_map18

class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5,), (0.5,))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth
        cloth_array = np.array(Image.open(osp.join(self.data_path, 'cloth', c_name))).astype('uint8')
        cloth = torch.from_numpy(cloth_array.astype(np.float32).transpose(2,0,1)/255)
        # cloth_mask
        mloth_array = np.array(Image.open(osp.join(self.data_path, 'cloth-mask', c_name))).astype(np.float32)
        mloth = torch.from_numpy(mloth_array).unsqueeze(0)/255
        # parse1_s
        parse1_s = np.array(Image.open(osp.join(self.data_path, 'image-parse', im_name.replace('.jpg', '.png')))).astype('uint8')
        parse1_s = ParseFine(parse1_s) # [0-19] -> [0-6]
        # image
        img_array = np.array(Image.open(osp.join(self.data_path, 'image', im_name))).astype(np.float32)
        image = torch.from_numpy(img_array.transpose(2,0,1)/255)
        
        # parse1_occ
        img_cloth_high, mask, parse1_occ = mask_parse(parse1_s)
        parse1_occ = parse1_occ.astype('uint8')

        # pre_cloth
        pre_cloth, pre_mloth = get_pre_cloth(cloth_array, mloth_array, parse1_s, img_cloth_high)
        pre_cloth = torch.from_numpy(pre_cloth.transpose(2,0,1))
        
        # parse7_occ
        parse7_s = ParseOneHot(parse1_s, num_class=7)
        parse7_s = torch.from_numpy(parse7_s.astype(np.float32))
        parse7_occ = ParseOneHot(parse1_occ, num_class=7).astype(np.float32)

        # img_occ
        img_occ = mask_image(mask, image.numpy().transpose(1,2,0))
        img_occ = torch.from_numpy(img_occ.transpose(2,0,1).astype(np.float32))

        # pose_map18
        pose_map18 = get_pose_map18(im_name, self.data_path, self.fine_height, self.fine_width, self.radius, self.transform)

        # limb
        limb_mask = (parse1_s==4) + (parse1_s==5)
        limb_mask = np.expand_dims(limb_mask, axis=2)
        limb_mask = np.concatenate((limb_mask, limb_mask, limb_mask), axis=2)
        img_limb = img_array * limb_mask
        limb = Image.fromarray(img_limb.astype('uint8'))
        img_limb = torch.from_numpy(img_limb.transpose(2,0,1)/255)
        # limb_patch
        scale = 8
        patch_height = 256 // scale
        patch_width = 192 // scale
        limb_patches = []
        for i in range(scale):
            for j in range(scale):
                limb_patch = np.array(limb.crop((j*patch_width, i*patch_height, (j+1)*patch_width, (i+1)*patch_height)))
                limb_patches.append(limb_patch)
        limb_patches = np.array(limb_patches).astype(np.float32)/255
        limbs = limb_patches[0]
        for i in range(limb_patches.shape[0]):
            if i != 0:
                limbs = np.concatenate((limbs, limb_patches[i]), axis=2)
        limbs = limbs.transpose(2,0,1)

        result = {
            'im_name':              im_name,                # list
            'cloth':                pre_cloth,              # [b, 3, 256, 192]
            'pose_map18':           pose_map18,             # [b, 18, 256, 192]
            'parse7_occ':           parse7_occ,             # [b, 7, 256, 192]
            'img_occ':              img_occ,                # [b, 3, 256, 192]
            'limbs':                limbs,                  # [b, 192, 32, 24]
            }          

        return result

    def __len__(self):
        return len(self.im_names)

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "./data/")
    parser.add_argument("--datamode", default = "test")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False, help='shuffle input data')
    opt = parser.parse_args()
    opt.data_list = opt.datamode + "_pairs.txt"

    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset)
   
    for step, inputs in enumerate(data_loader.data_loader):
        im_name = inputs['im_name']                                # list
        cloth = inputs['cloth'].cuda()                             # [b, 3, 256, 192]
        pose_map18 = inputs['pose_map18'].cuda()                   # [b, 18, 256, 192]
        parse7_occ = inputs['parse7_occ'].cuda()                   # [b, 7, 256, 192]
        img_occ = inputs['img_occ'].cuda()                         # [b, 3, 256, 192]
        limbs = inputs['limbs'].cuda()                             # [b, 3, 256, 192]

        cloth = cloth.cpu().numpy()
        pose_map18 = pose_map18.cpu().numpy()
        parse7_occ = parse7_occ.cpu().numpy()
        img_occ = img_occ.cpu().numpy()
        limbs = limbs.cpu().numpy()

        print("cloth:", np.min(cloth), np.max(cloth), cloth.shape, cloth.dtype)
        print("pose_map18:", np.min(pose_map18), np.max(pose_map18), pose_map18.shape, pose_map18.dtype)
        print("parse7_occ:", np.min(parse7_occ), np.max(parse7_occ), parse7_occ.shape, parse7_occ.dtype)
        print("img_occ:", np.min(img_occ), np.max(img_occ), img_occ.shape, img_occ.dtype)
        print("limbs:", np.min(limbs), np.max(limbs), limbs.shape, limbs.dtype)
        exit()


