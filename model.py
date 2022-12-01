import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
import numpy as np
import argparse
from torch.autograd import Variable

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # (b,t,c,h,w)
                                              h_cur=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class MCW(nn.Module):
    def __init__(self, input_A_channels=31):
        super(MCW, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_A_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5]) # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]   # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]   # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]   # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        # flow op
        self.conv_flow1 = nn.Conv2d(256, 2, 1, 1)
        self.conv_flow2 = nn.Conv2d(128, 2, 1, 1)
        self.conv_flow3 = nn.Conv2d(64, 2, 1, 1)
        self.conv_flow4 = nn.Conv2d(64, 2, 1, 1)

        self.conv_gru = ConvGRU(input_size=(256, 192),
                                input_dim=2,
                                hidden_dim=[32, 64, 2],
                                kernel_size=(3,3),
                                num_layers=3,
                                dtype=torch.cuda.FloatTensor,
                                batch_first=True,
                                bias = True,
                                return_all_layers = False)
        self.tanh = nn.Tanh()
      
    def forward(self, pre_cloth, pose_map18, parse7_occ, image_occ):
        input = torch.cat((pre_cloth, pose_map18, parse7_occ, image_occ), axis=1)    # [b, 3+18+7+3 (31), 256, 192]
        e1 = self.encode1(input)     # [b,64,128,96]
        e2 = self.encode2(e1)        # [b,64,64,48]
        e3 = self.encode3(e2)        # [b,128,32,24]
        e4 = self.encode4(e3)        # [b,256,16,12]
        f = self.encode5(e4)         # [b,512,8,6]
      
        d4 = self.decode5(f, e4)     # [b,256,16,12]  --->  flow1
        d3 = self.decode4(d4, e3)    # [b,128,32,24]  --->  flow2
        d2 = self.decode3(d3, e2)    # [b,64,64,48]   --->  flow3
        d1 = self.decode2(d2, e1)    # [b,64,128,96]  --->  flow4
        d0 = self.decode1(d1)        # [b,64,256,192] 
        flow = self.conv_last(d0)    # [b,2,256,192]  --->  flow5

        flow1 = torch.nn.functional.interpolate(d4, scale_factor=16, mode='bilinear', align_corners=True)   # [b,256,256,192]
        flow1 = self.conv_flow1(flow1)  # [b,2,256,192]

        flow2 = torch.nn.functional.interpolate(d3, scale_factor=8, mode='bilinear', align_corners=True)
        flow2 = self.conv_flow2(flow2)

        flow3 = torch.nn.functional.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)
        flow3 = self.conv_flow3(flow3)

        flow4 = torch.nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        flow4 = self.conv_flow4(flow4)

        flow5 = flow

        flow_all = torch.cat((flow1.unsqueeze(1), flow2.unsqueeze(1), flow3.unsqueeze(1), flow4.unsqueeze(1), flow5.unsqueeze(1)), axis=1)
        layer_output_list, last_state_list = self.conv_gru(flow_all)

        gru_flow = last_state_list[0][0]
        flow_all = flow_all.permute(0,1,3,4,2)
        gru_flow = gru_flow.permute(0,2,3,1)
        gru_flow = self.tanh(gru_flow)

        gridY = torch.linspace(-1, 1, steps = 256).view(1, -1, 1, 1).expand(1, 256, 192, 1)
        gridX = torch.linspace(-1, 1, steps = 192).view(1, 1, -1, 1).expand(1, 256, 192, 1)
        grid = torch.cat((gridX, gridY), dim=3).type(gru_flow.type())

        grid = torch.repeat_interleave(grid, repeats=gru_flow.shape[0], dim=0)
        gru_flow = torch.clamp(gru_flow + grid, min=-1, max=1)

        warp_cloth = F.grid_sample(pre_cloth, gru_flow, mode='bilinear', padding_mode='border')

        return warp_cloth

class HPE(nn.Module):
    def __init__(self, input_channels=31):
        super(HPE, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5])  # [b, 64, 64, 48]
        self.encode3 = self.base_layers[5]                    # [b, 128, 32, 24]
        self.encode4 = self.base_layers[6]                    # [b, 256, 16, 12]
        self.encode5 = self.base_layers[7]                    # [b, 512, 8, 6]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=7, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.se0 = SE_Block(ch_in=input_channels)
        self.se1 = SE_Block(ch_in=64)
        self.se2 = SE_Block(ch_in=64)
        self.se3 = SE_Block(ch_in=128)
        self.se4 = SE_Block(ch_in=256)
        self.se5 = SE_Block(ch_in=512)

    def forward(self, warp_cloth, pose_map18, parse7_occ, image_occ):
        input = torch.cat((warp_cloth, pose_map18, parse7_occ, image_occ), axis=1)
        input = self.se0(input)
        e1 = self.encode1(input)
        e1 = self.se1(e1)
        e2 = self.encode2(e1)
        e2 = self.se2(e2)
        e3 = self.encode3(e2)
        e3 = self.se3(e3)
        e4 = self.encode4(e3)
        e4 = self.se4(e4)
        f = self.encode5(e4)
        f = self.se5(f)

        d4 = self.decode5(f, e4)
        d3 = self.decode4(d4, e3)
        d2 = self.decode3(d3, e2)
        d1 = self.decode2(d2, e1) 
        d0 = self.decode1(d1)       
        parse = self.conv_last(d0)
        parse = self.sigmoid(parse)
        return parse

class TryOnCoarse(nn.Module):
    def __init__(self, input_channels=47):
        super(TryOnCoarse, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                      stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   # [b, 64, 128, 96]
        self.encode2 = nn.Sequential(*self.base_layers[3:5])  
        self.encode3 = self.base_layers[5]   
        self.encode4 = self.base_layers[6]  
        self.encode5 = self.base_layers[7]   

        self.decode5 = Decoder(
            in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(
            in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(
            in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(
            in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, warp_cloth, pose_map18, parse7_t, img_occ):
        input = torch.cat((warp_cloth, pose_map18, parse7_t, img_occ), axis=1)
        e1 = self.encode1(input)     
        e2 = self.encode2(e1)      
        e3 = self.encode3(e2)        
        e4 = self.encode4(e3)        
        f = self.encode5(e4)        

        d4 = self.decode5(f, e4)     
        d3 = self.decode4(d4, e3)    
        d2 = self.decode3(d3, e2)  
        d1 = self.decode2(d2, e1)    
        d0 = self.decode1(d1)
        try_on = self.conv_last(d0)  
        try_on = self.sigmoid(try_on)
        return try_on

class TryOnFine(nn.Module):
    def __init__(self, input_channels=28, limb_channels=192):
        super(TryOnFine, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                      stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )   
        self.encode2 = nn.Sequential(*self.base_layers[3:5])  
        self.encode3 = self.base_layers[5]  
        self.encode4 = self.base_layers[6]   
        self.encode5 = self.base_layers[7]  

        self.decode5 = Decoder(
            in_channels=1024, middle_channels=512+512, out_channels=512)
        self.decode4 = Decoder(
            in_channels=512, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(
            in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(
            in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        )
        self.conv_last = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.limb_conv1 = nn.Sequential(
            nn.Conv2d(limb_channels, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.limb_conv2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def forward(self, limb, try_on_coarse, pose_map18, parse7_t):
        limb_feature1 = self.limb_conv1(limb)
        limb_feature2 = self.limb_conv2(limb_feature1)

        input = torch.cat((try_on_coarse, pose_map18, parse7_t), axis=1)
        e1 = self.encode1(input)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        f = self.encode5(e4)

        e4 = torch.cat((e4, limb_feature1), axis=1)
        f = torch.cat((f, limb_feature2), axis=1)

        d4 = self.decode5(f, e4)
        d3 = self.decode4(d4, e3)
        d2 = self.decode3(d3, e2)
        d1 = self.decode2(d2, e1)
        d0 = self.decode1(d1)

        try_on_fine = self.conv_last(d0)
        try_on_fine = self.sigmoid(try_on_fine)
        return try_on_fine

class LTF(nn.Module):
    def __init__(self, istrain=True):
        super(LTF, self).__init__()
        self.istrain = istrain
        self.try_on_model = TryOnCoarse(input_channels=31)
        self.limb_model = TryOnFine(input_channels=28, limb_channels=192)

    def forward(self, limb, warp_cloth, pose_map18, parse7_t, img_occ):
        try_on_coarse = self.try_on_model(
            warp_cloth, pose_map18, parse7_t, img_occ)
        try_on_fine = self.limb_model(
            limb, try_on_coarse, pose_map18, parse7_t)
        return try_on_fine

class PLVTON(nn.Module):
    def __init__(self, opt):
        super(PLVTON, self).__init__()

        self.mcw = MCW()
        self.mcw.load_state_dict(torch.load(
            os.path.join(opt.model_path, "MCW.pth")))

        self.hpe = torch.nn.DataParallel(HPE())
        self.hpe.load_state_dict(torch.load(
            os.path.join(opt.model_path, "HPE.pth")))

        self.ltf = LTF()
        self.ltf.load_state_dict(torch.load(
            os.path.join(opt.model_path, "LTF.pth")))

    def forward(self, cloth, pose_map18, parse7_occ, img_occ, limb):
        warp_cloth = self.mcw(cloth, pose_map18, parse7_occ, img_occ)
        parse7_t = self.hpe(warp_cloth, pose_map18, parse7_occ, img_occ)
        try_on = self.ltf(limb, warp_cloth, pose_map18, parse7_t, img_occ)
        return try_on

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./checkpoints/")
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = get_opt()

    cloth = torch.from_numpy(
        np.zeros((2, 3, 256, 192)).astype(np.float32)).cuda()
    pose_map18 = torch.from_numpy(
        np.zeros((2, 18, 256, 192)).astype(np.float32)).cuda()
    parse7_occ = torch.from_numpy(
        np.zeros((2, 7, 256, 192)).astype(np.float32)).cuda()
    img_occ = torch.from_numpy(
        np.zeros((2, 3, 256, 192)).astype(np.float32)).cuda()
    limb = torch.from_numpy(
        np.zeros((2, 192, 32, 24)).astype(np.float32)).cuda()

    model = PLVTON(opt).cuda()
    res = model(cloth, pose_map18, parse7_occ, img_occ, limb)
    print(res.shape)
