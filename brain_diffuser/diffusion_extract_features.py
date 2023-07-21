import sys
sys.path.append('versatile_diffusion')

import os
import argparse
import tqdm
import numpy as np

import torch
import torchvision.transforms as tvtrans
import PIL
from PIL import Image

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean


##### START: CONSTANT VARIABLES
strength = 0.75
mixing = 0.4
cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False) 

sampler = sampler(net)
batch_size = 1

n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'

vdvae_inputs = "/SSD/slava/algonauts/brain_diffuser_vdvae_features/"
clip_large_features_dir = "/SSD/slava/algonauts/brain_diffuser_clip_vision_features/"
clip_large_txt_features_dir = "/SSD/slava/algonauts/brain_diffuser_clip_text_features/"
save_dir = "/SSD/slava/algonauts/versatile_diffusion_features"

##### END

def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x
    
def load_numpy(path):
    with open(path, 'rb') as f:
        data = np.load(f)
    return torch.tensor(data)

def load_img(path):
    img = Image.open(path)
    img = regularize_image(img)
    return img*2 - 1

def generate_features(folder, devices, save_dir, start_idx):
    net.clip.cuda(devices[0])
    net.autokl.cuda(devices[0])
    net.autokl.half()
    
    #sampler.model.model.cuda(1)
    #sampler.model.cuda(1)
    
    os.makedirs(save_dir, exist_ok=True)
    
    clip_vision = os.path.join(
        clip_large_features_dir, folder
    )
    clip_vision_paths = sorted(os.listdir(clip_vision))[start_idx:]
    
    clip_text = os.path.join(
        clip_large_txt_features_dir, folder
    )
    clip_text_paths = sorted(os.listdir(clip_text))[start_idx:]
    
    vdvae_feat = os.path.join(
        vdvae_inputs, folder
    )
    vdvae_feat_paths = sorted(os.listdir(vdvae_feat))[start_idx:]
    
    for clip_v_path, clip_t_path, vdvae_path in tqdm.tqdm(zip(clip_vision_paths,
                        clip_text_paths,
                        vdvae_feat_paths)):
        
        filename = clip_v_path
        save_path = os.path.join(save_dir, filename)
        
        if os.path.exists(save_path):
            continue
        
        print("DOING: ", save_path)
        
        clip_v_path = os.path.join(clip_vision, clip_v_path)
        clip_t_path = os.path.join(clip_text, clip_t_path)
        vdvae_path = os.path.join(vdvae_feat, vdvae_path)
        
        zin = load_img(vdvae_path)
        zin = zin.unsqueeze(0).cuda(devices[0]).half()
        
        init_latent = net.autokl_encode(zin)
    
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
        
        assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(strength * ddim_steps)
        device = f'cuda:{devices[0]}'
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
        
        dummy = ''
        utx = net.clip_encode_text(dummy)
        utx = utx.cuda(devices[1]).half()
        
        dummy = torch.zeros((1,3,224,224)).cuda(devices[0])
        uim = net.clip_encode_vision(dummy)
        uim = uim.cuda(devices[1]).half()
        
        z_enc = z_enc.cuda(devices[1])

        h, w = 512,512
        shape = [n_samples, 4, h//8, w//8]
        
        cim = load_numpy(clip_v_path).half().cuda(devices[1]).unsqueeze(0)
        ctx = load_numpy(clip_t_path).half().cuda(devices[1]).unsqueeze(0)
        
        sampler.model.model.diffusion_model.device=f'cuda:{devices[1]}'
        sampler.model.model.diffusion_model.half().cuda(devices[1])
        
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=t_enc,
            unconditional_guidance_scale=scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1-mixing),
        )
        
        # z = z.cuda(devices[0]).half()
        z = z.cpu().detach().numpy()
        assert np.isnan(z).sum()==0, 'nan value in the z'
        # save features
        
        with open(save_path, 'wb') as f:
            np.save(f, z)
        
        # # generate images
        # x = net.autokl_decode(z)

        # x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        # x = [tvtrans.ToPILImage()(xi) for xi in x]
        
        # x[0].save(save_path.replace(".npy", ".png"))


def main(args, first_gpu, second_gpu):
    torch.manual_seed(args.seed)
    
    subj_dir = f'subj0{args.subj}'
    train_dir = os.path.join(subj_dir, "train_features")
    test_dir = os.path.join(subj_dir, "test_features")
    devices = (first_gpu, second_gpu)
    
    generate_features(train_dir, devices, os.path.join(
        save_dir, train_dir
    ), args.start_idx)
    generate_features(test_dir, devices, os.path.join(
        save_dir, test_dir
    ))
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate features based on Versatile Diffusion latents")
    parser.add_argument('--subj', type=int)
    parser.add_argument('--gpus', type=str, default="1,2")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    gpus = args.gpus.split(",")
    first_gpu, second_gpu = int(gpus[0]), int(gpus[1])
    main(args, first_gpu, second_gpu)
    