import argparse

import os
import pickle

import PIL
from PIL import Image
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import UniPCMultistepScheduler

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 5.0  # Scale for classifier-free guidance
strength = 0.8
batch_size = 1

# Setting number of steps in scheduler
scheduler.set_timesteps(num_inference_steps)

nsd_stiminfo_file = '/SSD/slava/algonauts/algonauts_2023_challenge_data/nsd_stim_info_merged.pkl'
stiminfo = pd.read_pickle(nsd_stiminfo_file)
cocoId = np.array(stiminfo['cocoId'])

annotations = "/SSD/slava/algonauts/algonauts_2023_challenge_data/annotations/captions_train2017.json"
dataset = json.load(open(annotations, 'r'))

imgIdToAnns = defaultdict(list)

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)

annotation_val = "/SSD/slava/algonauts/algonauts_2023_challenge_data/annotations/captions_val2017.json"
dataset = dict()
dataset = json.load(open(annotation_val, 'r'))

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgIdToAnns[ann['image_id']].append(ann)

def load_img(img_path):
    image = Image.open(img_path).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def make_latents(sample_img, text_input):
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # take average of 5 captions
    text_embeddings = text_embeddings.mean(0).unsqueeze(0).to(torch_device)
    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    with torch.no_grad():
        init_latents = vae.encode(sample_img.to(torch_device)).latent_dist.sample()* 0.18215
     
    init_timestep = int(num_inference_steps * strength) 
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device=torch_device).long()
    
    # Adding noise to the latents 
    noise = torch.randn(init_latents.shape, generator=generator, device=torch_device, dtype=init_latents.dtype)
    noised_latents = scheduler.add_noise(init_latents, noise, timesteps)
    
    del uncond_embeddings, noise, sample_img
    
    latents = noised_latents
    
    init_latents = init_latents.cpu()
    noised_latents = noised_latents.cpu()

    # Computing the timestep to start the diffusion loop
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:].to(torch_device).long()
    
    # Iterating through defined steps
    for i,ts in enumerate(timesteps):
        # We need to scale the i/p latents to match the variance
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
        

        # Predicting noise residual using U-Net
        with torch.no_grad(): 
            u,t = unet(inp, ts, encoder_hidden_states=text_embeddings).sample.chunk(2)
            
        # Performing Guidance
        pred = u + guidance_scale*(t-u)
        
        # Conditioning  the latents
        latents = scheduler.step(pred, ts, latents).prev_sample
    
    latents = 1 / 0.18215 * latents
    latents = latents.cpu()
    
    return init_latents.squeeze().detach().numpy(), noised_latents.squeeze().detach().numpy(), latents.squeeze().detach().numpy()
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate features based on Stable Diffusion latents")
    parser.add_argument('--subj', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    global generator
    global torch_device
    
    generator = torch.cuda.manual_seed(args.seed)
    torch_device = f'cuda:{args.gpu}'
    subj = args.subj
    
    vae.to(torch_device)
    unet.to(torch_device)
    text_encoder.to(torch_device)
    
    data_dir = '/SSD/slava/algonauts/algonauts_2023_challenge_data'
    save_dir = '/SSD/slava/algonauts/stable_diffusion_v1_features'
    os.makedirs(save_dir, exist_ok=True)
    
    # Training features
    images_folder = os.path.join(data_dir, f'subj0{subj}/training_split/training_images')
    image_filenames = sorted(os.listdir(images_folder))
    print(f'Subj0{subj}: filenames are {image_filenames[:3]}')
    
    save_dir_train = os.path.join(save_dir, 'subj'+format(subj, '02'), 'train_features')
    os.makedirs(save_dir_train, exist_ok=True)
    
    print("Save directory is ", save_dir_train)    
    for img_name in tqdm(image_filenames):
        img_path = os.path.join(images_folder, img_name)
        
        sample_img = load_img(img_path)
        
        nsd_id = img_name.split("-")[-1].replace(".png", "")
        coco_id = cocoId[int(nsd_id)]

        captions = imgIdToAnns[coco_id]
        captions = [x['caption'] for x in captions]
        
        assert len(captions) > 0, "Empty captions"
        
        text_input = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        
        z, z_t, z_c = make_latents(sample_img, text_input)
                
        save_name = os.path.join(
            save_dir_train, img_name.replace(".png", ".pickle")
        )
        
        with open(save_name, 'wb') as f:
            pickle.dump({
                "z": z, "z_t": z_t, "z_c": z_c
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    # Testing features
    save_dir_test = os.path.join(save_dir, 'subj'+format(subj, '02'), 'test_features')
    os.makedirs(save_dir_test, exist_ok=True)
    
    print("Save directory is ", save_dir_test)    
    for img_name in tqdm(image_filenames):
        img_path = os.path.join(images_folder, img_name)
        
        sample_img = load_img(img_path)
        
        nsd_id = img_name.split("-")[-1].replace(".png", "")
        coco_id = cocoId[int(nsd_id)]

        captions = imgIdToAnns[coco_id]
        captions = [x['caption'] for x in captions]
        
        assert len(captions) > 0, "Empty captions"
        
        text_input = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
 
        z, z_t, z_c = make_latents(sample_img, text_input)
                
        save_name = os.path.join(
            save_dir_test, img_name.replace(".png", ".pickle")
        )
        
        with open(save_name, 'wb') as f:
            pickle.dump({
                "z": z, "c": z_t, "z_c": z_c
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    