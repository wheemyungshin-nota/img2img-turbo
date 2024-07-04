import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo import Pix2Pix_Turbo
from image_prep import canny_from_pil
import time

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.set_eval()

    start_time = time.time()
    data_list = os.listdir(args.input_dir)
    for input_idx, input_name in enumerate(data_list):
        print("Running... : {}/{}".format(input_idx, len(data_list)), end='\r')
        input_path = os.path.join(args.input_dir, input_name)
        # make sure that the input image is a multiple of 8
        input_image = Image.open(input_path).convert('RGB')
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        #new_height = 512
        #new_width = int(new_height * (input_image.width / input_image.height))
        #new_width = new_width - new_width % 8
        
        #input_image = input_image.filter(ImageFilter.BLUR)

        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = input_name

        # translate the image
        with torch.no_grad():
            if args.model_name == 'edge_to_image':
                canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
                canny_viz_inv = Image.fromarray(255 - np.array(canny))
                canny_viz_inv.save(os.path.join(args.output_dir, bname.replace('.png', '_canny.png')))
                c_t = F.to_tensor(canny).unsqueeze(0).cuda()
                output_image = model(c_t, args.prompt)

            elif args.model_name == 'sketch_to_image_stochastic':
                image_t = F.to_tensor(input_image) < 0.5
                c_t = image_t.unsqueeze(0).cuda().float()
                torch.manual_seed(args.seed)
                B, C, H, W = c_t.shape
                noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
                output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

            else:
                c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
                output_image = model(c_t, args.prompt)

            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        
        concat_pil = get_concat_h(input_image, output_pil)

        # save the output image
        concat_pil.save(os.path.join(args.output_dir, bname))

    print("Full Time: ", start_time - time.time())