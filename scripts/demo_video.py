import os
import sys
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
from PIL import Image
from core.utils.visualize import get_color_pallete
from core.models import get_model

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',
                    help='model name (default: fcn32_vgg16)')
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc','pascal_aug','ade20k','citys','ycb','robocup'],
                    help='dataset name (default: pascal_voc)')
parser.add_argument('--save-folder', default='~/.torch/models',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-video', type=str, default='../datasets/xtion_video.mp4',
                    help='path to the input video file')
parser.add_argument('--outdir', default='./eval', type=str,
                    help='path to save the resulting video file')
parser.add_argument('--out_video_name', default='xtion_video.avi', type=str,
                    help='path to save the resulting video file')
parser.add_argument('--display', action='store_true', default='true')

args = parser.parse_args()

if not os.path.exists(args.input_video): raise Exception('Video file not found in :: '+args.input_video)
cap = cv2.VideoCapture(args.input_video)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.out_video_name, fourcc, 30.0, (640, 480))

def demo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(config.outdir):
        os.makedirs(config.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    #read the video file here
    model = get_model(args.model, pretrained=True, root=args.save_folder).to(device)
    model.eval()
    print('Finished loading model!')
    count = 0
    pbar = tqdm(total=150)
    while cap.isOpened():
        count += 1
        ret, image = cap.read()
        # image = Image.open(config.input_pic).convert('RGB')
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB
            image = cv2.resize(image, (640, 480))
            # image = cv2.flip(image, 1)
        except: continue
        image = cv2.GaussianBlur(image, (5, 5), 0)
        images = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(images)

        pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, args.dataset)
        # print(mask.shape)
        # print('type is :: ',type(mask))
        outname = os.path.splitext(os.path.split('tmp')[-1])[0] + '.png'
        mask.save(os.path.join(args.outdir, outname))
        mask = cv2.imread(os.path.join(args.outdir, outname)) #in BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(image, 0.5, mask, 0.5, 0.0)
        if args.display:
            cv2.imshow('output', blended)
            cv2.waitKey(1)
        out.write(blended)
        pbar.update(1)

        # if count==300: break

    cap.release()
    out.release()
    print('Done. Video file generated')

# if __name__ == '__main__':
#     demo(args)
demo(args)