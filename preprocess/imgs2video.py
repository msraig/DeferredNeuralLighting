import imageio
import argparse
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataDir', type=str,  required=True)
parser.add_argument('--dstFile', type=str,  default=None)

parser.add_argument('--output_gamma', type=float,  default=2.2)
parser.add_argument('--output_scale', type=float,  default=1.0)

parser.add_argument('--fps', type=float,  default=30)
parser.add_argument('--crf', type=float,  default=5)

parser.add_argument("--HDR", dest = "HDR", action = "store_true")
parser.add_argument("--no-HDR", dest = "HDR", action = "store_false")
parser.set_defaults(HDR=False)
args,unknown = parser.parse_known_args()

def imgs2video(imgs, dst_file, fps=30, crf=5):
    quality = 10 - (args.crf / 51.0) * 10
    writer = imageio.get_writer(dst_file, format='FFMPEG', mode='I', fps=fps, codec='libx264', quality=quality)

    for img in tqdm(imgs):
        writer.append_data(img)

    writer.close()

def load_img(idx):
    if args.HDR:
        img = np.load(os.path.join(args.dataDir, 'output_%d.npy') % idx)
        img = img.astype(np.float32)
    else:
        img = cv2.imread(os.path.join(args.dataDir, 'output_%d.png' % idx))
        img = img.astype(np.float32) / 255.0  # [0, 255] ==> [0, 1]
        img = img ** 2.2
        img = img[...,::-1] # BGR ==> RGB
    
    return img



if __name__ == '__main__':
    if args.HDR:
        files = glob.glob(os.path.join(args.dataDir, "output_*.npy"))
    else:
        files = glob.glob(os.path.join(args.dataDir, "output_*.png"))
    

    imgs = []
    print("Loading images...")
    for i in tqdm(range(len(files))):
        img = load_img(i)
        out_img = img * args.output_scale
        out_img = out_img ** (1.0 / args.output_gamma)
        out_img = out_img * 255.0
        out_img = np.clip(out_img, 0, 255.0)
        out_img = out_img.astype(np.uint8)
        imgs.append(out_img)

    
    if args.dstFile is None:
        dst_file = os.path.join(args.dataDir, "combined.mp4")
    else:
        dst_file = args.dstFile
        
    print("Packing images to video... [path: %s] [imgs: #%d] " % (dst_file, len(imgs)))
    imgs2video(imgs, dst_file, args.fps, args.crf)