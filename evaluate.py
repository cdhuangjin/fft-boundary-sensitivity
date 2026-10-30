"""Paired observation-boundary evaluation for a locally supplied ConverseUSRNet."""
import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def kernel(kind):
    if kind == 'gaussian':
        y, x = np.mgrid[-3:4, -3:4]
        k = np.exp(-(x*x+y*y)/(2*1.6**2))
    else:
        k = np.zeros((7, 7), np.float64)
        for t in np.linspace(-3, 3, 201):
            x = 3+t*math.cos(math.pi/4)
            y = 3+t*math.sin(math.pi/4)
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            for yy in (y0, y0+1):
                for xx in (x0, x0+1):
                    if 0 <= yy < 7 and 0 <= xx < 7:
                        k[yy, xx] += (1-abs(x-xx))*(1-abs(y-yy))
    return (k/k.sum()).astype(np.float32)


def sample(image, index, kind, mode, center):
    if center:
        h, w = image.shape[:2]
        side = center+6
        if h < side or w < side:
            raise ValueError('Image too small for center crop and support')
        y, x = (h-side)//2, (w-side)//2
        image = image[y:y+side, x:x+side]
    support = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)[None])).float()/255
    target = support[..., 3:-3, 3:-3].clone()
    k = kernel(kind)
    weight = torch.from_numpy(k.copy()).flip(-2, -1)[None, None].repeat(3, 1, 1, 1)
    source = support if mode == 'context' else F.pad(target, [3]*4, mode=mode)
    observation = F.conv2d(source, weight, groups=3)
    noise = np.random.default_rng(12345+index).normal(0, .01, observation.shape).astype(np.float32)
    observation += torch.from_numpy(noise)
    return observation, target, torch.from_numpy(k)[None, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream', type=Path, required=True, help='ConverseNet checkout at the documented revision')
    ap.add_argument('--checkpoint', type=Path, required=True, help='Released state dict or project pilot checkpoint')
    ap.add_argument('--images', type=Path, required=True, help='Directory containing Kodak24 PNG files')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--center', type=int, default=0, help='0 for full retained image; 128 for center crop')
    ap.add_argument('--limit', type=int, default=24, help='Number of filename-sorted images, starting at index 0')
    args = ap.parse_args()
    if not 1 <= args.limit <= 24:
        ap.error('--limit must be between 1 and 24')
    paths = sorted(args.images.glob('kodim*.png'))
    if len(paths) != 24:
        ap.error('Expected all 24 Kodak images named kodim*.png, even when --limit is used')
    sys.path.insert(0, str(args.upstream.resolve()))
    from models.converse_usrnet import ConverseUSRNet
    from utils import utils_image as util
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    model = ConverseUSRNet(num_iterations=5, in_channels=64, num_blocks=7)
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['image','kernel','observation','input_psnr','input_ssim','output_psnr','output_ssim']
    def scores(pred, target):
        a = pred[0].detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
        b = target[0].cpu().numpy().transpose(1,2,0).astype(np.float64)
        q = np.rint(np.clip(a,0,1)*255).astype(np.uint8)
        ref = np.rint(b*255).astype(np.uint8)
        return float(util.calculate_psnr(q,ref,0)), float(util.calculate_ssim(q,ref,0))
    with args.output.open('w', newline='', encoding='utf-8') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(paths[:args.limit]):
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError(f'Cannot read {path}')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for kind in ('gaussian', 'motion'):
                for mode in ('circular', 'context'):
                    x, y, k = sample(im, index, kind, mode, args.center)
                    inp = scores(x, y)
                    out = model(x.to(args.device), k.to(args.device), 1)
                    if not torch.isfinite(out).all():
                        raise ValueError('Non-finite model output')
                    result = scores(out, y)
                    row = dict(image=path.name, kernel=kind, observation=mode, input_psnr=inp[0], input_ssim=inp[1], output_psnr=result[0], output_ssim=result[1])
                    writer.writerow(row)
                    f.flush()
                    print(row, flush=True)
    print('checkpoint sha256:', hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())

if __name__ == '__main__':
    main()
