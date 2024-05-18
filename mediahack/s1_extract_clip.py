import subprocess
from pathlib import Path

import click
import imageio.v3 as iio
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from xztrainer import enable_tf32

from mediahack.intra.helper import get_video_list


class ClipModel():
    def __init__(self, device):
        self.device = torch.device(device)
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device).eval()

    def video_to_frames(self, input_path):
        frames = iio.imread(input_path, index=None)
        meta = iio.immeta(input_path, index=None)
        fps = int(meta['fps'] / 2)  # each half a second
        frames = [Image.fromarray(frame) for i, frame in enumerate(frames) if i % fps == 0]
        return frames

    def process_sample(self, filename):
        print(filename)
        try:
            code = subprocess.call(
                [
                    'ffmpeg',
                    '-y',
                    '-i',
                    filename,
                    '/tmp/video.mp4'
                ]
            )
            if code != 0:
                raise ValueError()
            frames = self.video_to_frames('/tmp/video.mp4')
        except:
            return None
        with torch.inference_mode():
            inputs = self.processor(images=frames)
            inputs['pixel_values'] = torch.tensor(inputs.pixel_values).to(self.device)
            outputs = self.model.vision_model(**inputs)
            return outputs.pooler_output.to(torch.device('cpu'))


@click.command()
@click.option('--video-dir', type=Path, required=True)
@click.option('--embed-dir', type=Path, required=True)
def main(video_dir: Path, embed_dir: Path):
    if not embed_dir.is_dir():
        embed_dir.mkdir(parents=True, exist_ok=True)
    enable_tf32()
    clip = ClipModel(device='cuda')
    video_list = get_video_list(video_dir)
    for file in tqdm(video_list):
        embed_file = embed_dir / (file.name.split('.')[0] + '.pt')
        result = clip.process_sample(str(file))
        if result is not None:
            with embed_file.open('wb') as f:
                torch.save(result, f)


if __name__ == '__main__':
    main()
