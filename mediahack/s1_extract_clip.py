import random
import subprocess
from pathlib import Path

import click
import imageio.v3 as iio
import line_profiler_pycharm
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from xztrainer import enable_tf32

from mediahack.intra.helper import get_video_list


class ClipModel():
    def __init__(self, device, partition):
        self.device = torch.device(device)
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device).eval()
        self.part = partition

    def video_to_frames(self, input_path):
        frames = iio.imread(input_path, index=None)
        frames = [Image.fromarray(frame) for i, frame in enumerate(frames)]
        return frames

    def process_sample(self, filename):
        print(filename)
        try:
            video_tmp = f'/tmp/video{self.part}.mp4'
            code = subprocess.call(
                [
                    'ffmpeg',
                    '-y',
                    '-threads',
                    '32',
                    '-i',
                    filename,
                    '-filter:v',
                    'fps=2',
                    '-qscale',
                    '0',
                    '-an',
                    video_tmp
                ]
            )
            if code != 0:
                raise ValueError()
            frames = self.video_to_frames(video_tmp)
        except (ValueError, OSError):
            return None
        with torch.inference_mode():
            inputs = self.processor(images=frames)
            inputs['pixel_values'] = torch.tensor(inputs.pixel_values).to(self.device)
            outputs = self.model.vision_model(**inputs)
            return outputs.pooler_output.to(torch.device('cpu'))


@click.command()
@click.option('--video-dir', type=Path, required=True)
@click.option('--embed-dir', type=Path, required=True)
@click.option('--partition', type=int, default=-1)
@click.option('--n-partitions', type=int, default=-1)
def main(video_dir: Path, embed_dir: Path, partition: int, n_partitions: int):
    if not embed_dir.is_dir():
        embed_dir.mkdir(parents=True, exist_ok=True)
    enable_tf32()
    clip = ClipModel(device='cuda', partition=partition)
    video_list = get_video_list(video_dir)
    for file in tqdm(video_list):
        file_id = file.name.split('.')[0]
        if partition == -1 or int(file_id) % n_partitions == partition:
            embed_file = embed_dir / (file_id + '.pt')
            if embed_file.is_file():
                continue
            result = clip.process_sample(str(file))
            if result is not None:
                with embed_file.open('wb') as f:
                    torch.save(result, f)


if __name__ == '__main__':
    main()
