import random
import subprocess
from pathlib import Path

import click
import easyocr
import imageio.v3 as iio
import line_profiler_pycharm
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from xztrainer import enable_tf32

from mediahack.intra.helper import get_video_list


class OcrModel():
    def __init__(self, partition):
        self.model = easyocr.Reader(['en', 'ru'], gpu=True)
        self.part = partition

    def video_to_frames(self, input_path):
        frames = iio.imread(input_path, index=None)
        frames = [frame for i, frame in enumerate(frames)]
        return frames

    def process_sample(self, filename):
        print(filename)
        try:
            video_tmp = f'/tmp/video-ocr{self.part}.mp4'
            code = subprocess.call(
                [
                    'ffmpeg',
                    '-y',
                    '-threads',
                    '32',
                    '-i',
                    filename,
                    '-filter:v',
                    "fps=1,scale=min'(640,iw)':-2",
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
        all_outs = []
        for frame in frames:
            outs = self.model.readtext(frame)
            outs = [x[1] for x in outs if x[2] >= 0.5]
            all_outs.append(' '.join(outs) if len(outs) > 0 else None)
        return all_outs


@click.command()
@click.option('--video-dir', type=Path, required=True)
@click.option('--ocr-dir', type=Path, required=True)
@click.option('--partition', type=int, default=-1)
@click.option('--n-partitions', type=int, default=-1)
def main(video_dir: Path, ocr_dir: Path, partition: int, n_partitions: int):
    if not ocr_dir.is_dir():
        ocr_dir.mkdir(parents=True, exist_ok=True)
    enable_tf32()
    ocr = OcrModel(partition=partition)
    video_list = get_video_list(video_dir)
    for file in tqdm(video_list):
        file_id = file.name.split('.')[0]
        if partition == -1 or int(file_id) % n_partitions == partition:
            ocr_file = ocr_dir / (file_id + '.pt')
            if ocr_file.is_file():
                continue
            result = ocr.process_sample(str(file))
            if result is not None:
                with ocr_file.open('wb') as f:
                    torch.save(result, f)


if __name__ == '__main__':
    main()
