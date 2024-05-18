import subprocess
import sys
from pathlib import Path

import click
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from xztrainer import enable_tf32

from mediahack.intra.helper import get_video_list


class WhisperModel():
    def __init__(self, model_id, device):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = torch_dtype
        self.device = device

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.model = model
        self.processor = processor
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
        self.resampler = {}
        self.vad, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=True,
                                                  onnx=True)

    def process_sample(self, filename):
        print(filename)
        try:
            code = subprocess.call(
                [
                    'ffmpeg',
                    '-y',
                    '-i',
                    filename,
                    '-vn',
                    '/tmp/audio.wav'
                ]
            )
            if code != 0:
                raise ValueError()
            clip, clip_hz = torchaudio.load('/tmp/audio.wav', backend='ffmpeg')
        except:
            return ''
        if clip_hz != 16000:
            if clip_hz not in self.resampler:
                self.resampler[clip_hz] = Resample(clip_hz, 16000)
            clip = self.resampler[clip_hz](clip)
        clip = clip.mean(dim=0)  # stereo to mono
        speech_ts = self.vad_utils[0](clip.unsqueeze(0), self.vad, sampling_rate=16000, threshold=0.1)
        full = []
        for ts in speech_ts:
            with torch.inference_mode():
                input_features = self.processor(
                    clip[ts['start']:ts['end'] + 1], sampling_rate=16000, return_tensors="pt"
                ).input_features.to(self.dtype).to(self.device)
                predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids,
                                                    temperature=0.3)
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                full.append(transcription[0])

        return ' '.join(full)


@click.command()
@click.option('--video-dir', type=Path, required=True)
@click.option('--transcription-path', type=Path, required=True)
def main(video_dir: Path, transcription_path: Path):
    enable_tf32()
    whisper = WhisperModel(model_id='openai/whisper-medium', device='cuda')
    video_list = get_video_list(video_dir)
    if transcription_path.is_file():
        df = pd.read_csv(transcription_path)
        mapper = df.dropna().to_dict(orient='records')
        existing = set(x['file'] for x in mapper)
    else:
        mapper = []
        existing = set()
    for file in tqdm(video_list):
        if file.name in existing:
            continue
        result = whisper.process_sample(str(file))
        mapper.append({'file': file.name, 'transcription': result})
        pd.DataFrame(mapper).to_csv(transcription_path, index=False)


if __name__ == '__main__':
    main()
