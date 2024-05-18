from pathlib import Path

import click
import pandas as pd
import safetensors.torch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from xztrainer import enable_tf32

from mediahack.intra.sharemodel import ShareDataset, ShareModel, ShareCollator


def file_name_to_id(x: str) -> int:
    return int(x.split('.')[0])


def unmap_class(x: int) -> int:
    if x < 6:
        return x
    else:
        return x + 1


def extract_ids(video_dir: Path):
    identifiers = []
    for file in video_dir.glob('*'):
        file_id = file.name.split('.')[0]
        try:
            file_id = int(file_id)
        except ValueError:
            continue
        else:
            identifiers.append(file_id)
    return identifiers


@click.command()
@click.option('--video-dir', type=Path, required=True)
@click.option('--model-file', type=Path, required=True)
@click.option('--transcription-path', type=Path, required=True)
@click.option('--clip-dir', type=Path, required=True)
@click.option('--ocr-dir', type=Path, required=True)
@click.option('--out-path', type=Path, required=True)
@click.option('--device', type=str, default='cuda')
@click.option('--batch-size', type=int, default=4)
def main(video_dir: Path, model_file: Path, transcription_path: Path, clip_dir: Path, ocr_dir: Path, out_path: Path, device: str, batch_size: int):
    enable_tf32()
    ids = extract_ids(video_dir)
    transcriptions = pd.read_csv(transcription_path, dtype={'file': str, 'transcription': str}).fillna('')
    transcriptions = {file_name_to_id(x.file): x.transcription for x in transcriptions.itertuples()}
    infer_ds = ShareDataset(ids, transcriptions, clip_dir, ocr_dir, is_train=False)
    model = ShareModel().eval().to(device)
    model_weights = safetensors.torch.load_file(str(model_file), device=device)
    model.load_state_dict(model_weights)
    del model_weights
    predicts = []
    dataloader = DataLoader(infer_ds, shuffle=False, batch_size=batch_size, num_workers=batch_size, collate_fn=ShareCollator(),
                            pin_memory=True, persistent_workers=False)
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(**batch)
            class_from_model = torch.argmax(logits, dim=-1)
            class_from_model = class_from_model.tolist()
            class_from_model = [unmap_class(x) for x in class_from_model]
            for key, clazz in zip(batch['key'], class_from_model):
                predicts.append({'Advertisement ID': key, 'Segment_num': clazz})
    pd.DataFrame(predicts).to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
