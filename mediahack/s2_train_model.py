from pathlib import Path

import click
import pandas as pd
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_cosine_schedule_with_warmup
from xztrainer import enable_tf32, XZTrainer, XZTrainerConfig

from mediahack.intra.sharemodel import ShareDataset, ShareCollator, ShareModel, ShareTrainable


def file_name_to_id(x: str) -> int:
    return int(x.split('.')[0])


def map_class(x: int) -> int:
    if x < 6:
        return x
    elif x == 6:
        return 0
    else:
        return x - 1


def split_dict(x, include):
    return {k: v for k, v in x.items() if k in include}


@click.command()
@click.option('--transcription-path', type=Path, required=True)
@click.option('--clip-dir', type=Path, required=True)
@click.option('--ocr-dir', type=Path, required=True)
@click.option('--target-path', type=Path, required=True)
def main(transcription_path: Path, clip_dir: Path, ocr_dir: Path, target_path: Path):
    enable_tf32()
    transcriptions = pd.read_csv(transcription_path, dtype={'file': str, 'transcription': str}).fillna('')
    transcriptions = {file_name_to_id(x.file): x.transcription for x in transcriptions.itertuples()}
    targets = pd.read_csv(target_path, dtype={'Advertisement ID': int, 'Segment_num': int})
    targets = {int(x['Advertisement ID']): map_class(x['Segment_num']) for _, x in targets.iterrows()}

    train_ids, val_ids = train_test_split(list(targets.keys()), test_size=0.1, random_state=0xCAFE, stratify=list(targets.values()))
    train_ids, val_ids = set(train_ids), set(val_ids)
    train_targets, val_targets = split_dict(targets, train_ids), split_dict(targets, val_ids)
    train_transcriptions, val_transcriptions = split_dict(transcriptions, train_ids), split_dict(transcriptions, val_ids)
    train_ds, val_ds = ShareDataset(train_targets, train_transcriptions, clip_dir, ocr_dir, is_train=True), ShareDataset(val_targets, val_transcriptions, clip_dir, ocr_dir, is_train=True)

    accel = Accelerator(
        gradient_accumulation_steps=4,
        log_with='tensorboard',
        project_dir='.',
        kwargs_handlers=[]
    )

    trainer = XZTrainer(
        config=XZTrainerConfig(
            experiment_name='train-clip+audio+ocr',
            minibatch_size=4,
            minibatch_size_eval=4,
            epochs=1,
            gradient_clipping=3.0,
            optimizer=lambda module: AdamW(module.parameters(), lr=5e-5, weight_decay=1e-3),
            scheduler=lambda optimizer, total_steps: get_cosine_schedule_with_warmup(optimizer,
                                                                                     int(total_steps * 0.1),
                                                                                     total_steps),
            collate_fn=ShareCollator(),
            dataloader_persistent_workers=True,
            dataloader_num_workers=8,
            dataloader_shuffle_train_dataset=True,
            dataloader_pin_memory=True,
            log_steps=10,
            save_steps=100,
            eval_steps=100
        ),
        model=ShareModel(),
        trainable=ShareTrainable(),
        accelerator=accel
    )
    trainer.train(train_ds, val_ds)


if __name__ == '__main__':
    main()
