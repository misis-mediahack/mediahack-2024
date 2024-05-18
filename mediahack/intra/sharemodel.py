from pathlib import Path

import torch
import torchmetrics
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric
from transformers import AutoModel, AutoTokenizer, RobertaConfig
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler
from xztrainer import XZTrainable, ContextType, BaseContext, ModelOutputsType, DataType, TrainContext

from mediahack.intra.ext_att_mask import get_extended_attention_mask

AUDIO_ENCODER = 'Tochka-AI/ruRoPEBert-e5-base-2k'
OCR_ENCODER = 'Tochka-AI/ruRoPEBert-e5-base-2k'
NUM_CLASSES = 18
TRUNCATE_TEXT_TO = 1024
MAX_CLIP_EMBEDDINGS = 512

CLASS_WEIGHTS = [1.0,
                 3.2580115710662065,
                 9.103051478404778,
                 5.704783471390088,
                 4.342198119895993,
                 4.816882076735579,
                 7.638659072509749,
                 6.8689022544958505,
                 6.8689022544958505,
                 6.305719370079321,
                 7.485212695101529,
                 5.988170156081223,
                 6.146674645318581,
                 4.886359925055796,
                 4.645178201658588,
                 2.5045759647810204,
                 2.8268508803107704,
                 5.255649221504019]  # sqrt(max(w)/w_i)


class ShareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AutoModel.from_pretrained(AUDIO_ENCODER, trust_remote_code=True,
                                                       pooler_type='first_token_transform')
        self.clip_transform = nn.Linear(768, 768)
        self.clip_pos = nn.Embedding(MAX_CLIP_EMBEDDINGS, 768)
        self.no_clip_embed = nn.Parameter(torch.ones((1, 1, 768)), requires_grad=True)
        nn.init.uniform_(self.no_clip_embed)

        cfg = RobertaConfig(
            hidden_size=768,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=768 * 4,
            hidden_act='gelu'
        )
        self.total_encoder = RobertaEncoder(cfg)
        self.total_pooler = RobertaPooler(cfg)
        self.cls_token = nn.Parameter(torch.ones((1, 1, 768)), requires_grad=True)
        nn.init.uniform_(self.cls_token)
        self.cls = nn.Linear(768, NUM_CLASSES)

    def forward(self, audio_ids, audio_mask, clip_embeddings, clip_mask, has_clip, **kwargs):
        audio_out = self.audio_encoder(input_ids=audio_ids, attention_mask=audio_mask).pooler_output

        hasnt_clip = ~has_clip
        clip_embeddings[hasnt_clip] = self.no_clip_embed.repeat(hasnt_clip.sum(), 1, 1)
        clip_embeddings = self.clip_transform(clip_embeddings)
        clip_pos = self.clip_pos(
            torch.arange(0, clip_embeddings.shape[1], device=clip_embeddings.device, dtype=torch.long).unsqueeze(
                0).repeat(clip_embeddings.shape[0], 1))
        clip_embeddings = clip_embeddings + clip_pos

        transformer_input_embeds = torch.cat([
            self.cls_token.repeat(audio_out.shape[0], 1, 1),
            audio_out.unsqueeze(1),
            clip_embeddings
        ], dim=1)
        transformer_input_mask = torch.cat([
            torch.ones((audio_mask.shape[0], 2), dtype=audio_mask.dtype, device=audio_mask.device),
            clip_mask
        ], dim=1)
        transformer_input_mask = get_extended_attention_mask(transformer_input_mask, dtype=torch.float32)
        transformer_outs = self.total_encoder(transformer_input_embeds, transformer_input_mask).last_hidden_state
        transformer_outs = self.total_pooler(transformer_outs)
        logits = self.cls(transformer_outs)
        return logits


class ShareTrainable(XZTrainable):
    def __init__(self):
        self._loss = None

    def on_load(self, context: TrainContext, step: int):
        self._loss = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, device=context.model_unwrapped.cls_token.device, dtype=torch.float32))

    def step(self, context: BaseContext, data: DataType) -> tuple[Tensor, ModelOutputsType]:
        model_logits = context.model(**data)
        model_predict_val = torch.argmax(model_logits, dim=-1)
        model_predict_distribution = torch.softmax(model_logits, dim=-1)
        target = data['target']
        loss = self._loss(model_logits, target)
        return loss, {
            'loss': loss,
            'predict_val': model_predict_val,
            'predict_distribution': model_predict_distribution,
            'target': target
        }

    def create_metrics(self, context_type: ContextType) -> dict[str, Metric]:
        return {
            'loss': torchmetrics.MeanMetric(),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='weighted'),
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, top_k=1),
            'accuracy_top5': torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, top_k=5),
        }

    def update_metrics(self, context_type: ContextType, model_outputs: dict[str, list], metrics: dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        metrics['f1_macro'].update(model_outputs['predict_distribution'], model_outputs['target'])
        metrics['f1_micro'].update(model_outputs['predict_distribution'], model_outputs['target'])
        metrics['f1_weighted'].update(model_outputs['predict_distribution'], model_outputs['target'])
        metrics['accuracy'].update(model_outputs['predict_distribution'], model_outputs['target'])
        metrics['accuracy_top5'].update(model_outputs['predict_distribution'], model_outputs['target'])


class ShareDataset(Dataset):
    def __init__(self, targets: dict[int, int], transcriptions: dict[int, str], clip_embed_dir: Path):
        self.targets = targets
        self.transcriptions = transcriptions
        self.clip_embed_dir = clip_embed_dir
        self.idx_to_key = {i: k for i, k in enumerate(targets)}
        self.tokenizer_audio = AutoTokenizer.from_pretrained(AUDIO_ENCODER, max_length=TRUNCATE_TEXT_TO,
                                                             truncation='longest_first')
        self.tokenizer_ocr = AutoTokenizer.from_pretrained(OCR_ENCODER, max_length=TRUNCATE_TEXT_TO,
                                                           truncation='longest_first')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        key = self.idx_to_key[item]
        target = self.targets[key]
        transcription = self.transcriptions[key] if key in self.transcriptions else ''
        transcription_enc = self.tokenizer_audio('passage: ' + transcription)
        clip_file = self.clip_embed_dir / f'{key}.pt'
        if clip_file.is_file():
            with clip_file.open('rb') as f:
                clip_embed = torch.load(f)[:MAX_CLIP_EMBEDDINGS]
            has_clip = True
        else:
            clip_embed = torch.zeros((1, 768))
            has_clip = False
        clip_mask = torch.ones((clip_embed.shape[0],), dtype=torch.long)

        return {
            'target': torch.scalar_tensor(target, dtype=torch.long),
            'audio_ids': torch.tensor(transcription_enc.input_ids[:TRUNCATE_TEXT_TO], dtype=torch.long),
            'audio_mask': torch.tensor(transcription_enc.attention_mask[:TRUNCATE_TEXT_TO], dtype=torch.long),
            'clip_embed': clip_embed,
            'clip_mask': clip_mask,
            'has_clip': torch.scalar_tensor(has_clip, dtype=torch.bool)
        }


def stack_pad_right(items: list[torch.Tensor], pad_value):
    max_len = max(x.shape[0] for x in items)
    items = [F.pad(x, (0, max_len - x.shape[0]), value=pad_value) for x in items]
    return torch.stack(items, dim=0)


def stack_pad_right_3d(items: list[torch.Tensor], pad_value):
    max_len = max(x.shape[0] for x in items)
    items = [F.pad(x, (0, 0, 0, max_len - x.shape[0]), value=pad_value) for x in items]
    return torch.stack(items, dim=0)


class ShareCollator:
    def __call__(self, batch):
        return {
            'target': torch.stack([x['target'] for x in batch]),
            'audio_ids': stack_pad_right([x['audio_ids'] for x in batch], 0),
            'audio_mask': stack_pad_right([x['audio_mask'] for x in batch], 0),
            'clip_embeddings': stack_pad_right_3d([x['clip_embed'] for x in batch], pad_value=0),
            'clip_mask': stack_pad_right([x['clip_mask'] for x in batch], pad_value=0),
            'has_clip': torch.stack([x['has_clip'] for x in batch]),
        }
