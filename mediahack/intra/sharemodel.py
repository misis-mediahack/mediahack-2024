import torch
import torchmetrics
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from xztrainer import XZTrainable, ContextType, BaseContext, ModelOutputsType, DataType

AUDIO_ENCODER = 'Tochka-AI/ruRoPEBert-e5-base-2k'
NUM_CLASSES = 18
TRUNCATE_TEXT_TO = 1024


class ShareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AutoModel.from_pretrained(AUDIO_ENCODER, trust_remote_code=True, pooler_type='first_token_transform')
        self.cls = nn.Linear(768, NUM_CLASSES)

    def forward(self, audio_ids, audio_mask, **kwargs):
        audio_out = self.audio_encoder(input_ids=audio_ids, attention_mask=audio_mask).pooler_output
        logits = self.cls(audio_out)
        return logits


class ShareTrainable(XZTrainable):
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

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
    def __init__(self, targets: dict[int, int], transcriptions: dict[int, str]):
        self.targets = targets
        self.transcriptions = transcriptions
        self.idx_to_key = {i: k for i, k in enumerate(targets)}
        self.tokenizer_audio = AutoTokenizer.from_pretrained(AUDIO_ENCODER)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        target = self.targets[self.idx_to_key[item]]
        transcription = self.transcriptions[self.idx_to_key[item]]
        transcription_enc = self.tokenizer_audio(transcription)

        return {
            'target': torch.scalar_tensor(target, dtype=torch.long),
            'audio_ids': torch.tensor(transcription_enc.input_ids[:TRUNCATE_TEXT_TO], dtype=torch.long),
            'audio_mask': torch.tensor(transcription_enc.attention_mask[:TRUNCATE_TEXT_TO], dtype=torch.long)
        }


def stack_pad_left(items: list[torch.Tensor], pad_value):
    max_len = max(x.shape[0] for x in items)
    items = [F.pad(x, (max_len - x.shape[0], 0), value=pad_value) for x in items]
    return torch.stack(items, dim=0)


class ShareCollator:
    def __call__(self, batch):
        return {
            'target': torch.stack([x['target'] for x in batch]),
            'audio_ids': stack_pad_left([x['audio_ids'] for x in batch], 0),
            'audio_mask': stack_pad_left([x['audio_mask'] for x in batch], 0)
        }
