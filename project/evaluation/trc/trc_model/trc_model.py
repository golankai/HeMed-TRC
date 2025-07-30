from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertModel,
    BertForSequenceClassification,
    RobertaModel,
    RobertaForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from evaluation.trc.trc_model.trc_config import TRCBertConfig, TRCRobertaConfig


def _init_fn(model, config):
    model.config = config
    model.num_labels = config.num_labels
    model.architecture = config.architecture
    model.class_weights = config.class_weights
    model.EMS1 = config.EMS1
    model.EMS2 = config.EMS2
    model.EME1 = config.EME1
    model.EME2 = config.EME2

    classifier_dropout = (
        config.classifier_dropout
        if config.classifier_dropout is not None
        else config.hidden_dropout_prob
    )

    if model.architecture == "SEQ_CLS":
        model.post_transformer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(classifier_dropout),
        )
        model.classification_layer = nn.Linear(config.hidden_size, config.num_labels)

    if model.architecture in ["EMP", "ESS"]:
        model.post_transformer_1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(classifier_dropout),
        )

        model.post_transformer_2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(classifier_dropout),
        )
        model.relation_representation = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Dropout(classifier_dropout),
        )
        model.classification_layer = nn.Linear(config.hidden_size, config.num_labels)


def _get_entities_and_start_markers_indices(model, input_ids):
    em1_s = torch.tensor(
        [(ids == model.EMS1).nonzero().item() for ids in input_ids], device=model.device
    )
    em1_e = torch.tensor(
        [(ids == model.EME1).nonzero().item() for ids in input_ids], device=model.device
    )

    em2_s = torch.tensor(
        [(ids == model.EMS2).nonzero().item() for ids in input_ids], device=model.device
    )
    em2_e = torch.tensor(
        [(ids == model.EME2).nonzero().item() for ids in input_ids], device=model.device
    )

    return em1_s, em1_e, em2_s, em2_e


def _max_pool_entity(model, mark_start, mark_end, sequence_output):
    return (
        torch.stack(
            [
                torch.max(
                    sentence[mark_start[i] + 1 : mark_end[i]], dim=0, keepdim=True
                )[0]
                for i, sentence in enumerate(sequence_output)
            ]
        )
        .reshape(sequence_output.shape[0], -1)
        .to(model.device)
    )


def forward(
    model,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = (
        return_dict if return_dict is not None else model.config.use_return_dict
    )

    outputs = model.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if model.architecture == "SEQ_CLS":
        pooled_output = outputs[1]
        relation_representation = model.post_transformer(pooled_output)

    else:
        sequence_output = outputs[0]

        (
            entity_mark_1_s,
            entity_mark_1_e,
            entity_mark_2_s,
            entity_mark_2_e,
        ) = model._get_entities_and_start_markers_indices(input_ids)

        if model.architecture == "EMP":
            entity_1_max_pool = model._max_pool_entity(
                entity_mark_1_s, entity_mark_1_e, sequence_output
            )
            entity_1_norm = model.post_transformer_1(entity_1_max_pool)

            entity_2_max_pool = model._max_pool_entity(
                entity_mark_2_s, entity_mark_2_e, sequence_output
            )
            entity_2_norm = model.post_transformer_2(entity_2_max_pool)

            relation_representation = model.relation_representation(
                torch.cat((entity_1_norm, entity_2_norm), 1)
            )

        else:  # self.architecture == 'ESS'
            e1_start_mark_tensors = sequence_output[
                torch.arange(sequence_output.size(0)), entity_mark_1_s
            ]
            e1_start_mark_norm = model.post_transformer_1(e1_start_mark_tensors)

            e2_start_mark_tensors = sequence_output[
                torch.arange(sequence_output.size(0)), entity_mark_2_s
            ]
            e2_start_mark_norm = model.post_transformer_2(e2_start_mark_tensors)

            relation_representation = model.relation_representation(
                torch.cat((e1_start_mark_norm, e2_start_mark_norm), 1)
            )

    logits = model.classification_layer(relation_representation)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(
            weight=torch.tensor(model.class_weights, device=model.device)
        )
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class TRCBert(BertForSequenceClassification):
    config_class = TRCBertConfig

    def __init__(self, config):
        super().__init__(config)
        _init_fn(self, config)

        self.model: BertModel = BertModel.from_pretrained(config.base_lm)
        self.model.resize_token_embeddings(config.vocab_size)

    def _get_entities_and_start_markers_indices(self, input_ids):
        return _get_entities_and_start_markers_indices(self, input_ids)

    def _max_pool_entity(self, mark_start, mark_end, sequence_output):
        return _max_pool_entity(self, mark_start, mark_end, sequence_output)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


class TRCRoberta(RobertaForSequenceClassification):
    config_class = TRCRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        _init_fn(self, config)

        self.model: RobertaModel = RobertaModel.from_pretrained(config.base_lm)
        self.model.resize_token_embeddings(config.vocab_size)

    def _get_entities_and_start_markers_indices(self, input_ids):
        return _get_entities_and_start_markers_indices(self, input_ids)

    def _max_pool_entity(self, mark_start, mark_end, sequence_output):
        return _max_pool_entity(self, mark_start, mark_end, sequence_output)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
