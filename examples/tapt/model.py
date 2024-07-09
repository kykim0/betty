from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import RobertaConfig, RobertaTokenizer

from transformers import RobertaPreTrainedModel, RobertaModel

from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

from transformers.models.roberta.modeling_roberta import (
    ROBERTA_START_DOCSTRING,
    ROBERTA_INPUTS_DOCSTRING,
    RobertaLMHead,
    RobertaClassificationHead,
)


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING
)
class BaseModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)  # MLM Head

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        print("Classifier dropout : ", config.classifier_dropout)
        self.classifier = RobertaClassificationHead(config)  # Task classifier/ head

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduction="mean",
        use_mlm_head=False,
        use_classifier_head=False,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        final_output = {}

        if use_classifier_head:
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            final_output["classifier"] = SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        if use_mlm_head:
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(reduction=reduction)
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
                )

                hidden_states = outputs.hidden_states
                if reduction == "none":
                    masked_lm_loss = masked_lm_loss[labels.view(-1) != -100]
                    hidden_states = sequence_output.view(-1, sequence_output.shape[-1])
                    hidden_states = hidden_states[labels.view(-1) != -100]

            final_output["mlm"] = MaskedLMOutput(
                loss=masked_lm_loss,
                logits=prediction_scores,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )

        return final_output


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(
        self,
        in_size=1,
        hidden_size=100,
        num_layers=1,
        global_weight=False,
        zeroinit_output_layer=False,
    ):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(in_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(
            *[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.global_weight = global_weight
        if self.global_weight:
            self.scale = nn.Parameter(torch.zeros(1))

        if zeroinit_output_layer:
            self.zeroinit()

    def zeroinit(self):
        print("Zero initializing output layer weights and bias!")
        self.output_layer.weight.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        if self.global_weight:
            return torch.sigmoid(x) * 2 * torch.sigmoid(self.scale) * 2
        return torch.sigmoid(x) * 2
