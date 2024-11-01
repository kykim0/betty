
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class LMWeightNet(nn.Module):

    def __init__(self, model_name_or_path):
        """LMWeightNet constructor.
        
        Args:
            model: Transformers sequence classification model.
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.config = self.model.config  # Hack

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_classifier_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # To compute the loss ourselves.
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = seq_classifier_outputs.logits
        return torch.sigmoid(logits)
