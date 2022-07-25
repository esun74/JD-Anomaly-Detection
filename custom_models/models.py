#code from https://github.com/bit-ml/date/blob/master/simpletransformers/simpletransformers/custom_models/models.py

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    BertModel,
    BertPreTrainedModel,
    DistilBertModel,
    ElectraForMaskedLM,
)

from transformers.modeling_electra import (
    ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    ElectraConfig,
    ElectraModel,
    ElectraPreTrainedModel,
)


class ElectraRMD(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, disc_hid, seq_len, output_size):
        super().__init__()

        self.fc1 = nn.Linear(disc_hid, disc_hid)
        self.fc2 = nn.Linear(disc_hid, output_size)

    def forward(self, discriminator_hidden_states):
        discriminator_sequence_output = discriminator_hidden_states[:, 0, :]

        input_ = torch.squeeze(discriminator_sequence_output, dim=1)

        fc1_out = F.relu(self.fc1(input_))
        fc2_out = self.fc2(fc1_out)

        return fc2_out

class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config, output_size=6, extra_args=None):
        super().__init__(config)
        self.extra_args = extra_args
        if "vanilla_electra" in extra_args and extra_args["vanilla_electra"] != False:
            self.electra = ElectraModel(config)
            self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
            self.init_weights()
        else:
            disc_hid = config.hidden_size
            seq_len = extra_args["max_seq_length"]

            self.electra = ElectraModel(config)
            self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
            self.rmd_predictions = ElectraRMD(disc_hid, seq_len, output_size)
            self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):
        r"""
        labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates the token is an original token,
            ``1`` indicates the token was replaced.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.ElectraConfig`) and inputs:
        loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss of the ELECTRA objective.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`)
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when :obj:`config.output_hidden_states=True`):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import ElectraTokenizer, ElectraForPreTraining
        import torch
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]
        """

        discriminator_hidden_states = self.electra(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions,
        )

        discriminator_sequence_output = discriminator_hidden_states[0]

        representations = discriminator_sequence_output[:, 0, :]

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            rmd_output = self.rmd_predictions(discriminator_sequence_output)

        logits = self.discriminator_predictions(discriminator_sequence_output)
        rtd_output = logits

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            return (rmd_output, loss, representations, rtd_output)  # (loss), scores, (hidden_states), (attentions)

        return loss, representations, rtd_output


class ElectraForLanguageModelingModel(PreTrainedModel):
    def __init__(self, config, output_size=100, extra_args=None, **kwargs):
        super(ElectraForLanguageModelingModel, self).__init__(config, **kwargs)
        self.extra_args = extra_args
        if "generator_config" in kwargs:
            generator_config = kwargs["generator_config"]
        else:
            generator_config = config
        self.generator_model = ElectraForMaskedLM(generator_config)
        if "discriminator_config" in kwargs:
            discriminator_config = kwargs["discriminator_config"]
        else:
            discriminator_config = config
        self.discriminator_model = ElectraForPreTraining(discriminator_config, output_size=output_size, extra_args=self.extra_args)
        self.vocab_size = generator_config.vocab_size
        if kwargs.get("tie_generator_and_discriminator_embeddings", True):
            self.tie_generator_and_discriminator_embeddings()
        if "random_generator" in kwargs:
            self.random_generator = kwargs['random_generator']
            print(f'IN MODEL: RANDOM GENERATOR: {self.random_generator}')

    def tie_generator_and_discriminator_embeddings(self):
        gen_embeddings = self.generator_model.electra.embeddings
        disc_embeddings = self.discriminator_model.electra.embeddings

        # tie word, position and token_type embeddings
        gen_embeddings.word_embeddings.weight = disc_embeddings.word_embeddings.weight
        gen_embeddings.position_embeddings.weight = disc_embeddings.position_embeddings.weight
        gen_embeddings.token_type_embeddings.weight = disc_embeddings.token_type_embeddings.weight

    def forward(self, inputs, masked_lm_labels, attention_mask=None, token_type_ids=None, replace_tokens=True):
        d_inputs = inputs.clone()

        if replace_tokens:
            g_out = self.generator_model(
                inputs, masked_lm_labels=masked_lm_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

            sample_probs = torch.softmax(g_out[1], dim=-1, dtype=torch.float32)
            sample_probs = sample_probs.view(-1, self.vocab_size)

            sampled_tokens = torch.multinomial(sample_probs, 1).view(-1)
            sampled_tokens = sampled_tokens.view(d_inputs.shape[0], -1)

            if self.random_generator:
                sampled_tokens = torch.randint(5, self.vocab_size-1, sampled_tokens.shape)
                sampled_tokens = sampled_tokens.cuda()

            # labels have a -100 value to mask out loss from unchanged tokens.
            mask = masked_lm_labels.ne(-100)

            # replace the masked out tokens of the input with the generator predictions.
            d_inputs[mask] = sampled_tokens[mask]

            # turn mask into new target labels.  1 (True) for corrupted, 0 otherwise.
            # if the prediction was correct, mark it as uncorrupted.
            correct_preds = sampled_tokens == masked_lm_labels
            d_labels = mask.long()
            d_labels[correct_preds] = 0
        else:
            # no element is corrupted
            d_labels = torch.zeros_like(masked_lm_labels)

        # run token classification, predict whether each token was corrupted.
        output = self.discriminator_model(
            d_inputs, labels=d_labels, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        if not ("vanilla_electra" in self.extra_args and self.extra_args["vanilla_electra"] != False):
            d_out, d_loss, representations, rtd_out = output
            if replace_tokens:
                g_loss = g_out[0]
                g_scores = g_out[1]
            else:
                g_loss = None
                g_scores = None
            d_scores = d_out #useless?

            return g_loss, g_scores, d_out, d_inputs, d_loss, d_scores, d_labels, representations, rtd_out
        else:
            d_loss, representations, rtd_out = output
            g_loss = g_out[0]
            # d_loss = output_electra[0]
            g_scores = g_out[1]

            return g_loss, g_scores, d_inputs, d_loss, d_labels, representations, rtd_out

