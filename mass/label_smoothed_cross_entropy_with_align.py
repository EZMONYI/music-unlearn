# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import math
import numpy as np

import torch
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F


@register_criterion("label_smoothed_cross_entropy_with_align")
class LabelSmoothedCrossEntropyCriterionWithAlign(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        attn_loss_weight=1.0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.attn_loss_weight = attn_loss_weight
        self.report_accuracy = report_accuracy
        self.T = 4
        self.gamma = 0.99
        self.alpha = 0.001

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing',
                            default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means \
                                  no label smoothing')
        parser.add_argument('--attn-loss-weight',
                            default=1., type=float, metavar='D',
                            help='weight of supervised attention loss')
        # fmt: on

    def forward(self, student, teacher, sample, reduce=True):
        net_output_t = student(**sample["net_input"])
        net_output_s = teacher(**sample["net_input"])

        loss, nll_loss, distill_loss = self.unlearn(
            student, teacher, net_output_s, net_output_t, sample, reduce=reduce)

        if self.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "distill_loss": distill_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def unlearn(self, model_s, model_t, net_output_s, net_output_t, sample, reduce=True):
        # T: target_loss
        lprobs_s = model_s.get_normalized_probs(net_output_s, log_probs=True)
        lprobs_s = lprobs_s.view(-1, lprobs_s.size(-1))
        target = model_t.get_targets(sample, net_output_t).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs_s.gather(dim=-1, index=target)[non_pad_mask]
        lprobs_t = model_t.get_normalized_probs(net_output_t, log_probs=False)
        lprobs_t = lprobs_t.view(-1, lprobs_t.size(-1))
        distill_loss = F.kl_div(lprobs_s, lprobs_t, size_average=False) * (self.T**2) / lprobs_s.shape[0]
        if reduce:
            nll_loss = nll_loss.sum()
        loss = self.gamma * nll_loss + self.alpha * distill_loss
        return loss, nll_loss, distill_loss
        # smooth_loss = -lprobs_s.sum(dim=-1, keepdim=True)[non_pad_mask]

        # if not ("word_ids" in sample.keys() and "attns" in net_output_t[1].keys()):
        #     attn_loss = torch.zeros(nll_loss.size())
        # else:
        #     attns = net_output_s[1]["attns"]  # list[Tensors], Tensor.shape: B, T, 80
        #     src_len = attns[0].size()[2]
        #     tgt_len = attns[0].size()[1]

        #     source_word_ids = sample["word_ids"]["source_word_ids"]
        #     target_word_ids = sample["word_ids"]["target_word_ids"]

        #     # (B, S) -> (B, 1, S) -> (B, T, S)
        #     s = source_word_ids.unsqueeze(1).repeat(1, tgt_len, 1)
        #     # (B, T) -> (B, T, 1) -> (B, T, S)
        #     t = target_word_ids.unsqueeze(2).repeat(1, 1, src_len)
        #     word_attn = torch.eq(s, t).float()  # (B, T, S)

        #     attn_word_num = torch.sum(word_attn, dim=-1, keepdim=True)
        #     mx = torch.max(attn_word_num)
        #     attn_word_num = torch.clamp(attn_word_num, 1, mx)

        #     true_word_attn = word_attn / attn_word_num

        #     # Sentence_Normalize
        #     source_sent_ids = sample["net_input"]["source_sent_ids"]
        #     target_sent_ids = sample["net_input"]["target_sent_ids"]
        #     # (B, S) -> (B, 1, S) -> (B, T, S)
        #     s = source_sent_ids.unsqueeze(1).repeat(1, tgt_len, 1)
        #     # (B, T) -> (B, T, 1) -> (B, T, S)
        #     t = target_sent_ids.unsqueeze(2).repeat(1, 1, src_len)
        #     # (B, T, S)
        #     sent_mask = torch.eq(s, t).float()
        #     sent_word_num = torch.sum(sent_mask, dim=-1, keepdim=True)
        #     mx = torch.max(sent_word_num)
        #     sent_word_num = torch.clamp(sent_word_num, 1, mx)
        #     sent_word_weight = sent_mask / sent_word_num

        #     attn_loss_each_layer = []
        #     for attn in attns:
        #         attn_loss_layer = attn - true_word_attn
        #         attn_loss_layer = attn_loss_layer.pow(2)
        #         attn_loss_layer = torch.mul(attn_loss_layer, sent_word_weight)
        #         attn_loss_layer = torch.sum(
        #             attn_loss_layer, dim=-1, keepdim=True)
        #         attn_loss_layer = attn_loss_layer.view(-1, 1)
        #         attn_loss_each_layer.append(attn_loss_layer)

        #     attn_loss_average_all_layer = torch.mean(
        #         torch.stack(attn_loss_each_layer), dim=0
        #     )
        #     attn_loss = attn_loss_average_all_layer[non_pad_mask]

        # if reduce:
        #     nll_loss = nll_loss.sum()
        #     smooth_loss = smooth_loss.sum()
        #     attn_loss = attn_loss.sum()
        # eps_i = self.eps / lprobs_s.size(-1)
        # loss = (
        #     (1.0 - self.eps) * nll_loss
        #     + eps_i * smooth_loss
        #     + self.attn_loss_weight * attn_loss
        # )
        # return loss, nll_loss, attn_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        return {
            "loss": sum(log.get("loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2),
            "nll_loss": sum(log.get("nll_loss", 0) for log in logging_outputs)
            / ntokens
            / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
