from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CaptionGenerator(nn.Module):

    def __init__(self, word_to_idx, dim_feature=(196,512), dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        super(CaptionGenerator, self).__init__()
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector # TODO: this should be an nn.Linear unit
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self.h = nn.Linear(self.D, self.H) # get_initial_lstm, h
        self.c = nn.Linear(self.D, self.H) # get_initial_lstm, c
        self.w = nn.Embedding(self.V, self.M) # _word_embedding, w
        self.proj = nn.Linear(self.D, self.D) # _project_features, proj
        self.attn_h = nn.Linear(self.H, self.D) # _attention_layer
        self.attn = nn.Linear(self.D, 1, bias=False) # _attention_layer, attn MLP
        self.decoder = nn.Linear(self.H, self.M)
        self.decoder_out = nn.Linear(self.M, self.V)

    def _init_weights(self):
        pass

    def _get_initial_lstm(self, features):
        features = features.mean(1)
        h = F.tanh(self.h(features))
        c = F.tanh(self.c(features))
        return h, c

    def _word_embedding(self, inputs):
        return self.w(inputs)

    def _project_features(self, features):
        features_flat = features.contiguous().view(-1, self.D)
        feature_proj = self.proj(features_flat)
        return feature_proj.view(-1, self.L, self.D)

    def _attention_layer(self, features, features_proj, h):
        h_attn = F.relu(features_proj + self.attn_h(h).expand(-1, self.L, self.D)) # I'm not sure about this line
        out_attn = self.attn(h_attn.view(-1, self.D)).view(-1, self.L)
        alpha = F.softmax(out_attn)
        context = (features * alpha.view(-1, self.L, 1)).sum(1)
        return context, alpha

    def _selector(self, context, h, reuse=False):
        beta = F.sigmoid(self.selector(h))    # (N, 1)
        context = torch.mm(beta, context)
        return context, beta

    def _decode_lstm(self, x, h, context, dropout=False):
        if dropout:
            h = F.dropout(h)
        h_logits = self.decoder(h)
        if self.ctx2out:
            pass  #TODO: add this logic
        if self.prev2out:
            h_logits += x
        h_logits = F.tanh(h_logits)
        if dropout:
            h_logits = F.dropout(h_logits)
        out_logits = self.decoder_out(h_logits)
        return out_logits

    def forward(self, features, captions):
        # TODO: add batch norm
        batch_size = features.size(0)
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = (captions_out != self._null).__float__()

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)
        lstm_cell = nn.LSTMCell(self.H, self.H)
        loss = 0.0
        alpha_list = []
        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h)
                _, (h, c) = lstm_cell(torch.cat(1, [x[:, t, :], context]), (h, c))

            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout)
            loss += torch.sum(nn.CrossEntropyLoss(logits, captions_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas = torch.transpose(Variable(torch.FloatTensor(alpha_list)), 0, 1)  # (N, T, L)
            alphas_all = alphas.sum(1)  # (N, L)
            alpha_reg = self.alpha_c * torch.sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / float(batch_size)
