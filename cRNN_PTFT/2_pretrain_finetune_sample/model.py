#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_structs import pad_seq, mask_seq
from utils import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
 
class MultiGRU(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        #self.voc_size = voc_size
        #print(voc_size)
        self.embedding = nn.Embedding(voc_size, 768)
        self.gru_1 = nn.GRUCell(768, 1024)
        self.gru_2 = nn.GRUCell(1024, 1024)
        self.gru_3 = nn.GRUCell(1024, 1024)
        self.linear = nn.Linear(1024, voc_size)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = Variable(torch.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 1024))

class MultiGRUHead(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size, pretrained_weights=None):
        super(MultiGRUHead, self).__init__()
        #self.voc_size = voc_size
        #print(voc_size)
        self.embedding = nn.Embedding(voc_size, 768)
        self.dense = nn.Linear(1, 768) 
        self.gru_1 = nn.GRUCell(768, 1024)
        self.gru_2 = nn.GRUCell(1024, 1024)
        self.gru_3 = nn.GRUCell(1024, 1024)
        self.linear = nn.Linear(1024, voc_size)
        if pretrained_weights:
            self.load_state_dict(pretrained_weights, strict=False)

    def forward(self, x, e, h):
        x = self.embedding(x)
        e = self.dense(e)
        x_e = x + e
        h_out = Variable(torch.zeros(h.size()))
        x_e = h_out[0] = self.gru_1(x_e, h[0])
        x_e = h_out[1] = self.gru_2(x_e, h[1])
        x_e = h_out[2] = self.gru_3(x_e, h[2])
        x_e = self.linear(x_e)
        return x_e, h_out

    def init_h(self, batch_size):
        # Initial cell state is zero
        return Variable(torch.zeros(3, batch_size, 1024))

class RNN():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRU(voc.vocab_size)
        self.voc_size = voc.vocab_size
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc

    def likelihood(self, target):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenghth) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        #max_length = max[seq.size for seq in target]
        #max_length = max([seq.size(0) for seq in arr])
        #collated_arr = Variable(torch.zeros(len(arr), max_length))
        #for i, seq in enumerate(arr):
        #    collated_arr[i, :seq.size(0)] = seq
        #return collated_arr
        seq_lens, target = pad_seq(target)
        target = Variable(target)
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        # x is one step behand target, use step n-1 of x to generate step n of target
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = Variable(torch.zeros(batch_size))
        log_losses = Variable(torch.zeros(batch_size,seq_length))
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], h)
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            #log_probs += NLLLoss(log_prob, target[:, step])
            log_losses[:, step] = NLLLoss(log_prob, target[:,step])
            entropy += -torch.sum((log_prob * prob), 1)
        log_losses = mask_seq(log_losses, seq_lens)
        
        log_all = torch.sum(log_losses, 1)
        return log_all, entropy

    def sample(self, batch_size, max_length=666):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token_init = Variable(torch.zeros(batch_size).long())
        start_token = start_token_init.clone()
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token

        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x, h)
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            x = torch.multinomial(prob,1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

class RNNHead():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc):
        self.rnn = MultiGRUHead(voc.vocab_size)
        self.voc_size = voc.vocab_size
        if torch.cuda.is_available():
            self.rnn.cuda()
        self.voc = voc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def likelihood(self, target, energy):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_lenghth) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        #max_length = max[seq.size for seq in target]
        #max_length = max([seq.size(0) for seq in arr])
        #collated_arr = Variable(torch.zeros(len(arr), max_length))
        #for i, seq in enumerate(arr):
        #    collated_arr[i, :seq.size(0)] = seq
        #return collated_arr
        seq_lens, target = pad_seq(target)
        target = Variable(target)
        batch_size, seq_length = target.size()
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        # x is one step behand target, use step n-1 of x to generate step n of target
        x = torch.cat((start_token, target[:, :-1]), 1)
        h = self.rnn.init_h(batch_size)

        log_probs = Variable(torch.zeros(batch_size))
        log_losses = Variable(torch.zeros(batch_size,seq_length))
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits, h = self.rnn(x[:, step], energy, h)
            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)
            #log_probs += NLLLoss(log_prob, target[:, step])
            log_losses[:, step] = NLLLoss(log_prob, target[:,step])
            entropy += -torch.sum((log_prob * prob), 1)
        log_losses = mask_seq(log_losses, seq_lens)
        
        log_all = torch.sum(log_losses, 1)
        return log_all, entropy

    def sample(self, batch_size, energy, max_length=666):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample 
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token_init = Variable(torch.zeros(batch_size).long())
        start_token = start_token_init.clone()
        start_token[:] = self.voc.vocab['GO']
        h = self.rnn.init_h(batch_size)
        x = start_token
        energies = torch.ones(batch_size, device=self.device) * energy
        energies = energies.unsqueeze(1)
        sequences = []
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()

        for step in range(max_length):
            logits, h = self.rnn(x, energies, h)
            prob = F.softmax(logits, dim=1)
            log_prob = F.log_softmax(logits, dim=1)
            x = torch.multinomial(prob,1).view(-1)
            sequences.append(x.view(-1, 1))
            log_probs +=  NLLLoss(log_prob, x)
            entropy += -torch.sum((log_prob * prob), 1)

            x = Variable(x.data)
            EOS_sampled = (x == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        sequences = torch.cat(sequences, 1)
        return sequences.data, log_probs, entropy

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
