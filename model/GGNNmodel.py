from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.data_utils import computeFFT
from model.cell import GGNNBasedRNNCell
from torch.autograd import Variable
import utils
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)


class GGNNEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_nodes, num_edge_types, num_rnn_layers, n_steps, device=None):
        super(GGNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.num_nodes = num_nodes

        encoding_cells = list()
        # 第一层有不同的输入维度
        encoding_cells.append(
            GGNNBasedRNNCell(
                input_dim=input_dim,
                num_units=hid_dim,
                num_nodes=num_nodes,
                num_edge_types=num_edge_types,
                n_steps=n_steps))

        # 构建多层 RNN
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                GGNNBasedRNNCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    num_nodes=num_nodes,
                    num_edge_types=num_edge_types,
                    n_steps=n_steps))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, adjacency_matrix):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    current_inputs[t, ...], adjacency_matrix)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(torch.zeros(batch_size, self.hid_dim * self.num_nodes).to(self._device))
        return torch.stack(init_states, dim=0)


class GGNNDecoder(nn.Module):
    def __init__(self, input_dim, num_nodes, hid_dim, output_dim, num_edge_types, num_rnn_layers, n_steps, device=None, dropout=0.0):
        super(GGNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = nn.Dropout(p=dropout)

        decoding_cells = list()
        decoding_cells.append(
            GGNNBasedRNNCell(
                input_dim=input_dim,
                num_units=hid_dim,
                num_nodes=num_nodes,
                num_edge_types=num_edge_types,
                n_steps=n_steps))

        for _ in range(1, num_rnn_layers):
            decoding_cells.append(
                GGNNBasedRNNCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    num_nodes=num_nodes,
                    num_edge_types=num_edge_types,
                    n_steps=n_steps))
        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)

    def forward(self, inputs, initial_hidden_state, adjacency_matrix, teacher_forcing_ratio=None):
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim)).to(self._device)

        outputs = torch.zeros(seq_length, batch_size, self.num_nodes * self.output_dim).to(self._device)

        current_input = go_symbol
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](current_input, adjacency_matrix)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(self.dropout(output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs



########## Model for seizure classification/detection ##########
class GGNNModelClassification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(GGNNModelClassification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        num_edge_types = args.num_edge_types  # Add this parameter to args
        n_steps = args.n_steps  # Add this parameter to args

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = GGNNEncoder(input_dim=enc_input_dim,
                                   hid_dim=rnn_units,
                                   num_nodes=num_nodes,
                                   num_edge_types=num_edge_types,
                                   num_rnn_layers=num_rnn_layers,
                                   n_steps=n_steps,
                                   device=device)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, adjacency_matrix):
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        init_hidden_state = self.encoder.init_hidden(batch_size).to(self._device)

        _, final_hidden = self.encoder(input_seq, init_hidden_state, adjacency_matrix)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        last_out = utils.last_relevant_pytorch(output, seq_lengths, batch_first=True)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        last_out = last_out.to(self._device)

        logits = self.fc(self.relu(self.dropout(last_out)))

        pool_logits, _ = torch.max(logits, dim=1)

        return pool_logits
########## Model for seizure classification/detection ##########


########## Model for next time prediction ##########
class GGNNModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(GGNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        num_edge_types = args.num_edge_types  # Add this to args
        n_steps = args.n_steps  # Add this to args

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = GGNNEncoder(
            input_dim=enc_input_dim,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_edge_types=num_edge_types,
            num_rnn_layers=num_rnn_layers,
            n_steps=n_steps,
            device=device
        )

        self.decoder = GGNNDecoder(
            input_dim=dec_input_dim,
            num_nodes=num_nodes,
            hid_dim=rnn_units,
            output_dim=output_dim,
            num_edge_types=num_edge_types,
            num_rnn_layers=num_rnn_layers,
            n_steps=n_steps,
            device=device,
            dropout=args.dropout
        )

    def forward(self, encoder_inputs, decoder_inputs, adjacency_matrix, batches_seen=None):
        adjacency_matrix = torch.stack(adjacency_matrix)
        adjacency_matrix = torch.squeeze(adjacency_matrix, dim=0)
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        init_hidden_state = self.encoder.init_hidden(batch_size)

        encoder_hidden_state, _ = self.encoder(encoder_inputs, init_hidden_state, adjacency_matrix)

        if self.training and self.use_curriculum_learning and batches_seen is not None:
            teacher_forcing_ratio = utils.compute_sampling_threshold(self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None

        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            adjacency_matrix,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs

########## Model for next time prediction ##########
