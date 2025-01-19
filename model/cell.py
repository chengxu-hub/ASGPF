"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import utils
import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                # (batch, num_nodes, input_dim+hidden_dim)
                x1 = torch.matmul(support, x0)
                # (batch, ?, num_nodes, input_dim+hidden_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # (batch, num_nodes, input_dim+hidden_dim)
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2)  # (batch, ?, num_nodes, input_dim+hidden_dim)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * \
            self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True):
        """
        Args:
            input_dim: input feature dim
            num_units: number of DCGRU hidden units
            max_diffusion_step: maximum diffusion step
            num_nodes: number of nodes in the graph
            filter_type: 'laplacian' for undirected graph, 'dual_random_walk' for directed graph
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
            use_gc_for_ru: decide whether to use graph convolution inside rnn. Default True
        """
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":  # ChebNet graph conv
            self._num_supports = 1
        elif filter_type == "random_walk":  # Forward random walk
            self._num_supports = 1
        elif filter_type == "dual_random_walk":  # Bidirectional random walk
            self._num_supports = 2
        else:
            self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(
                output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        # batch_size, self._num_nodes * output_size
        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A
        # A_out = A[:, :, self.n_node*self.n_edge_types:]
        A_out = A.transpose(1, 2)

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output

class GGNNBasedLayer(nn.Module):
    def __init__(self, annotation_dim, state_dim, n_node, n_edge_types, n_steps):
        super(GGNNBasedLayer, self).__init__()

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_node = n_node
        self.n_steps = n_steps

        # Incoming and outgoing edge embedding
        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propagation Model
        self.propagator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.output_layer = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.normal_(0.0, 0.02)
                m.bias.data = m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node * self.n_edge_types, self.state_dim)

            prop_state = self.propagator(in_states, out_states, prop_state, A)

        joined_state = torch.cat((prop_state, annotation), 2)
        output = self.output_layer(joined_state)
        output = output.view(-1, self.n_node * self.state_dim)
        hidden_state = output
        return output, hidden_state


class GGNNBasedRNNCell(nn.Module):
    def __init__(self, input_dim, num_units, num_nodes, num_edge_types, n_steps):
        super(GGNNBasedRNNCell, self).__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.ggnn = GGNNBasedLayer(input_dim, num_units, num_nodes, num_edge_types, n_steps)

    def forward(self, inputs, adjacency_matrix):
        batch_size, _ = inputs.shape
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))
        # Initialize the propagation state
        padding = torch.zeros(len(inputs), self.num_nodes, self.num_units - self.input_dim).float()
        prop_state = torch.cat((inputs, padding), 2)
        # prop_state = torch.zeros(batch_size, self.num_nodes, self.num_units).to(inputs.device)
        return self.ggnn(prop_state, inputs, adjacency_matrix)
