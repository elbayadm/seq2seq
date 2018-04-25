class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, context_size):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = 1

        self.input_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size)
        )
        self.hidden_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.input_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, context_size)
        )
        self.hidden_weights_2 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_2 = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.context2attention = nn.Parameter(
            torch.Tensor(context_size, context_size)
        )
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(
            torch.Tensor(context_size, hidden_size)
        )

        self.input2attention = nn.Parameter(
            torch.Tensor(input_size, context_size)
        )

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)

        self.input_weights_2.data.uniform_(-stdv_ctx, stdv_ctx)
        self.hidden_weights_2.data.uniform_(-stdv, stdv)
        self.input_bias_2.data.fill_(0)
        self.hidden_bias_2.data.fill_(0)

        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden, projected_input, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(
                input, self.input_weights_1, self.input_bias_1
            ) + F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )

            # Add this to projected input at this time step
            hidden_ctx_sum = hidden_ctx_sum + \
                projected_input.unsqueeze(0).expand(hidden_ctx_sum.size())

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(
                hidden_ctx_sum.transpose(0, 1),
                self.recurrent2attention.unsqueeze(0).expand(
                    hidden_ctx_sum.size(1),
                    self.recurrent2attention.size(0),
                    self.recurrent2attention.size(1)
                )
            ).squeeze()
            alpha = F.softmax(alpha, dim=1)
            weighted_context = torch.mul(
                ctx, alpha.t().unsqueeze(2).expand(ctx.size())
            ).sum(0).squeeze()

            gates = F.linear(
                weighted_context, self.input_weights_2, self.input_bias_2
            ) + F.linear(hy, self.hidden_weights_2, self.hidden_bias_2)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cy) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        input = input.transpose(0, 1)
        projected_ctx = torch.bmm(
            ctx,
            self.context2attention.unsqueeze(0).expand(
                ctx.size(0),
                self.context2attention.size(0),
                self.context2attention.size(1)
            ),
        )
        projected_ctx += \
            self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(
                projected_ctx.size()
            )

        projected_input = torch.bmm(
            input,
            self.input2attention.unsqueeze(0).expand(
                input.size(0),
                self.input2attention.size(0),
                self.input2attention.size(1)
            ),
        )

        output = []
        steps = list(range(input.size(0)))
        for i in steps:
            hidden = recurrence(
                input[i], hidden, projected_input[i], projected_ctx
            )
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class StackedAttentionLSTM(nn.Module):
    """Deep Attention LSTM."""

    def __init__(
        self,
        input_size,
        rnn_size,
        num_layers,
        batch_first=True,
        dropout=0.
    ):
        """Initialize params."""
        super(StackedAttentionLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_first = batch_first

        self.layers = []
        for i in range(num_layers):
            layer = LSTMAttentionDot(
                input_size, rnn_size, batch_first=self.batch_first
            )
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the layer."""
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            if ctx_mask is not None:
                ctx_mask = torch.ByteTensor(
                    ctx_mask.data.cpu().numpy().astype(np.int32).tolist()
                ).cuda()
            output, (h_1_i, c_1_i) = layer(input, (h_0, c_0), ctx, ctx_mask)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class DeepBidirectionalLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""

    def __init__(
        self, input_size, hidden_size,
        num_layers, dropout, batch_first
    ):
        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.bi_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout
        )

    def get_init_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))

        h0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        return (h0_encoder_bi.cuda(), c0_encoder_bi.cuda()), \
            (h0_encoder.cuda(), c0_encoder.cuda())

    def forward(self, input):
        """Propogate input forward through the network."""
        hidden_bi, hidden_deep = self.get_init_state(input)
        bilstm_output, (_, _) = self.bi_encoder(input, hidden_bi)
        return self.encoder(bilstm_output, hidden_deep)



