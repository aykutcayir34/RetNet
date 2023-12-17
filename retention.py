import math
import torch
import torch.nn as nn
from xpos import XPOS

class Retention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        super(Retention, self).__init__()

        self.hidden_size = hidden_size
        self.gamma = gamma
        self.head_size = head_size
        self.v_dim = 2 * head_size if double_v_dim else head_size

        self.W_Q = nn.Parameter(torch.randn(self.hidden_size, self.head_size) / self.hidden_size)
        self.W_K = nn.Parameter(torch.randn(self.hidden_size, self.head_size) / self.hidden_size)
        self.W_V = nn.Parameter(torch.randn(self.hidden_size, self.v_dim) / self.hidden_size)

        self.xpos = XPOS(self.head_size)

    def forward(self, X):
        seq_len = X.size[1] # x.size = (batch_size, seq_len, hidden_size)
        D = self._get_D(seq_len).to(self.W_Q.device)
        Q = X @ self.W_Q # Q.size = (batch_size, seq_len, head_size)
        K = X @ self.W_K # K.size = (batch_size, seq_len, head_size)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)
        V = X @ self.W_V

        retention = (Q @ K.transpose(1, 2)) * D.unsqueeze(0)
        return retention @ V   

    def forward_recurrent(self, x_n, s_n_1, n):
        #x_n.size = (batch_size, 1, hidden_size)
        #s_n_1.size = (batch_size, hidden_size, v_dim)
        Q = x_n @ self.W_Q
        K = x_n @ self.W_K

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V
        s_n = self.gamma * s_n_1 + (Q @ K.transpose(1, 2)) @ V
        return Q @ s_n, s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        #x_i.size = (batch_size, chunk_size, hidden_size)
        #r_i_1.size = (batch_size, hidden_size, v_dim)
        batch_size, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size).to(self.W_Q.device)
        Q = x_i @ self.W_Q
        K = x_i @ self.W_K
        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma**chunk_size) * r_i_1
        inner_chunk = ((Q @ K.transpose(-1, -2))) * D.unsqueeze(0) @ V
        e = torch.zeros(batch_size, chunk_size, 1)
        for j in range(chunk_size):
            e[:, j, :] = (self.gamma**(j + 1))

        cross_chunk = (Q @ r_i_1) * e
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, seq_len):
        n = torch.arange(seq_len).float()
        m = torch.arange(seq_len).float()

        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0
        return D
    
class MultiscaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim) -> None:
        super(MultiscaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert self.hidden_size % self.heads == 0, "hidden_size must be divisible by heads"
        self.head_size = self.hidden_size // self.heads
        self.v_dim = 2 * self.hidden_size if double_v_dim else self.hidden_size
        self.head_v_dim = 2 * self.hidden_size if double_v_dim else self.hidden_size
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()
        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(self.hidden_size, self.v_dim) / self.hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, self.hidden_size) / self.hidden_size)
        self.group_norm = nn.GroupNorm(self.heads, self.v_dim)

        self.retentions = nn.ModuleList([Retention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas])

    def forward(self, X):
        Y = []
        for _, retention in enumerate(self.retentions):
            Y.append(retention(X))
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.view(-1, self.v_dim)).view(Y_shape)
        return (self.swish(X @ self.W_G) @ Y) @ self.W_O
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        #x_n.size = (batch_size, 1, hidden_size)
        #s_n_1.size = (batch_size, hidden_size, v_dim)
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(x_n[:, :, :], s_n_1s[i], n)
            Y.append(y)
            s_ns.append(s_n)
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        batch_size, chunk_size, _ = x_i.shape
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(x_i[:, :, :], r_i_1s[j], i)
            Y.append(y)
            r_is.append(r_i)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is
