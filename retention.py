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

         

    def _get_D(self, seq_len):
            pass