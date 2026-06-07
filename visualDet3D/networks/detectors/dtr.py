import torch.nn as nn
import torch
import torch
from torch.nn import Module, Dropout

class DepthAwareTransformer(nn.Module):
    def __init__(self, output_channel_num):
        super().__init__()
        self.output_channel_num = output_channel_num
        self.encoder = TransEncoderLayer(self.output_channel_num)
        self.decoder = TransDecoderLayer(self.output_channel_num)

    def forward(self, depth_feat, context_feat, depth_pos=None):
        
        # context_feat: N, L, C
        # depth_feat: N, L, C
        # depth_pos: N, L, C

        context_feat = context_feat + depth_pos
        context_feat = self.encoder(context_feat)
        integrated_feat = self.decoder(depth_feat, context_feat)
        return integrated_feat


class TransEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=2048,
                 nhead=8,
                 attention='linear'):
        super(TransEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = FAVORAttention(self.dim)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and drop_path
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = nn.Identity()

    def forward(self, x):
        
        bs = x.size(0)
        query, key, value = x, x, x

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm1(message))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, L, H, D]
            values: [N, L, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (L,D)' @ L,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


def orthogonal_random_features(num_rows, num_cols, device=None, dtype=None):
    """ Builds a random projection matrix with orthogonal rows and chi-distributed
    row norms ("Orthogonal Random Features"), the variance-reducing construction
    that FAVOR+ uses for its random feature maps. """
    num_full_blocks = num_rows // num_cols
    blocks = []
    for _ in range(num_full_blocks):
        unstructured = torch.randn(num_cols, num_cols, device=device, dtype=dtype)
        q, _ = torch.qr(unstructured)
        blocks.append(q.t())
    remainder = num_rows - num_full_blocks * num_cols
    if remainder > 0:
        unstructured = torch.randn(num_cols, num_cols, device=device, dtype=dtype)
        q, _ = torch.qr(unstructured)
        blocks.append(q.t()[:remainder])
    matrix = torch.cat(blocks, dim=0)

    # rescale rows to chi-distributed norms so each row keeps a N(0, I) marginal,
    # matching the rows of an unstructured iid Gaussian projection matrix
    row_norms = torch.randn(num_rows, num_cols, device=device, dtype=dtype).norm(dim=-1, keepdim=True)
    return row_norms * matrix


class FAVORAttention(Module):
    """ Multi-head linear attention via FAVOR+ positive orthogonal random features
    (Choromanski et al., "Rethinking Attention with Performers", ICLR 2021).

    Approximates softmax(QK^T / sqrt(D)) with an unbiased random-feature kernel
    phi(q).phi(k), reusing the same O(L) linear-attention computation as
    LinearAttention but grounding the feature map in the softmax kernel instead
    of the ad-hoc elu(x)+1 map. """

    def __init__(self, dim, num_features=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_features = num_features or dim
        self.eps = eps
        self.register_buffer('projection_matrix', orthogonal_random_features(self.num_features, dim))

    def _feature_map(self, x):
        # x: [N, L, H, D]
        x = x * (self.dim ** -0.25)  # so phi(x/d^.25).phi(y/d^.25) approximates exp(x.y / sqrt(d))
        projection = self.projection_matrix.to(device=x.device, dtype=x.dtype)

        x_proj = torch.einsum('nlhd,md->nlhm', x, projection)        # [N, L, H, M]
        x_norm_sq = 0.5 * (x ** 2).sum(dim=-1, keepdim=True)         # [N, L, H, 1]

        # clamp rather than max-subtract: a uniform rescaling of phi would have to
        # be exactly cancelled against the `+ eps` below, which it isn't in floating
        # point once values shrink by several orders of magnitude. The exponent here
        # realistically stays under ~12 for this network's feature scale (eps^88 is
        # float32's overflow point), so clamping only guards rare outliers.
        exponent = torch.clamp(x_proj - x_norm_sq, max=30.0)
        return torch.exp(exponent) / (self.num_features ** 0.5)

    def forward(self, queries, keys, values):
        """ Args:
            queries, keys, values: [N, L, H, D]
        Returns:
            queried_values: [N, L, H, D]
        """
        Q = self._feature_map(queries)
        K = self._feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshm,nshv->nhmv", K, values)
        Z = 1 / (torch.einsum("nlhm,nhm->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhm,nhmv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class TransDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead=8,
                 attention='linear'):
        super(TransDecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.attention0 = FAVORAttention(self.dim)
        self.merge0 = nn.Linear(d_model, d_model, bias=False)

        # multi-head attention
        self.q_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.attention1 = FAVORAttention(self.dim)
        self.merge1 = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = nn.Identity()

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, L, C]
        """
        
        bs = x.size(0)

        #Self-Attentiion for x (depth_feat)
        query, key, value = x, x, x

        query = self.q_proj0(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj0(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj0(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention0(query, key, value)  # [N, L, (H, D)]
        message = self.merge0(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm0(message))

        #Cross-Attentiion for x and source (depth_feat & context_feat)
        query, key, value = x, source, source

        query = self.q_proj1(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj1(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj1(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention1(query, key, value)  # [N, L, (H, D)]
        message = self.merge1(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm1(message))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        
        return x

