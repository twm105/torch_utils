import torch
from torch import nn
import torch.nn.functional as F


# standard variables
EMBED_DIM = 256
CONTEXT_LENGTH = 512
MLP_HIDDEN_DIM = 4 * EMBED_DIM
MLP_DROPOUT = 0.1
N_ROUTED_EXPERTS = 8
N_SHARED_EXPERTS = 2
TOP_K = 2


# DeepSeekMoE MLP block
class ExpertsMLPBlock(nn.Module):
    """docstring"""

    def __init__(
        self,
        num_experts: int = N_ROUTED_EXPERTS,
        embed_dim: int = EMBED_DIM,
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
        mlp_dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()

        # MOE constants
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.hidden_dim = mlp_hidden_dim

        # MOE weights and operations
        self.rmsnorm = nn.RMSNorm(normalized_shape=embed_dim)
        self.W1 = nn.Parameter(data=torch.randn(num_experts, embed_dim, mlp_hidden_dim))
        self.b1 = nn.Parameter(data=torch.randn(num_experts, mlp_hidden_dim))
        self.W2 = nn.Parameter(data=torch.randn(num_experts, mlp_hidden_dim, embed_dim))
        self.b2 = nn.Parameter(data=torch.randn(num_experts, embed_dim))
        self.dropout = nn.Dropout(p=mlp_dropout)

    def forward(self, x, expert_ids=None) -> torch.Tensor:
        # perform MLP block operations, x assumed [B T C], expert_ids [B T K]
        B, T, C = x.size()
        W = self.hidden_dim

        # apply RMSNorm before MLP
        x = self.rmsnorm(x)

        # set weights and biases for MLP based on mode of operation
        W1 = self.W1  # [E C W]
        b1 = self.b1  # [E W]
        W2 = self.W2  # [E W C]
        b2 = self.b2  # [E C]

        # index weights and biases if expert ids passed for topk routed experts (i.e. E dims replaced with K dims)
        if expert_ids is not None:
            # index and expand experts over topk indices for [B T K], e.g. [E C H] -> [B T K C H]
            W1 = W1[expert_ids]
            b1 = b1[expert_ids]
            W2 = W2[expert_ids]
            b2 = b2[expert_ids]
            # TODO consider bucketing tokens per expert for performance (this mapping to indices allocates more memory)

            exp_dim = expert_ids.size()[2]
        else:
            # set experts dim and expand weights and biases to [1 1 E n n] (n dependent on layer)
            exp_dim = self.num_experts
            W1 = W1.unsqueeze(0).unsqueeze(0).expand([B, T, -1, -1, -1])
            b1 = b1.unsqueeze(0).unsqueeze(0).expand([B, T, -1, -1])
            W2 = W2.unsqueeze(0).unsqueeze(0).expand([B, T, -1, -1, -1])
            b2 = b2.unsqueeze(0).unsqueeze(0).expand([B, T, -1, -1])

        # expand x to match B, T of input and flatten all to B*T*E for matmuls
        x = x.unsqueeze(2).expand([-1, -1, exp_dim, -1])  # [B T E C]
        x = x.reshape(-1, C).unsqueeze(1)  # [B*T*E, 1, C]
        W1 = W1.reshape(-1, C, W)  # [B*T*E C W]
        b1 = b1.reshape(-1, 1, W)  # [B*T*E 1 W]
        W2 = W2.reshape(-1, W, C)  # [B*T*E W C]
        b2 = b2.reshape(-1, 1, C)  # [B*T*E 1 C]

        print(f"x: {x.size()}")
        print(f"W1: {W1.size()}, b1: {b1.size()}, W2: {W2.size()}, b2: {b2.size()}")
        print(f"x @ W1: {(x @ W1).size()}")

        # implement MLP (E is number of experts, which is K experts if Top-K indices passed to forward)
        x = F.gelu(
            x @ W1 + b1
        )  # [B*T*E 1 C] @ [B*T*E C H] + [B*T*E 1 H] -> [B*T*E 1 H]
        print(f"hidden layer: {x.size()}")
        print(f"x @ W2: {(x @ W2).size()}")
        x = x @ W2 + b2  # [B*T*E 1 C] @ [B*T*E C H] + [B*T*E 1 H] -> [B*T*E 1 H]
        print(f"layer 2: {x.size()}")
        h = self.dropout(x.squeeze(1)).reshape(B, T, exp_dim, C)  # [B T E C]
        print(f"MLP out: {h.size()}")
        return h


# MOE
class DeepSeekMoE(nn.Module):
    """docstring"""

    # TODO DeepSeekMoE docstring

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        n_tokens: int = CONTEXT_LENGTH,
        mlp_hidden_dim: int = MLP_HIDDEN_DIM,
        mlp_dropout: float = MLP_DROPOUT,
        n_shared_experts: int = N_SHARED_EXPERTS,
        n_routed_experts: int = N_ROUTED_EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.top_k = top_k

        self.shared_experts = ExpertsMLPBlock(
            num_experts=n_shared_experts,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
        )
        self.routed_experts = ExpertsMLPBlock(
            num_experts=n_routed_experts,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_dropout=mlp_dropout,
        )
        self.routed_centroids = nn.Parameter(
            data=torch.randn(embed_dim, n_routed_experts)
        )
        # TODO add bias term bi for each expert to balance routing (buffer?) dims: [1 1 E]

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # get dims from u
        B, T, C = u.size()
        print(f"u: {u.size()}")

        # shared experts
        print("SHARED EXPERTS")
        shared_experts = self.shared_experts(u)  # returns [B T E C]
        h_s = shared_experts.sum(-2)  # [B T C]

        # routed experts
        print("ROUTED EXPERTS")
        s = F.sigmoid(u @ self.routed_centroids)  # [B T C] @ [C E] -> [B T E]
        print(f"s: {s.size()}")

        # generate top-k affinities and associated indices
        gpk, i_topks = torch.topk(
            input=s,  # TODO add the bias term bi, which is updated during training by gamma
            k=self.top_k,
            dim=-1,
        )  # [B T K]
        print(f"gpk: {gpk.size()}")

        # normalised gated affinities
        gk = (
            (
                gpk
                / s.sum(
                    dim=-1,
                    keepdim=True,
                ).expand(-1, -1, self.top_k)
            )
            .unsqueeze(-1)
            .expand([-1, -1, -1, self.embed_dim])
        )  # gk = gpk / sum(s), summed for all expert centroids (summed dim expanded for Top-K), then expanded for C -> [B T K C]
        print(f"gk: {gk.size()}")

        # generate top-k routed experts
        routed_experts = self.routed_experts(u, expert_ids=i_topks)  # returns [B T K C]

        # routed output is normalised gated affinities * routed experts, summed along top-k dimension
        h_r = (gk * routed_experts).sum(-2)  # [B T C]
        print(f"h_r: {h_r.size()}")

        return u + h_s + h_r


if __name__ == "__main__":
    B, T, C = 1, 5, 4
    H = 16
    Ns, Nr, Kr = 3, 8, 2
    u = torch.randn([B, T, C])
    moe = DeepSeekMoE(
        embed_dim=C,
        mlp_hidden_dim=H,
        mlp_dropout=0.1,
        n_shared_experts=Ns,
        n_routed_experts=Nr,
        top_k=Kr,
    )

    h = moe(u)
    print(f"DeepSeekMoE Out: {h.size()}")
