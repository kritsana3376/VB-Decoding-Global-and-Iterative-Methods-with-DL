"""CNN and GNN fault locators, and the inference helpers that drive them.

A fault locator scores each symbol position with the probability that it is
still in error after the baseline E-hMP pass. Both architectures consume the
same features:

    vn_feat (n, r+2)  verified flag, the r bits of Y, variable-node degree
    cn_feat (m, 1)    whether each check node's value is non-zero

The architectures here must stay in step with the ones in the training
notebooks, since checkpoints are loaded by `state_dict`.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class ResidualBlock1D(nn.Module):
    """Two 1-D convolutions with a residual connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class FaultCNN(nn.Module):
    """CNN fault locator for any BCH(n, k) and symbol width r.

    ``vn_dim`` comes from the dataset and equals ``r + 2``. One global
    check-node feature is appended inside ``forward()``, so the convolution
    sees ``vn_dim + cn_dim`` input channels.
    """

    def __init__(self, vn_dim, cn_dim=1, hidden=64):
        super().__init__()
        self.vn_dim = vn_dim
        self.cn_dim = cn_dim
        in_channels = vn_dim + cn_dim

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
        )

        self.res1 = ResidualBlock1D(hidden)
        self.res2 = ResidualBlock1D(hidden)

        self.classifier = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, H, vn_feat, cn_feat):
        # H is unused by the CNN but kept in the signature so CNN and GNN are
        # interchangeable at the call site.
        n = vn_feat.shape[0]

        # Compress the check-node information into one global vector.
        cn_global = torch.mean(cn_feat, dim=0, keepdim=True)   # (1, cn_dim)
        cn_repeat = cn_global.repeat(n, 1)                     # (n, cn_dim)

        x = torch.cat([vn_feat, cn_repeat], dim=1)             # (n, vn_dim+cn_dim)
        x = x.transpose(0, 1).unsqueeze(0)                     # (1, channels, n)

        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)

        logits = self.classifier(x)                            # (1, 1, n)
        return logits.squeeze(0).transpose(0, 1)               # (n, 1)


class FaultGNN(nn.Module):
    """Message-passing fault locator on the Tanner graph defined by H.

    The variable-node count n and check-node count m are not fixed; both are
    taken from H in ``forward()``.
    """

    def __init__(self, vn_dim, cn_dim=1, hidden=64, num_iters=5):
        super().__init__()
        self.num_iters = num_iters

        self.vn_embed = nn.Linear(vn_dim, hidden)
        self.cn_update = nn.Sequential(nn.Linear(hidden + cn_dim, hidden), nn.ReLU())
        self.cn_to_vn = nn.Linear(hidden, hidden)
        self.vn_update = nn.Sequential(nn.Linear(hidden + hidden, hidden), nn.ReLU())
        self.output = nn.Linear(hidden, 1)

    def forward(self, H, vn_feat, cn_feat):
        # H: (m, n), vn_feat: (n, vn_dim), cn_feat: (m, cn_dim)
        if not torch.is_tensor(H):
            H = torch.tensor(H, dtype=torch.float32, device=vn_feat.device)
        else:
            H = H.to(device=vn_feat.device, dtype=torch.float32)

        vn_hidden = self.vn_embed(vn_feat)                     # (n, hidden)

        for _ in range(self.num_iters):
            cn_agg = torch.matmul(H, vn_hidden)                # (m, hidden)
            cn_input = torch.cat([cn_agg, cn_feat], dim=1)     # (m, hidden+cn_dim)
            cn_hidden = self.cn_update(cn_input)               # (m, hidden)

            cn_msg = self.cn_to_vn(cn_hidden)                  # (m, hidden)
            vn_agg = torch.matmul(H.T, cn_msg)                 # (n, hidden)

            vn_input = torch.cat([vn_agg, vn_hidden], dim=1)   # (n, 2*hidden)
            vn_hidden = self.vn_update(vn_input) + vn_hidden   # residual update

        return self.output(vn_hidden)                          # (n, 1)


def load_fault_model(model_path, r, model_type="gnn"):
    """Load a trained fault locator and put it in eval mode."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # vn_dim = verified flag (1) + Y_bin (r) + degree (1)
    vn_dim = r + 2

    model_type = model_type.lower()
    if model_type == "cnn":
        model = FaultCNN(vn_dim=vn_dim, cn_dim=1, hidden=64)
    elif model_type == "gnn":
        model = FaultGNN(vn_dim=vn_dim)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"{model_type.upper()} model loaded from {model_path} (vn_dim={vn_dim})")
    return model


# ---------------------------------------------------------------------------
# Features and inference
# ---------------------------------------------------------------------------

def build_gnn_features(H, Y_bin, verified_mask, bin_total_check_node):
    """Assemble the variable-node and check-node feature matrices."""
    syndrome_flag = (np.sum(bin_total_check_node, axis=1) > 0).astype(int)
    cn_feat = syndrome_flag.reshape(-1, 1).astype(np.float32)

    degree = np.sum(H, axis=0)
    vn_feat = np.concatenate(
        [verified_mask.reshape(-1, 1), Y_bin, degree.reshape(-1, 1)], axis=1
    ).astype(np.float32)

    return vn_feat, cn_feat


def _rank_unverified(prob, verified_mask):
    """Rank unverified positions by descending fault probability."""
    return sorted(
        [(i, prob[i]) for i in range(len(prob)) if verified_mask[i] == 0],
        key=lambda x: x[1],
        reverse=True,
    )


def predict_fault_positions_model(model, H, Y_bin, verified_mask, bin_total_check_node):
    """Score fault likelihood with a PyTorch CNN or GNN."""
    device = next(model.parameters()).device

    vn_feat, cn_feat = build_gnn_features(H, Y_bin, verified_mask, bin_total_check_node)
    vn_feat = torch.tensor(vn_feat).to(device)
    cn_feat = torch.tensor(cn_feat).to(device)

    with torch.no_grad():
        logits = model(H, vn_feat, cn_feat)
        prob = torch.sigmoid(logits).cpu().numpy().flatten()

    return _rank_unverified(prob, verified_mask), prob


def predict_fault_positions_onnx(session, H, Y_bin, verified_mask, bin_total_check_node):
    """Score fault likelihood with an exported ONNX model.

    ``session`` is an ``onnxruntime.InferenceSession``; onnxruntime is only
    needed by whoever builds it, not by this module.
    """
    vn_feat, cn_feat = build_gnn_features(H, Y_bin, verified_mask, bin_total_check_node)

    inputs = {
        "vn_feat": vn_feat.astype(np.float32),
        "cn_feat": cn_feat.astype(np.float32),
    }
    # Only pass H if the exported graph actually declares it.
    if "H" in [i.name for i in session.get_inputs()]:
        inputs["H"] = H.astype(np.float32)

    logits = session.run(None, inputs)[0]
    prob = 1.0 / (1.0 + np.exp(-logits.flatten()))

    return _rank_unverified(prob, verified_mask), prob


def predict_fault_positions_rand_gauss(verified_mask, mu=0.5, sigma=0.1):
    """Random Gaussian baseline for error-location prediction.

    Only unverified symbols receive a random score; verified symbols are
    forced to 0 so they are never selected. Output matches the CNN/GNN
    predictors.
    """
    verified_mask = np.asarray(verified_mask).astype(np.int8).flatten()
    n = len(verified_mask)

    prob = np.zeros(n, dtype=np.float32)
    unverified_idx = np.where(verified_mask == 0)[0]

    if len(unverified_idx) > 0:
        rand_score = np.random.normal(loc=mu, scale=sigma, size=len(unverified_idx))
        prob[unverified_idx] = np.clip(rand_score, 0.0, 1.0).astype(np.float32)

    ranked_faults = sorted(
        [(int(i), float(prob[i])) for i in unverified_idx],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked_faults, prob


def predict_fault_positions_any(model, H, Y_bin, verified_mask, bin_total_check_node,
                                use_onnx=False, use_rand_gauss=False):
    """Unified predictor for CNN, GNN, ONNX, and the random Gaussian baseline."""
    if use_rand_gauss or str(model).lower() in ("rand_gauss", "random_gauss", "random_gaussian"):
        return predict_fault_positions_rand_gauss(verified_mask)

    if use_onnx:
        return predict_fault_positions_onnx(
            model, H, Y_bin, verified_mask, bin_total_check_node
        )
    return predict_fault_positions_model(
        model, H, Y_bin, verified_mask, bin_total_check_node
    )
