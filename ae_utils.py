"""
ae_utils.py  --  Shared AE utilities for trajectory noise-robustness experiments.

Extracted from zakData.ipynb. Import this module in any dataset-specific notebook:
    from ae_utils import (TrajectoryAutoencoder, TrajectoryLSTMAE, TrajectoryGRUAE,
                          train_fc_ae, train_lstm_ae, train_gru_ae,
                          compare_noise_sweep,
                          compare_noise_sweep_random,
                          compare_noise_sweep_separate_best)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib


# ===========================================================================
# 1. Model classes
# ===========================================================================

class TrajectoryAutoencoder(nn.Module):
    """
    FC Autoencoder for fixed-length trajectories.
    Input / output: flat vector (batch, 2T) in scaler space.
    Architecture: 2T -> 128 -> 64 -> latent_dim -> 64 -> 128 -> 2T
    """
    def __init__(self, input_dim=200, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class TrajectoryLSTMAE(nn.Module):
    """
    LSTM Autoencoder for fixed-length trajectories.
    Input / output: flat vector (batch, 2T) in scaler space.

    Encoder: (batch, 2T) -> reshape (batch, T, 2) -> LSTM -> last hidden -> FC -> latent
    Decoder: latent -> FC -> repeat T -> LSTM -> pointwise FC(2) -> (batch, 2T)
    """
    def __init__(self, T=100, n_features=2, latent_dim=32,
                 enc_hidden=128, dec_hidden=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.T          = T
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_layers   = n_layers

        _drop = dropout if n_layers > 1 else 0.0

        self.enc_lstm = nn.LSTM(n_features, enc_hidden, n_layers,
                                batch_first=True, dropout=_drop)
        self.enc_fc   = nn.Linear(enc_hidden, latent_dim)
        self.dec_fc   = nn.Linear(latent_dim, dec_hidden)
        self.dec_lstm = nn.LSTM(dec_hidden, dec_hidden, n_layers,
                                batch_first=True, dropout=_drop)
        self.out_fc   = nn.Linear(dec_hidden, n_features)

    def encode(self, x_flat):
        b   = x_flat.shape[0]
        seq = x_flat.view(b, self.T, self.n_features)
        _, (h_n, _) = self.enc_lstm(seq)
        return self.enc_fc(h_n[-1])

    def decode(self, z):
        b    = z.shape[0]
        seed = self.dec_fc(z).unsqueeze(1).expand(-1, self.T, -1)
        out, _ = self.dec_lstm(seed)
        return self.out_fc(out).reshape(b, self.T * self.n_features)

    def forward(self, x_flat):
        return self.decode(self.encode(x_flat))


class TrajectoryGRUAE(nn.Module):
    """
    GRU Autoencoder for fixed-length trajectories.
    Input / output: flat vector (batch, 2T) in scaler space.

    Encoder: (batch, 2T) -> reshape (batch, T, 2) -> GRU -> last hidden -> FC -> latent
    Decoder: latent -> FC -> repeat T -> GRU -> pointwise FC(2) -> (batch, 2T)
    """
    def __init__(self, T=100, n_features=2, latent_dim=32,
                 enc_hidden=128, dec_hidden=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.T          = T
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_layers   = n_layers

        _drop = dropout if n_layers > 1 else 0.0

        self.enc_gru = nn.GRU(n_features, enc_hidden, n_layers,
                               batch_first=True, dropout=_drop)
        self.enc_fc  = nn.Linear(enc_hidden, latent_dim)
        self.dec_fc  = nn.Linear(latent_dim, dec_hidden)
        self.dec_gru = nn.GRU(dec_hidden, dec_hidden, n_layers,
                               batch_first=True, dropout=_drop)
        self.out_fc  = nn.Linear(dec_hidden, n_features)

    def encode(self, x_flat):
        b   = x_flat.shape[0]
        seq = x_flat.view(b, self.T, self.n_features)
        _, h_n = self.enc_gru(seq)          # GRU: no cell state
        return self.enc_fc(h_n[-1])

    def decode(self, z):
        b    = z.shape[0]
        seed = self.dec_fc(z).unsqueeze(1).expand(-1, self.T, -1)
        out, _ = self.dec_gru(seed)
        return self.out_fc(out).reshape(b, self.T * self.n_features)

    def forward(self, x_flat):
        return self.decode(self.encode(x_flat))


# ===========================================================================
# 2. Training functions
# ===========================================================================

def train_fc_ae(
    resampled_csv = "resampled.csv",
    scaler_path   = "scaler.joblib",
    model_out     = "fc_ae.pth",
    T             = 100,
    latent_dim    = 32,
    lr            = 1e-3,
    epochs        = 200,
    batch_size    = 64,
    print_every   = 20,
):
    """Train FC-AE and save state_dict to model_out."""
    df = _load_and_pivot(resampled_csv, T)
    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(df).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_fc_ae] {df.shape[0]:,} flights  T={T}  latent={latent_dim}  device={device}")

    model  = TrajectoryAutoencoder(input_dim=2*T, latent_dim=latent_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled)),
                        batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        ep_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward(); opt.step()
            ep_losses.append(loss.item())
        if ep % print_every == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  loss={np.mean(ep_losses):.5f}")

    torch.save(model.state_dict(), model_out)
    print(f"[train_fc_ae] Saved -> {model_out}")
    return model


def train_lstm_ae(
    resampled_csv = "resampled.csv",
    scaler_path   = "scaler.joblib",
    model_out     = "lstm_ae.pth",
    T             = 100,
    latent_dim    = 32,
    enc_hidden    = 128,
    dec_hidden    = 128,
    n_layers      = 2,
    dropout       = 0.1,
    lr            = 1e-3,
    epochs        = 200,
    batch_size    = 64,
    print_every   = 20,
):
    """Train LSTM-AE and save state_dict + hyperparams to model_out."""
    df = _load_and_pivot(resampled_csv, T)
    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(df).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_lstm_ae] {df.shape[0]:,} flights  T={T}  latent={latent_dim}  device={device}")

    model  = TrajectoryLSTMAE(T=T, latent_dim=latent_dim,
                               enc_hidden=enc_hidden, dec_hidden=dec_hidden,
                               n_layers=n_layers, dropout=dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled)),
                        batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        ep_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward(); opt.step()
            ep_losses.append(loss.item())
        if ep % print_every == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  loss={np.mean(ep_losses):.5f}")

    torch.save({"state_dict": model.state_dict(),
                "T": T, "latent_dim": latent_dim,
                "enc_hidden": enc_hidden, "dec_hidden": dec_hidden,
                "n_layers": n_layers},
               model_out)
    print(f"[train_lstm_ae] Saved -> {model_out}")
    return model


def train_gru_ae(
    resampled_csv = "resampled.csv",
    scaler_path   = "scaler.joblib",
    model_out     = "gru_ae.pth",
    T             = 100,
    latent_dim    = 32,
    enc_hidden    = 128,
    dec_hidden    = 128,
    n_layers      = 2,
    dropout       = 0.1,
    lr            = 1e-3,
    epochs        = 200,
    batch_size    = 64,
    print_every   = 20,
):
    """Train GRU-AE and save state_dict + hyperparams to model_out."""
    df = _load_and_pivot(resampled_csv, T)
    scaler   = joblib.load(scaler_path)
    X_scaled = scaler.transform(df).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_gru_ae] {df.shape[0]:,} flights  T={T}  latent={latent_dim}  device={device}")

    model  = TrajectoryGRUAE(T=T, latent_dim=latent_dim,
                              enc_hidden=enc_hidden, dec_hidden=dec_hidden,
                              n_layers=n_layers, dropout=dropout).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_scaled)),
                        batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        ep_losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward(); opt.step()
            ep_losses.append(loss.item())
        if ep % print_every == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  loss={np.mean(ep_losses):.5f}")

    torch.save({"state_dict": model.state_dict(),
                "T": T, "latent_dim": latent_dim,
                "enc_hidden": enc_hidden, "dec_hidden": dec_hidden,
                "n_layers": n_layers},
               model_out)
    print(f"[train_gru_ae] Saved -> {model_out}")
    return model


# ===========================================================================
# 3. Internal helpers
# ===========================================================================

def _load_and_pivot(resampled_csv, T):
    """Load resampled CSV and return (N, 2T) numpy array."""
    df = pd.read_csv(resampled_csv, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
    for c in ["flight_id", "flightid", "fid", "icao", "trajectory_id"]:
        if c in df.columns: df = df.rename(columns={c: "flight_id"}); break
    for c in ["x", "lng", "lon", "longitude"]:
        if c in df.columns: df = df.rename(columns={c: "x"}); break
    for c in ["y", "lat", "latitude"]:
        if c in df.columns: df = df.rename(columns={c: "y"}); break
    df["_rank"] = df.groupby("flight_id").cumcount()
    xw = df.pivot(index="flight_id", columns="_rank", values="x")
    yw = df.pivot(index="flight_id", columns="_rank", values="y")
    ok = xw.notna().all(axis=1) & yw.notna().all(axis=1)
    return np.concatenate([xw[ok].to_numpy(np.float32),
                            yw[ok].to_numpy(np.float32)], axis=1)


def _cmp_load_csv(path):
    assert os.path.exists(path), f"Missing: {path}"
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True).str.lower()
    for c in ["flight_id", "flightid", "fid", "icao", "trajectory_id"]:
        if c in df.columns: df = df.rename(columns={c: "flight_id"}); break
    for c in ["x", "lng", "lon", "longitude"]:
        if c in df.columns: df = df.rename(columns={c: "x"}); break
    for c in ["y", "lat", "latitude"]:
        if c in df.columns: df = df.rename(columns={c: "y"}); break
    return df.dropna(subset=["x", "y"]).sort_values("flight_id")


def _cmp_build_X(df, ids, T):
    df = df[df["flight_id"].astype(str).isin(ids)].copy()
    df["_r"] = df.groupby("flight_id").cumcount()
    cnt  = df.groupby("flight_id")["_r"].max().add(1)
    ok   = cnt.index[cnt.eq(T)].astype(str)
    df   = df[df["flight_id"].astype(str).isin(ok)].copy()
    df["_ord"] = pd.Categorical(df["flight_id"].astype(str), categories=ids, ordered=True)
    df   = df.sort_values(["_ord", "_r"])
    xw   = df.pivot(index="flight_id", columns="_r", values="x").reindex(ids).dropna()
    yw   = df.pivot(index="flight_id", columns="_r", values="y").reindex(ids).dropna()
    kept = xw.index.astype(str).to_numpy()
    X    = np.concatenate([xw.to_numpy(np.float32), yw.to_numpy(np.float32)], axis=1)
    return X, kept


class _FCAE(nn.Module):
    """Minimal FC-AE used only by _load_fc_ae() for inference."""
    def __init__(self, input_dim=200, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),        nn.ReLU(),
            nn.Linear(128, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))


def _load_fc_ae(model_path, device):
    sd     = torch.load(model_path, map_location=device)
    in_dim = next(v for k, v in sd.items()
                  if "encoder.0.weight" in k).shape[1]
    latent = next(v for k, v in sd.items()
                  if "encoder.4.weight" in k).shape[0]
    m = _FCAE(in_dim, latent).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, in_dim // 2   # (model, T)


def _load_lstm_ae(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    m = TrajectoryLSTMAE(T=ckpt["T"], latent_dim=ckpt["latent_dim"],
                          enc_hidden=ckpt["enc_hidden"],
                          dec_hidden=ckpt["dec_hidden"],
                          n_layers=ckpt["n_layers"]).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return m, ckpt["T"]   # (model, T)


def _load_gru_ae(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    m = TrajectoryGRUAE(T=ckpt["T"], latent_dim=ckpt["latent_dim"],
                         enc_hidden=ckpt["enc_hidden"],
                         dec_hidden=ckpt["dec_hidden"],
                         n_layers=ckpt["n_layers"]).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return m, ckpt["T"]   # (model, T)


@torch.no_grad()
def _reconstruct(X, scaler, model, device):
    """Original-space input → original-space output."""
    Xs  = scaler.transform(X).astype(np.float32)
    Xt  = torch.from_numpy(Xs).to(device)
    B   = 4096 if device.type == "cuda" else 1024
    out = torch.cat([model(Xt[i:i+B]) for i in range(0, len(Xt), B)], 0)
    return scaler.inverse_transform(out.float().cpu().numpy()).astype(np.float32)


@torch.no_grad()
def _reconstruct_scaled(Xs, scaler, model, device):
    """Already-normalised input ([0,1] space) → original-space output."""
    Xt  = torch.from_numpy(Xs.astype(np.float32)).to(device)
    B   = 4096 if device.type == "cuda" else 1024
    out = torch.cat([model(Xt[i:i+B]) for i in range(0, len(Xt), B)], 0)
    return scaler.inverse_transform(out.float().cpu().numpy()).astype(np.float32)


def _rmse_vs_clean(recon_noisy, recon_clean, T):
    dx = recon_noisy[:, :T] - recon_clean[:, :T]
    dy = recon_noisy[:, T:] - recon_clean[:, T:]
    return np.sqrt((dx**2 + dy**2).mean(axis=1))   # (n,)


# ===========================================================================
# 4. Noise applicators
# ===========================================================================

def _noise_gaussian(X, T, std, seed):
    rng = np.random.default_rng(seed)
    n   = np.zeros_like(X)
    n[:, :T] = rng.normal(0, std, (X.shape[0], T))
    n[:, T:] = rng.normal(0, std, (X.shape[0], T))
    return np.clip(X + n.astype(np.float32), 0, 1)


def _noise_drift(X, T, mag, seed):
    rng = np.random.default_rng(seed)
    n   = X.copy()
    d   = rng.normal(0, 1, (X.shape[0], 2)).astype(np.float32)
    d  /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    n[:, :T] += d[:, 0:1] * mag
    n[:, T:] += d[:, 1:2] * mag
    return np.clip(n, 0, 1)


def _noise_spike(X, T, prob, seed, scale=4.0, base_std=5e-4):
    rng   = np.random.default_rng(seed)
    n     = X.copy()
    sp    = rng.random((X.shape[0], T)) < prob
    bump  = rng.normal(0, scale * base_std, (X.shape[0], T)).astype(np.float32)
    n[:, :T] += sp * bump
    sp2   = rng.random((X.shape[0], T)) < prob
    bump2 = rng.normal(0, scale * base_std, (X.shape[0], T)).astype(np.float32)
    n[:, T:] += sp2 * bump2
    return np.clip(n, 0, 1)


def _noise_missing(X, T, rate, seed, method="linear"):
    rng   = np.random.default_rng(seed)
    k     = max(0, int(np.floor(rate * T)))
    Xo    = X.copy()
    t_all = np.arange(T, dtype=np.float64)
    if k == 0:
        return Xo
    for i in range(X.shape[0]):
        mx = rng.choice(T, k, replace=False)
        ok = np.ones(T, bool); ok[mx] = False
        tv = t_all[ok]
        if len(tv) < 2:
            continue
        xv  = Xo[i, :T][ok].astype(np.float64)
        yv  = Xo[i, T:][ok].astype(np.float64)
        knd = method if (method != "cubic" or len(tv) >= 4) else "linear"
        fx  = interp1d(tv, xv, kind=knd, bounds_error=False, fill_value=(xv[0], xv[-1]))
        fy  = interp1d(tv, yv, kind=knd, bounds_error=False, fill_value=(yv[0], yv[-1]))
        Xo[i, :T] = np.clip(fx(t_all), 0, 1).astype(np.float32)
        Xo[i, T:] = np.clip(fy(t_all), 0, 1).astype(np.float32)
    return Xo


def _add_noise_mixed(X, T, noise_std, drift_mag, spike_prob, spike_scale, seed):
    """Gaussian + Drift + Spike applied simultaneously (zakData-style mixed noise).

    Spike amplitude = spike_scale × noise_std, so spike co-varies with the
    Gaussian sweep dimension — identical to zakData's add_noise_normalized.
    """
    rng = np.random.default_rng(seed)
    n   = X.copy()
    # Gaussian jitter
    n[:, :T] += rng.normal(0, noise_std, (X.shape[0], T)).astype(np.float32)
    n[:, T:]  += rng.normal(0, noise_std, (X.shape[0], T)).astype(np.float32)
    # Constant drift (random direction, uniform across time)
    if drift_mag > 0:
        d  = rng.normal(0, 1, (X.shape[0], 2)).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        n[:, :T] += d[:, 0:1] * drift_mag
        n[:, T:]  += d[:, 1:2] * drift_mag
    # Sparse spikes — amplitude = spike_scale × noise_std
    if spike_prob > 0:
        amp = spike_scale * max(noise_std, 1e-9)
        sp  = rng.random((X.shape[0], T)) < spike_prob
        n[:, :T] += sp  * rng.normal(0, amp, (X.shape[0], T)).astype(np.float32)
        sp2 = rng.random((X.shape[0], T)) < spike_prob
        n[:, T:]  += sp2 * rng.normal(0, amp, (X.shape[0], T)).astype(np.float32)
    return np.clip(n, 0, 1)


# ===========================================================================
# 5. Noise sweep -- shared internals
# ===========================================================================

def _build_noise_specs(T, spike_scale, spike_probs, noise_stds,
                        drift_mags, missing_rates, missing_method):
    return [
        ("Gaussian",
         "noise_std (normalised)",
         noise_stds,
         lambda X, p, s: _noise_gaussian(X, T, p, s)),
        ("Drift",
         "drift_mag (normalised)",
         drift_mags,
         lambda X, p, s: _noise_drift(X, T, p, s)),
        ("Spike",
         f"spike_prob (scale={spike_scale})",
         spike_probs,
         lambda X, p, s: _noise_spike(X, T, p, s, spike_scale)),
        ("Missing data\n(linear interp.)",
         "missing_rate",
         missing_rates,
         lambda X, p, s: _noise_missing(X, T, p, s, missing_method)),
    ]


def _build_noise_specs_mixed(T, spike_scale, spike_probs, noise_stds, drift_mags,
                              base_noise_std, base_drift_mag, base_spike_prob):
    """One-at-a-time sweep with mixed background noise (matches zakData design).

    3 noise types (no Missing). During each sweep dimension the other two types
    are held at their base values. Spike amplitude = spike_scale × noise_std,
    so it co-varies with the Gaussian sweep exactly as in zakData.
    """
    _bn, _bd, _bp, _sc = base_noise_std, base_drift_mag, base_spike_prob, spike_scale
    return [
        ("Gaussian",
         "noise_std (normalised)",
         noise_stds,
         lambda X, p, s: _add_noise_mixed(X, T, p,   _bd, _bp, _sc, s)),
        ("Drift",
         "drift_mag (normalised)",
         drift_mags,
         lambda X, p, s: _add_noise_mixed(X, T, _bn, p,   _bp, _sc, s)),
        ("Spike",
         f"spike_prob (scale={spike_scale}, amp=scale×noise_std)",
         spike_probs,
         lambda X, p, s: _add_noise_mixed(X, T, _bn, _bd, p,   _sc, s)),
    ]


def _build_noise_specs_4type_mixed(T, spike_scale, spike_probs, noise_stds, drift_mags,
                                    missing_rates, missing_method,
                                    base_noise_std, base_drift_mag, base_spike_prob):
    """4-type one-at-a-time sweep: Gaussian/Drift/Spike use mixed background noise
    (zakData-style, spike amp = spike_scale × noise_std); Missing remains isolated.
    """
    _bn, _bd, _bp, _sc = base_noise_std, base_drift_mag, base_spike_prob, spike_scale
    return [
        ("Gaussian",
         "noise_std (normalised)",
         noise_stds,
         lambda X, p, s: _add_noise_mixed(X, T, p,   _bd, _bp, _sc, s)),
        ("Drift",
         "drift_mag (normalised)",
         drift_mags,
         lambda X, p, s: _add_noise_mixed(X, T, _bn, p,   _bp, _sc, s)),
        ("Spike",
         f"spike_prob (scale={spike_scale}, amp=scale×noise_std)",
         spike_probs,
         lambda X, p, s: _add_noise_mixed(X, T, _bn, _bd, p,   _sc, s)),
        ("Missing data\n(linear interp.)",
         "missing_rate",
         missing_rates,
         lambda X, p, s: _noise_missing(X, T, p, s, missing_method)),
    ]


def _run_sweep(X_test, clean_base, scaler, model, device,
               noise_specs, n_runs, seed_offset):
    """Return dict noise_name -> [mean_rmse per param value].

    Noise is applied in [0,1] space (where X_test lives and where noise_std /
    drift_mag parameters are defined).  After noise, scaler.transform converts
    to model-input space.  RMSE is computed in original geographic space against
    clean_base (which must also be in original space).
    """
    results = {ns[0]: [] for ns in noise_specs}
    T = X_test.shape[1] // 2
    for ni, (nname, _, params, apply_fn) in enumerate(noise_specs):
        for pi, pval in enumerate(params):
            rmses = []
            for run in range(n_runs):
                seed   = seed_offset + (ni+1)*10000 + pi*n_runs + run
                X_noi  = apply_fn(X_test, pval, seed)               # noise in [0,1] space
                Xs_noi = scaler.transform(X_noi).astype(np.float32) # then scale to model space
                recon  = _reconstruct_scaled(Xs_noi, scaler, model, device)  # → orig space
                rmses.extend(_rmse_vs_clean(recon, clean_base, T).tolist())
            results[nname].append(rmses)           # full distribution per param value
        print(f"    {100*(ni+1)/len(noise_specs):5.1f}%  [{nname.split(chr(10))[0]}] done")
    return results


# ===========================================================================
# 6. Public sweep functions
# ===========================================================================

def compare_noise_sweep(
    resampled_csv   = "resampled.csv",
    best_list_csv   = "best100.csv",
    scaler_path     = "scaler.joblib",
    fc_model_path   = "fc_ae.pth",
    lstm_model_path = "lstm_ae.pth",
    N               = 100,
    n_runs          = 5,
    noise_stds      = None,
    drift_mags      = None,
    spike_probs     = None,
    spike_scale     = 4.0,
    missing_rates   = None,
    missing_method  = "linear",
    save_plot       = True,
    plot_path       = "noise_comparison.png",
):
    """FC-AE vs LSTM-AE on FC-AE's best-N trajectories (original experiment)."""
    if noise_stds    is None: noise_stds    = np.linspace(0, 0.005, 11)
    if drift_mags    is None: drift_mags    = np.linspace(0, 0.005, 11)
    if spike_probs   is None: spike_probs   = np.linspace(0, 0.30,  11)
    if missing_rates is None: missing_rates = np.linspace(0, 0.50,  11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    fc_model,  T  = _load_fc_ae  (fc_model_path,   device)
    lstm_model, _ = _load_lstm_ae(lstm_model_path, device)

    df = _cmp_load_csv(resampled_csv)
    ids_req = pd.read_csv(best_list_csv)
    id_col  = next(c for c in ["flight_id", "icao", "fid", "id"] if c in ids_req.columns)
    ids_req = ids_req[id_col].astype(str).head(N).to_numpy()
    X_best, _ = _cmp_build_X(df, ids_req, T)
    n = X_best.shape[0]
    print(f"Trajectories: {n}  T={T}  device={device}")

    fc_clean   = _reconstruct(X_best, scaler, fc_model,   device)
    lstm_clean = _reconstruct(X_best, scaler, lstm_model, device)

    noise_specs = _build_noise_specs(T, spike_scale, spike_probs,
                                     noise_stds, drift_mags,
                                     missing_rates, missing_method)
    results = {}
    for label, model, clean in [("FC-AE", fc_model, fc_clean),
                                  ("LSTM-AE", lstm_model, lstm_clean)]:
        print(f"  {label}")
        results[label] = _run_sweep(X_best, clean, scaler, model,
                                     device, noise_specs, n_runs, seed_offset=0)

    _plot_comparison(results, noise_specs, n, T, n_runs,
                     title="FC-AE vs LSTM-AE -- FC-AE best-N test set",
                     save_plot=save_plot, plot_path=plot_path)
    return results


def compare_noise_sweep_random(
    resampled_csv   = "resampled.csv",
    scaler_path     = "scaler.joblib",
    fc_model_path   = "fc_ae.pth",
    lstm_model_path = "lstm_ae.pth",
    gru_model_path  = None,
    N               = 100,
    sample_seed     = 42,
    n_runs          = 5,
    noise_stds      = None,
    drift_mags      = None,
    spike_probs     = None,
    spike_scale     = 4.0,
    missing_rates   = None,
    missing_method  = "linear",
    save_plot       = True,
    plot_path       = "exp_A_random.png",
):
    """
    Experiment A: Fair comparison -- all models evaluated on the same
    randomly sampled N trajectories (no selection bias).
    GRU-AE is included when gru_model_path is provided.
    """
    if noise_stds    is None: noise_stds    = np.linspace(0, 0.005, 11)
    if drift_mags    is None: drift_mags    = np.linspace(0, 0.005, 11)
    if spike_probs   is None: spike_probs   = np.linspace(0, 0.30,  11)
    if missing_rates is None: missing_rates = np.linspace(0, 0.50,  11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    fc_model,  T  = _load_fc_ae  (fc_model_path,   device)
    lstm_model, _ = _load_lstm_ae(lstm_model_path, device)
    gru_model     = None
    if gru_model_path is not None and os.path.exists(gru_model_path):
        gru_model, _ = _load_gru_ae(gru_model_path, device)

    df  = _cmp_load_csv(resampled_csv)
    cnt = df.groupby("flight_id")["x"].count()
    valid_ids = cnt.index[cnt == T].astype(str).to_numpy()
    rng    = np.random.default_rng(sample_seed)
    chosen = rng.choice(valid_ids, size=min(N, len(valid_ids)), replace=False)
    X_rand, _ = _cmp_build_X(df, chosen, T)
    n = X_rand.shape[0]
    print(f"[Exp-A] Random sample: {n}/{len(valid_ids)} flights  T={T}  seed={sample_seed}")

    model_list = [("FC-AE", fc_model), ("LSTM-AE", lstm_model)]
    if gru_model is not None:
        model_list.append(("GRU-AE", gru_model))

    noise_specs = _build_noise_specs(T, spike_scale, spike_probs,
                                     noise_stds, drift_mags,
                                     missing_rates, missing_method)
    results = {}
    for label, model in model_list:
        clean = _reconstruct(X_rand, scaler, model, device)
        print(f"  {label}")
        results[label] = _run_sweep(X_rand, clean, scaler, model,
                                     device, noise_specs, n_runs, seed_offset=99000)

    model_str = " vs ".join(k for k, _ in model_list)
    _plot_comparison(results, noise_specs, n, T, n_runs,
                     title=f"[Exp-A]  {model_str}  --  Random {n} Trajectories"
                           f"  (seed={sample_seed})",
                     save_plot=save_plot, plot_path=plot_path)
    return results


def compare_noise_sweep_separate_best(
    resampled_csv   = "resampled.csv",
    scaler_path     = "scaler.joblib",
    fc_model_path   = "fc_ae.pth",
    lstm_model_path = "lstm_ae.pth",
    gru_model_path  = None,
    N               = 100,
    n_runs          = 5,
    noise_stds      = None,
    drift_mags      = None,
    spike_probs     = None,
    spike_scale     = 4.0,
    missing_rates   = None,
    missing_method  = "linear",
    base_noise_std  = 5e-4,
    base_drift_mag  = 5e-4,
    base_spike_prob = 0.2,
    save_plot       = True,
    plot_path       = "exp_B_separate.png",
):
    """
    Experiment B: Each model evaluated on its own best-N trajectories.
    Results plotted in a Nx4 grid (N = number of models). Y-scales differ across
    rows -- absolute RMSE values are NOT comparable between models.
    GRU-AE is included when gru_model_path is provided.

    Noise design: 4 types (Gaussian / Drift / Spike / Missing). Gaussian, Drift,
    Spike use mixed one-at-a-time sweep with background noise at base values and
    spike amplitude = spike_scale × noise_std (matches zakData). Missing remains
    isolated (linear interpolation over dropped points).
    """
    if noise_stds    is None: noise_stds    = np.linspace(0, 0.005, 12)
    if drift_mags    is None: drift_mags    = np.linspace(0, 0.005, 12)
    if spike_probs   is None: spike_probs   = np.linspace(0, 0.30,  12)
    if missing_rates is None: missing_rates = np.linspace(0, 0.60,  13)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load(scaler_path)
    fc_model,  T  = _load_fc_ae  (fc_model_path,   device)
    lstm_model, _ = _load_lstm_ae(lstm_model_path, device)
    gru_model     = None
    if gru_model_path is not None and os.path.exists(gru_model_path):
        gru_model, _ = _load_gru_ae(gru_model_path, device)

    # ---- Score all flights ------------------------------------------------
    df  = _cmp_load_csv(resampled_csv)
    cnt = df.groupby("flight_id")["x"].count()
    all_ids = cnt.index[cnt == T].astype(str).to_numpy()
    print(f"[Exp-B] Total valid flights: {len(all_ids)}  T={T}")

    df2 = df[df["flight_id"].astype(str).isin(all_ids)].copy()
    df2["_r"] = df2.groupby("flight_id").cumcount()
    xw = df2.pivot(index="flight_id", columns="_r", values="x").dropna()
    yw = df2.pivot(index="flight_id", columns="_r", values="y").reindex(xw.index).dropna()
    xw = xw.reindex(yw.index)
    X_all = np.concatenate([xw.to_numpy(np.float32), yw.to_numpy(np.float32)], axis=1)

    Xs = scaler.transform(X_all).astype(np.float32)
    Xt = torch.from_numpy(Xs).to(device)
    B  = 1024

    @torch.no_grad()
    def _mse_all(model):
        """MSE between original-space input and original-space reconstruction."""
        parts = []
        for i in range(0, len(Xt), B):
            recon_sc = model(Xt[i:i+B]).float().cpu().numpy()
            parts.append(scaler.inverse_transform(recon_sc))
        recon_orig = np.concatenate(parts, 0)
        return ((X_all - recon_orig) ** 2).mean(axis=1)

    print("  Scoring all flights (FC-AE) ...")
    fc_mse   = _mse_all(fc_model)
    print("  Scoring all flights (LSTM-AE) ...")
    lstm_mse = _mse_all(lstm_model)

    fc_idx   = np.argsort(fc_mse)[:N]
    lstm_idx = np.argsort(lstm_mse)[:N]
    X_fc     = X_all[fc_idx]
    X_lstm   = X_all[lstm_idx]
    print(f"  FC-AE   best-{len(fc_idx)}: MSE [{fc_mse[fc_idx].min():.2e}, {fc_mse[fc_idx].max():.2e}]")
    print(f"  LSTM-AE best-{len(lstm_idx)}: MSE [{lstm_mse[lstm_idx].min():.2e}, {lstm_mse[lstm_idx].max():.2e}]")

    sweep_cfg = [("FC-AE",   fc_model,   X_fc,   len(fc_idx)),
                 ("LSTM-AE", lstm_model, X_lstm, len(lstm_idx))]

    if gru_model is not None:
        print("  Scoring all flights (GRU-AE) ...")
        gru_mse = _mse_all(gru_model)
        gru_idx = np.argsort(gru_mse)[:N]
        X_gru   = X_all[gru_idx]
        print(f"  GRU-AE  best-{len(gru_idx)}: MSE [{gru_mse[gru_idx].min():.2e}, {gru_mse[gru_idx].max():.2e}]")
        sweep_cfg.append(("GRU-AE", gru_model, X_gru, len(gru_idx)))

    noise_specs = _build_noise_specs_4type_mixed(
        T, spike_scale, spike_probs, noise_stds, drift_mags,
        missing_rates, missing_method,
        base_noise_std, base_drift_mag, base_spike_prob)
    results = {}
    print("[Exp-B] Running sweeps ...")
    for mname, model, X_best, n_best in sweep_cfg:
        clean_base = _reconstruct(X_best, scaler, model, device)
        print(f"  Model: {mname}")
        results[mname] = _run_sweep(X_best, clean_base, scaler, model,
                                     device, noise_specs, n_runs, seed_offset=77000)

    # ---- Nx4 boxplot plot ------------------------------------------------
    COLORS    = {"FC-AE": "#4878CF", "LSTM-AE": "#E87000", "GRU-AE": "#2CA02C"}
    ROW_LABEL = {mname: f"{mname} (own best-{n_best})"
                 for mname, _, _, n_best in sweep_cfg}

    n_rows = len(sweep_cfg)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.5 * n_rows), sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (mname, _, _, _) in enumerate(sweep_cfg):
        for col, (nname, xlabel, params, _) in enumerate(noise_specs):
            ax    = axes[row][col]
            color = COLORS[mname]
            ax.boxplot(
                results[mname][nname],
                positions=range(len(params)),
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                boxprops    =dict(facecolor=color, alpha=0.4),
                medianprops =dict(color=color, linewidth=2),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops    =dict(color=color, linewidth=1.2),
            )
            tick_idx = np.linspace(0, len(params) - 1, 5, dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([f"{params[i]:.3f}" for i in tick_idx],
                               fontsize=8, rotation=15)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_title(f"{ROW_LABEL[mname]}  |  {nname.split(chr(10))[0]}",
                         fontsize=10, fontweight="bold", color=color)
            ax.grid(True, alpha=0.3, axis="y")
            if col == 0:
                ax.set_ylabel("RMSE vs clean recon\n(original space, °)", fontsize=8)

    model_str = " & ".join(k for k, *_ in sweep_cfg)
    fig.suptitle(
        f"[Exp-B]  {model_str}  --  Each Model on Its Own Best-N\n"
        f"Mixed noise: base σ={base_noise_std:.4f}, drift={base_drift_mag:.4f}, "
        f"spike_prob={base_spike_prob}  |  spike amp = {spike_scale}×noise_std",
        fontsize=11, y=1.01, color="#333333"
    )
    fig.tight_layout()
    if save_plot:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {plot_path}")
    plt.show()
    return results


def _plot_comparison(results, noise_specs, n, T, n_runs,
                     title, save_plot, plot_path):
    COLORS = {"FC-AE": "#4878CF", "LSTM-AE": "#E87000", "GRU-AE": "#2CA02C"}
    MARKS  = {"FC-AE": "o",       "LSTM-AE": "s",       "GRU-AE": "^"}
    model_names = [m for m in ["FC-AE", "LSTM-AE", "GRU-AE"] if m in results]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
    for ax, (nname, xlabel, params, _) in zip(axes, noise_specs):
        for mname in model_names:
            _means = [np.mean(v) for v in results[mname][nname]]
            ax.plot(params, _means,
                    marker=MARKS[mname], color=COLORS[mname],
                    linewidth=2, markersize=5, label=mname)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mean RMSE vs clean recon\n(original space, °)", fontsize=9)
        ax.set_title(nname, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{title}\nn={n} | T={T} | {n_runs} runs/point",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    if save_plot:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {plot_path}")
    plt.show()
