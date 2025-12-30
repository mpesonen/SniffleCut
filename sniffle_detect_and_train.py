import os
import json
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

# --------------------------------------------------
# Get script directory for resolving relative paths
# --------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add beats directory to path for BEATs imports
sys.path.insert(0, str(SCRIPT_DIR / "beats"))
from BEATs import BEATs, BEATsConfig

# --------------------------------------------------
# Verbose logging control
# --------------------------------------------------
_verbose = False


def set_verbose(enabled: bool):
    """Enable or disable verbose logging."""
    global _verbose
    _verbose = enabled


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if _verbose:
        print(*args, **kwargs, file=sys.stderr)


# --------------------------------------------------
# Device setup (Apple Silicon / CUDA / CPU)
# --------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# --------------------------------------------------
# Load BEATs backbone (self-supervised checkpoint)
# --------------------------------------------------
beats_checkpoint_path = SCRIPT_DIR / "beats_pretrained_models" / "BEATs_iter3_plus_AS2M.pt"
if not beats_checkpoint_path.exists():
    raise FileNotFoundError(
        f"BEATs checkpoint not found at {beats_checkpoint_path}\n"
        "Make sure BEATs_iter3_plus_AS2M.pt is in the same directory as beats_test.py"
    )

checkpoint = torch.load(
    str(beats_checkpoint_path),
    map_location=device,
    weights_only=False,
)

cfg = BEATsConfig(checkpoint["cfg"])
beats_model = BEATs(cfg)
beats_model.load_state_dict(checkpoint["model"])
beats_model.to(device)
beats_model.eval()

# Freeze BEATs parameters (we only train the Sniffle head)
for p in beats_model.parameters():
    p.requires_grad = False


# --------------------------------------------------
# Model caching for inference
# --------------------------------------------------
_sniffle_head_cache = None
_sniffle_head_checkpoint_path = None


def _get_cached_sniffle_head(head_checkpoint: str):
    """Load and cache sniffle head model to avoid reloading on each inference call."""
    global _sniffle_head_cache, _sniffle_head_checkpoint_path
    
    # Return cached model if checkpoint path matches
    if _sniffle_head_cache is not None and _sniffle_head_checkpoint_path == head_checkpoint:
        return _sniffle_head_cache
    
    # Load new model
    if not os.path.exists(head_checkpoint):
        raise FileNotFoundError(
            f"{head_checkpoint} not found. Train the head first with train_sniffle_head()."
        )
    
    sniffle_head = SniffleHead(input_dim=beats_model.cfg.encoder_embed_dim).to(device)
    state = torch.load(head_checkpoint, map_location=device, weights_only=False)
    sniffle_head.load_state_dict(state)
    sniffle_head.eval()
    
    # Update cache
    _sniffle_head_cache = sniffle_head
    _sniffle_head_checkpoint_path = head_checkpoint
    
    return sniffle_head


# --------------------------------------------------
# Sniffle head on top of frozen BEATs
# --------------------------------------------------
class SniffleHead(nn.Module):
    """
    Binary classifier head that takes BEATs frame embeddings and outputs a sniffle logit.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Tiny attention pooling over frames
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classic MLP head: Linear -> GELU -> Dropout -> Linear
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, D] BEATs frame embeddings.
        Returns:
            logits: [B] raw logits for "sniffle" (use with BCEWithLogitsLoss).
        """
        # Attention scores over time: [B, T, 1]
        scores = self.attn(features)
        # Convert to normalized attention weights over frames: [B, T]
        weights = torch.softmax(scores.squeeze(-1), dim=1)
        # Weighted sum over time -> [B, D]
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)

        logits = self.net(pooled).squeeze(-1)
        return logits


# --------------------------------------------------
# FluSense dataset wrapper (binary sniffle / no-sniffle)
# --------------------------------------------------
class FluSenseSniffleDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = "train"):
        # Refer to https://huggingface.co/datasets/vtsouval/flusense/tree/main for the dataset structure
        self.ds = load_dataset("vtsouval/flusense", split=split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        audio = ex["audio"]  # TorchCodec AudioDecoder
        samples = audio.get_all_samples()
        wav = samples.data
        # Ensure mono [T] regardless of number of channels
        if wav.dim() == 2:
            wav = wav.mean(0)
        sr = samples.sample_rate

        # Resample to 16 kHz for BEATs
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)

        # Limit clip length to avoid extremely long sequences (memory blow-up in attention)
        max_seconds = 10.0
        max_samples = int(target_sr * max_seconds)
        if wav.numel() > max_samples:
            start = torch.randint(0, wav.numel() - max_samples + 1, (1,)).item()
            wav = wav[start:start + max_samples]

        label_str = ex["label"]
        label = 1.0 if label_str == "sniffle" else 0.0
        return wav, label


def collate_fn(batch):
    """
    Pad variable-length waveforms and build a padding mask for BEATs.
    """
    wavs, labels = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    wavs = pad_sequence(wavs, batch_first=True)  # [B, T_max]

    max_len = wavs.size(1)
    idxs = torch.arange(max_len)
    padding_mask = idxs.unsqueeze(0) >= lengths.unsqueeze(1)  # [B, T_max], True = pad

    labels = torch.tensor(labels, dtype=torch.float32)
    return wavs, padding_mask, labels


# --------------------------------------------------
# Training loop for the Sniffle head
# --------------------------------------------------
def train_sniffle_head(
    num_epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 5,
    weight_decay: float = 1e-4,
):
    """
    Train sniffle head with overfitting prevention:
    - Train/validation split
    - Early stopping based on validation loss
    - Weight decay (L2 regularization)
    - Saves best model based on validation performance
    """
    # Create train/validation split
    full_ds = FluSenseSniffleDataset(split="train")
    train_size = int((1 - val_split) * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    sniffle_head = SniffleHead(input_dim=beats_model.cfg.encoder_embed_dim).to(device)
    optimizer = torch.optim.Adam(sniffle_head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        sniffle_head.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            disable=False,
        )

        for wavs, padding_mask, labels in progress_bar:
            wavs = wavs.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features, _ = beats_model.extract_features(wavs, padding_mask=padding_mask)

            logits = sniffle_head(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_curr = labels.size(0)
            total_loss += loss.item() * batch_size_curr
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_examples += batch_size_curr

            avg_loss = total_loss / max(total_examples, 1)
            acc = total_correct / max(total_examples, 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

        train_loss = avg_loss
        train_acc = acc

        # Validation phase
        sniffle_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_examples = 0

        val_progress_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [val]",
            unit="batch",
            disable=False,
        )

        with torch.no_grad():
            for wavs, padding_mask, labels in val_progress_bar:
                wavs = wavs.to(device)
                padding_mask = padding_mask.to(device)
                labels = labels.to(device)

                features, _ = beats_model.extract_features(wavs, padding_mask=padding_mask)
                logits = sniffle_head(features)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_examples += labels.size(0)

                # Update progress bar with current metrics
                avg_val_loss = val_loss / max(val_examples, 1)
                avg_val_acc = val_correct / max(val_examples, 1)
                val_progress_bar.set_postfix({"loss": f"{avg_val_loss:.4f}", "acc": f"{avg_val_acc:.4f}"})

        val_loss = val_loss / max(val_examples, 1)
        val_acc = val_correct / max(val_examples, 1)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = sniffle_head.state_dict().copy()
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered after {patience} epochs without improvement")
                break

    saved_checkpoint_path = f"../training_checkpoints/sniffle_head_{datetime.now().strftime('%Y%m%d_%H%M')}.pt"

    # Save best model
    if best_model_state is not None:
        sniffle_head.load_state_dict(best_model_state)
    torch.save(sniffle_head.state_dict(), "sniffle_head.pt")
    print(f"Saved best sniffle head (val_loss={best_val_loss:.4f}) to sniffle_head.pt")


@torch.no_grad()
def inference_sniffle_head(
    wav_path: str = None,
    threshold: float = 0.5,
    head_checkpoint: str = None,
    window_ms: float = 200.0,
    hop_ms: float = 50.0,
) -> str:
    """
    Main inference function: Load a trained sniffle head and run it on top of BEATs
    (a known audio representation model) on a local WAV file in a sliding-window manner.
    Returns JSON with list of (start_time, end_time) pairs.
    
    Both BEATs model and sniffle head are cached to avoid reloading on subsequent calls.

    Args:
        wav_path: Path to WAV file. Defaults to ../audio_files/sniffle_test.wav relative to script.
        threshold: Probability threshold for sniffle vs no sniffle.
        head_checkpoint: Path to trained sniffle head weights. Defaults to ../training_checkpoints/sniffle_head_3epochs.pt relative to script.
        window_ms: Internal window length in milliseconds (min 200 ms for BEATs).
        hop_ms: Hop size between window starts in milliseconds (default 50 ms).
    
    Returns:
        JSON string containing a list of dictionaries with start, end, and probability.
    """
    # Set defaults relative to script directory if not provided
    if head_checkpoint is None:
        head_checkpoint = str(SCRIPT_DIR / "training_checkpoints" / "sniffle_head_3epochs.pt")
    
    # Load sniffle head (cached - only loads once per checkpoint path)
    sniffle_head = _get_cached_sniffle_head(head_checkpoint)

    # Load and preprocess WAV
    wav, sr = torchaudio.load(wav_path)
    target_sr = 16000
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.dim() == 2:
        wav = wav.mean(0)
    wav = wav.to(device)  # [T], full file

    # Build overlapping windows.
    # We keep the *time step* between window starts at `hop_ms`,
    # but enforce a minimum internal window length so BEATs has enough
    # context for its 16x16 patch embedding.
    min_window_ms = max(window_ms, 200.0)  # ~0.2s min window
    win_size = int(target_sr * (min_window_ms / 1000.0))  # samples per window
    hop_size = int(target_sr * (hop_ms / 1000.0))         # step between window starts

    # Slide across the *entire* file in hops, padding the last window if needed
    num_samples = wav.numel()
    windows = []
    starts = []
    ends = []

    start = 0
    while start < num_samples:
        end = min(start + win_size, num_samples)
        chunk = wav[start:end]
        real_end = end  # for reporting timestamps

        if chunk.numel() < win_size:
            pad = torch.zeros(win_size - chunk.numel(), device=wav.device, dtype=wav.dtype)
            chunk = torch.cat([chunk, pad], dim=0)

        windows.append(chunk)
        starts.append(start)
        ends.append(real_end)

        start += hop_size

    windows = torch.stack(windows, dim=0)  # [N, win_size]

    # Extract BEATs features and run head on all windows in a batch
    features, _ = beats_model.extract_features(windows)  # [N, T', D]
    logits = sniffle_head(features)  # [N]
    probs = torch.sigmoid(logits).cpu()

    sniffle_windows = []
    for start_idx, end_idx, p in zip(starts, ends, probs):
        p_val = float(p.item())
        if p_val >= threshold:
            t_start = start_idx / target_sr
            t_end = end_idx / target_sr
            sniffle_windows.append((t_start, t_end, p_val))

    # Merge overlapping / contiguous windows into continuous segments.
    segments = []
    if sniffle_windows:
        merged = []
        cur_start, cur_end, cur_max_p = sniffle_windows[0]
        for t_start, t_end, p in sniffle_windows[1:]:
            if t_start <= cur_end:
                # Extend current segment
                cur_end = max(cur_end, t_end)
                cur_max_p = max(cur_max_p, p)
            else:
                merged.append((cur_start, cur_end, cur_max_p))
                cur_start, cur_end, cur_max_p = t_start, t_end, p
        merged.append((cur_start, cur_end, cur_max_p))
        
        segments = [
            {
                "start": round(float(t_start), 3),
                "end": round(float(t_end), 3),
                "probability": round(float(probability), 2)
            }
            for t_start, t_end, probability in merged
        ]

    # Return JSON string
    return json.dumps(segments)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test sniffle head on top of BEATs.")
    parser.add_argument("--mode", choices=["train", "test"], default="test", help="Whether to train or test the head.")
    parser.add_argument(
        "--wav",
        type=str,
        help="Path to WAV file for testing."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (default: 0.2).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization) for optimizer.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for sniffle vs no sniffle.")
    parser.add_argument("--window_ms", type=float, default=200.0, help="Internal window length in ms for BEATs.")
    parser.add_argument("--hop_ms", type=float, default=50.0, help="Hop size between window starts in ms.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write JSON results to a file.")
    parser.add_argument(
        "--head_checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "training_checkpoints" / "sniffle_head_3epochs.pt"),
        help="Path to trained sniffle head weights."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (messages go to STDERR).")

    # Print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    
    # Set verbose mode
    set_verbose(args.verbose)
    vprint("Using device:", device)

    if args.mode == "train":
        train_sniffle_head(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            patience=args.patience,
            weight_decay=args.weight_decay,
        )
    else:
        result_json = inference_sniffle_head(
            wav_path=args.wav,
            threshold=args.threshold,
            head_checkpoint=args.head_checkpoint,
            window_ms=args.window_ms,
            hop_ms=args.hop_ms,
        )
        # Always output JSON to STDOUT
        print(result_json)
        
        # Write to file if output path is provided
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result_json)
            vprint(f"\nResults written to: {args.output}")
