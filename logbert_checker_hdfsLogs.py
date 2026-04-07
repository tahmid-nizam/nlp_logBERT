import os
import re
import json
import hashlib
import torch
import torch.nn as nn
import pandas as pd

# =========================
# 1. Local paths
# =========================
BASE_DIR = r"D:\Downloads\Capstone\Project\logOutput_v2\light_training_state"

CONFIG_FILE = os.path.join(BASE_DIR, "training_config.json")
VOCAB_FILE = os.path.join(BASE_DIR, "small_token2idx.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "model_checkpoint_light.pt")

# =========================
# 2. Load config and vocab
# =========================
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    small_token2idx = json.load(f)

PAD_ID = cfg["PAD_ID"]
UNK_ID = cfg["UNK_ID"]
MASK_ID = cfg["MASK_ID"]
MAX_LEN = cfg["max_len"]
FINAL_VOCAB_SIZE = cfg["final_vocab_size"]

device = torch.device("cpu")

# =========================
# 3. Model
# =========================
class SimpleLogBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=1, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.transformer(x)
        return self.output_layer(x)

model = SimpleLogBERT(
    vocab_size=FINAL_VOCAB_SIZE,
    embed_dim=cfg["embed_dim"],
    num_heads=cfg["num_heads"],
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    max_len=MAX_LEN
).to(device)

checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# 4. Template normalization
# =========================
ip_port_pattern = re.compile(r'\b\d+\.\d+\.\d+\.\d+:\d+\b')
ip_pattern = re.compile(r'\b\d+\.\d+\.\d+\.\d+\b')
block_pattern = re.compile(r'blk_-?\d+')
hex_pattern = re.compile(r'\b0x[a-fA-F0-9]+\b')
uuid_like_pattern = re.compile(r'\b[a-f0-9]{8,}\b', re.IGNORECASE)
path_pattern = re.compile(r'(/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+)')
host_num_pattern = re.compile(r'([A-Za-z_\-]+)-\d+\b')
long_num_pattern = re.compile(r'\b\d{2,}\b')
standalone_num_pattern = re.compile(r'\b\d+\b')

def make_template(text: str) -> str:
    text = str(text)
    text = ip_port_pattern.sub('<IP_PORT>', text)
    text = ip_pattern.sub('<IP>', text)
    text = block_pattern.sub('<BLOCK>', text)
    text = hex_pattern.sub('<HEX>', text)
    text = uuid_like_pattern.sub('<ID>', text)
    text = path_pattern.sub('<PATH>', text)
    text = host_num_pattern.sub(r'\1-<NUM>', text)
    text = long_num_pattern.sub('<NUM>', text)
    text = standalone_num_pattern.sub('<NUM>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def event_id_from_template(template: str) -> str:
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]

# =========================
# 5. Encode session
# =========================
def encode_session(log_lines):
    event_ids = []

    for line in log_lines:
        template = make_template(line)
        event_id = event_id_from_template(template)
        token_id = small_token2idx.get(event_id, UNK_ID)
        event_ids.append(token_id)

    event_ids = event_ids[:MAX_LEN]

    if len(event_ids) < MAX_LEN:
        event_ids += [PAD_ID] * (MAX_LEN - len(event_ids))

    return torch.tensor([event_ids], dtype=torch.long, device=device)

# =========================
# 6. Score session
# =========================
loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def anomaly_score(log_lines):
    x = encode_session(log_lines)

    labels = x.clone()
    inputs = x.clone()

    mask = inputs != PAD_ID
    inputs[mask] = MASK_ID
    labels[~mask] = -100

    with torch.no_grad():
        logits = model(inputs)
        token_losses = loss_fn(
            logits.view(-1, FINAL_VOCAB_SIZE),
            labels.view(-1)
        ).view(1, MAX_LEN)

    masked_positions = mask[0]
    if masked_positions.sum().item() == 0:
        return 0.0

    return float(token_losses[0][masked_positions].mean().item())

# =========================
# 7. HDFS alert explanation layer
# =========================
import pandas as pd

# =========================
# Score HDFS structured CSV
# =========================
def score_hdfs_structured_csv(csv_path, window_size=20, step_size=10):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "Content" not in df.columns:
        raise ValueError("CSV must contain a 'Content' column.")

    lines = df["Content"].dropna().astype(str).tolist()
    print(f"Total HDFS log lines loaded: {len(lines)}")

    results = []
    window_id = 0

    for start in range(0, max(1, len(lines) - window_size + 1), step_size):
        window_lines = lines[start:start + window_size]
        if not window_lines:
            continue

        score = anomaly_score(window_lines)

        results.append({
            "window_id": window_id,
            "start_row": start,
            "end_row": start + len(window_lines) - 1,
            "anomaly_score": score,
            "window_logs": window_lines
        })
        window_id += 1

    results = sorted(results, key=lambda x: x["anomaly_score"], reverse=True)
    return results

# =========================
# HDFS alert explanation
# =========================
def classify_hdfs_alert(log_lines, score):
    joined = " ".join(log_lines).lower()
    reasons = []

    if "packetresponder" in joined:
        reasons.append("packet responder activity")
    if "receiving" in joined or "received block" in joined:
        reasons.append("block transfer / receive activity")
    if "addstoredblock" in joined or "blockmap" in joined:
        reasons.append("metadata or block map update activity")
    if "terminating" in joined:
        reasons.append("unexpected responder termination pattern")
    if "exception" in joined or "error" in joined or "failed" in joined:
        reasons.append("explicit error/failure message")
    if "replica" in joined:
        reasons.append("replica handling activity")

    if score >= 15:
        level = "CRITICAL"
    elif score >= 12:
        level = "HIGH"
    elif score >= 9:
        level = "MEDIUM"
    elif score >= 6:
        level = "LOW"
    else:
        level = "INFO"

    if reasons:
        reason_text = ", ".join(sorted(set(reasons)))
    else:
        reason_text = "unusual HDFS event sequence"

    if "block transfer / receive activity" in reason_text and "packet responder activity" in reason_text:
        alert_type = "Block transfer anomaly"
    elif "metadata or block map update activity" in reason_text:
        alert_type = "Metadata update anomaly"
    elif "unexpected responder termination pattern" in reason_text:
        alert_type = "Responder termination anomaly"
    elif "explicit error/failure message" in reason_text:
        alert_type = "Failure-related anomaly"
    else:
        alert_type = "Unknown sequence anomaly"

    return {
        "alert_level": level,
        "alert_type": alert_type,
        "reason": reason_text
    }

# =========================
# Main
# =========================
if __name__ == "__main__":
    hdfs_file = r"D:\Downloads\Capstone\Dataset\Collected\HDFS_2k.log_structured.csv"

    results = score_hdfs_structured_csv(
        csv_path=hdfs_file,
        window_size=20,
        step_size=10
    )

    print("\nTop HDFS alerts:")
    for row in results[:10]:
        alert = classify_hdfs_alert(row["window_logs"], row["anomaly_score"])

        print("=" * 80)
        print(f"Window ID   : {row['window_id']}")
        print(f"Rows        : {row['start_row']}-{row['end_row']}")
        print(f"Score       : {row['anomaly_score']:.4f}")
        print(f"Alert Level : {alert['alert_level']}")
        print(f"Alert Type  : {alert['alert_type']}")
        print(f"Reason      : {alert['reason']}")
        print("Sample logs:")
        for log_line in row["window_logs"][:3]:
            print(" -", log_line)
        print()


#    hdfs_file = r"D:\Downloads\Capstone\Dataset\Collected\HDFS_2k.log_structured.csv"
