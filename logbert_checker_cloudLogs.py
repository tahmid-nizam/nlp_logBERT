import os
import re
import json
import hashlib
import torch
import torch.nn as nn

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
# 7. Cloud log scoring + alert explanation
# =========================
def classify_cloud_alert(log_lines, score):
    joined = " ".join(log_lines).lower()
    reasons = []

    # auth / ssh patterns
    if "invalid user" in joined:
        reasons.append("invalid user attempts")
    if "failed password" in joined:
        reasons.append("failed password attempts")
    if "accepted password" in joined or "accepted publickey" in joined:
        reasons.append("successful login activity")
    if "connection closed" in joined:
        reasons.append("connection closure activity")
    if "authentication failure" in joined:
        reasons.append("authentication failure activity")

    # privilege / sudo patterns
    if "sudo:" in joined:
        reasons.append("privilege escalation or sudo activity")
    if "session opened" in joined or "session closed" in joined:
        reasons.append("session management activity")

    # system / service patterns
    if "error" in joined or "failed" in joined or "exception" in joined:
        reasons.append("explicit error or failure message")
    if "segfault" in joined or "panic" in joined:
        reasons.append("system crash indicator")
    if "cron" in joined:
        reasons.append("scheduled task activity")
    if "systemd" in joined or "service" in joined:
        reasons.append("service management activity")
    if "unauthorized" in joined or "denied" in joined:
        reasons.append("access denial activity")

    # risk level by score
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

    # alert type
    if "invalid user attempts" in reasons or "failed password attempts" in reasons:
        alert_type = "Authentication attack pattern"
    elif "privilege escalation or sudo activity" in reasons:
        alert_type = "Privilege activity anomaly"
    elif "system crash indicator" in reasons:
        alert_type = "System stability anomaly"
    elif "service management activity" in reasons:
        alert_type = "Service behavior anomaly"
    elif "successful login activity" in reasons and score >= 12:
        alert_type = "Unusual login sequence"
    elif "explicit error or failure message" in reasons:
        alert_type = "Failure-related anomaly"
    else:
        alert_type = "Unknown cloud log anomaly"

    if reasons:
        reason_text = ", ".join(sorted(set(reasons)))
    else:
        reason_text = "unusual log sequence"

    return {
        "alert_level": level,
        "alert_type": alert_type,
        "reason": reason_text
    }


def score_log_file(file_path, window_size=20, step_size=10):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total log lines loaded: {len(lines)}")

    results = []
    window_id = 0

    for start in range(0, max(1, len(lines) - window_size + 1), step_size):
        window_lines = lines[start:start + window_size]
        if len(window_lines) == 0:
            continue

        score = anomaly_score(window_lines)
        alert = classify_cloud_alert(window_lines, score)

        results.append({
            "window_id": window_id,
            "start_line": start + 1,
            "end_line": start + len(window_lines),
            "anomaly_score": score,
            "alert_level": alert["alert_level"],
            "alert_type": alert["alert_type"],
            "reason": alert["reason"],
            "sample_logs": window_lines[:3]
        })

        window_id += 1

    return results


if __name__ == "__main__":
    log_file = r"D:\Downloads\Capstone\Dataset\Collected\cloud_auth.log"
    # log_file = r"D:\Downloads\Capstone\Dataset\Collected\cloud_syslog.log"

    results = score_log_file(
        file_path=log_file,
        window_size=20,
        step_size=10
    )

    print("\nTop cloud alerts:")
    results = sorted(results, key=lambda x: x["anomaly_score"], reverse=True)

    for row in results[:10]:
        print("=" * 80)
        print(f"Window ID   : {row['window_id']}")
        print(f"Lines       : {row['start_line']}-{row['end_line']}")
        print(f"Score       : {row['anomaly_score']:.4f}")
        print(f"Alert Level : {row['alert_level']}")
        print(f"Alert Type  : {row['alert_type']}")
        print(f"Reason      : {row['reason']}")
        print("Sample logs:")
        for log_line in row["sample_logs"]:
            print(" -", log_line)
        print()