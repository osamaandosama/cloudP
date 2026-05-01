# Smart Banking AI — CISC 886 (Group 24)

A cloud-hosted, fine-tuned retail-banking chatbot built on **AWS** + **Google Colab**.

| | |
|---|---|
| **Course** | CISC 886 — Cloud Computing (Winter 2026) |
| **Institution** | School of Computing, Queen's University |
| **Group** | Group 24 |
| **Resource prefix** | `25fwmh` (every AWS resource carries this prefix) |
| **AWS region** | `us-east-1` (N. Virginia) |
| **Base model** | Qwen2.5-1.5B-Instruct (Apache-2.0) |
| **Dataset** | Bitext Retail Banking LLM Chatbot (CDLA-Sharing 1.0) |

---

## 1 · Repository layout

```
.
├── README.md                                    ← you are here
├── docs/
│   ├── CISC886_Group24_FinalReport.pdf          ← final report (PDF, submission copy)
│   └── CISC886_Group24_FinalReport.docx         ← editable Word source
├── preprocessing/
│   └── preprocess.py                            ← PySpark job (§4)
├── training/
│   └── banking_finetuning_v2.ipynb              ← Colab notebook (§5)
└── deployment/
    ├── app.py                                   ← Gradio UI (§7)
    ├── banking-api.service                      ← systemd unit, OpenAI API on :8000
    ├── banking-ui.service                       ← systemd unit, Gradio UI on :7860
    └── requirements.txt
```

---

## 2 · Prerequisites

| Tool / account | Why |
|---|---|
| AWS account with EMR + EC2 + S3 | Sections 2, 4, 6, 7 |
| AWS CLI configured for `us-east-1` | uploading scripts and downloading the GGUF model |
| Hugging Face account (free) | downloading the base model and dataset |
| Google account (free) | running the Colab fine-tuning notebook on a free T4 |
| SSH keypair | reaching the EC2 instance |

---

## 3 · Step-by-step replication

The replication is **end-to-end**: starting from an empty AWS account you can rebuild every component. Replace the prefix `25fwmh` with your own netID.

### 3.1   Create the VPC (§2)

In the AWS Console (region `us-east-1`):

1. **VPC** → *Create VPC* → name `25fwmh-vpc`, IPv4 CIDR `10.0.0.0/16`.
2. **Subnets** → *Create subnet* → name `25fwmh-subnet-public`, CIDR `10.0.0.0/20`, AZ `us-east-1a`.
3. **Internet gateways** → *Create* → attach to `25fwmh-vpc`.
4. **Route tables** → edit the public RT → add `0.0.0.0/0 → IGW`.
5. **Endpoints** → *Create* → `com.amazonaws.us-east-1.s3` (Gateway), associate with the public RT.
6. **Security groups** → *Create* → name `25fwmh-inference-sg`. Inbound rules:

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22   | TCP | your IP / 32 | SSH |
| 7860 | TCP | 0.0.0.0/0 | Gradio UI fallback |
| 8000 | TCP | 0.0.0.0/0 | OpenAI API (curl) |
| 8998 | TCP | 10.0.0.0/16 | EMR Livy (VPC only) |

### 3.2   Create the S3 bucket

```bash
aws s3 mb s3://25fwmh-bank-project --region us-east-1
aws s3api put-bucket-versioning \
    --bucket 25fwmh-bank-project \
    --versioning-configuration Status=Enabled
```

Upload the raw dataset (already pulled from Hugging Face) and the PySpark script:

```bash
aws s3 cp bitext-retail-banking-llm-chatbot-training-dataset.csv \
    s3://25fwmh-bank-project/input/

aws s3 cp preprocessing/preprocess.py \
    s3://25fwmh-bank-project/scripts/preprocess.py
```

### 3.3   Run the EMR + Spark preprocessing step (§4)

In the AWS Console → **EMR** → *Create cluster*:

| Field | Value |
|---|---|
| Name | `25fwmh-emr-cluster` |
| EMR release | `emr-7.13.0` |
| Applications | Hadoop, Hive, Spark, Livy, JupyterEnterpriseGateway |
| Primary instance | `m5.xlarge` × 1 (no Core, no Task) |
| Auto-terminate | after 1 hour idle |
| Log destination | `s3://25fwmh-bank-project/` |
| Subnet | `25fwmh-subnet-public` |

Then add a **Step**:

```
Type            : Spark application
Deploy mode     : Cluster
Application location : s3://25fwmh-bank-project/scripts/preprocess.py
```

This produces three Parquet files at `s3://25fwmh-bank-project/cleaned-data/`. The cluster auto-terminates after the step finishes — see **Figure 8** in the report for the Terminated status evidence.

### 3.4   Fine-tune on Colab (§5)

1. Open `training/banking_finetuning_v2.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all** (training takes ~13 minutes).
4. The notebook produces two GGUF artifacts in `/content/`:
   - `qwen2.5-1.5b-instruct.F16.gguf` (high-precision, used by Gradio)
   - `qwen2.5-1.5b-instruct-q4_k_m.gguf` (Q4_K_M, used by the API server)
5. Upload both to S3:

```bash
aws s3 cp qwen2.5-1.5b-instruct.F16.gguf \
    s3://25fwmh-bank-project/model-weights/
aws s3 cp qwen2.5-1.5b-instruct-q4_k_m.gguf \
    s3://25fwmh-bank-project/model-weights/
```

### 3.5   Launch the EC2 inference host (§6)

In the AWS Console → **EC2** → *Launch instances*:

| Field | Value |
|---|---|
| Name | `25fwmh-web-server` |
| AMI | Ubuntu Server 22.04 LTS (`ami-0ec10929233384c7f` or current) |
| Instance type | `t3.xlarge` |
| Key pair | your SSH keypair |
| Network | VPC `25fwmh-vpc`, subnet `25fwmh-subnet-public` |
| Security group | `25fwmh-inference-sg` |
| IAM role | `EMR_EC2_DefaultRole` (must allow `s3:GetObject` on the bucket) |

SSH in and run the bootstrap commands **verbatim**:

```bash
# 1. System update + Python toolchain
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv build-essential git awscli

# 2. Project directory and virtual environment
mkdir -p ~/banking && cd ~/banking
python3 -m venv banking-env
source banking-env/bin/activate
pip install --upgrade pip wheel setuptools

# 3. Install the LLM runner (with server extras) and Gradio
pip install "llama-cpp-python[server]" gradio uvicorn

# 4. Pull the GGUF artifacts from S3
aws s3 cp s3://25fwmh-bank-project/model-weights/qwen2.5-1.5b-instruct-q4_k_m.gguf .
aws s3 cp s3://25fwmh-bank-project/model-weights/qwen2.5-1.5b-instruct.F16.gguf .

# 5. Pull app.py from this repo
git clone <THIS_REPO_URL> /tmp/repo
cp /tmp/repo/deployment/app.py .
```

#### 3.5.1   Launch the OpenAI-compatible API server (port 8000)

```bash
python3 -m llama_cpp.server \
    --model qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8000
```

Or, equivalently, from inside a notebook:

```python
import subprocess
subprocess.Popen([
    "python3", "-m", "llama_cpp.server",
    "--model", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "--host", "0.0.0.0", "--port", "8000",
])
```

#### 3.5.2   Verify with curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "What is a credit card?"}]}'
```

You should see a JSON response containing
`"model": "qwen2.5-1.5b-instruct-q4_k_m.gguf"` and a coherent
natural-language definition under `choices[0].message.content`
(see **Figures 18 and 19** of the report).

### 3.6   Launch the Gradio UI (§7)

```bash
cd ~/banking
source banking-env/bin/activate
python app.py
```

Two URLs appear in the terminal:

```
* Running on local URL:  http://0.0.0.0:7860
* Running on public URL: https://<random>.gradio.live
```

### 3.7   Make both services auto-start on reboot (rubric requirement)

```bash
sudo cp deployment/banking-api.service /etc/systemd/system/
sudo cp deployment/banking-ui.service  /etc/systemd/system/
sudo systemctl daemon-reload

sudo systemctl enable  banking-api.service
sudo systemctl enable  banking-ui.service
sudo systemctl start   banking-api.service
sudo systemctl start   banking-ui.service

systemctl status banking-api.service   # active (running); enabled
systemctl status banking-ui.service    # active (running); enabled
```

### 3.8   Tear down (cost hygiene)

```bash
# EMR auto-terminates; nothing to do.
aws ec2 terminate-instances --instance-ids i-097a36bb8c1e0d139
aws s3 rm s3://25fwmh-bank-project --recursive   # optional
aws s3 rb s3://25fwmh-bank-project               # optional
```

---

## 4 · Cost summary

All prices are public `us-east-1` list prices.

| Resource | Unit price | Usage | Cost (USD) |
|---|---|---|---|
| EMR `m5.xlarge` Primary (Spark uplift incl.) | $0.096 / h | 12 min | $0.02 |
| EC2 `t3.xlarge` | $0.1664 / h | ~ 2.0 h | $0.33 |
| S3 storage (~ 4 GB GGUF + Parquet + logs) | $0.023 / GB-mo | prorated | < $0.01 |
| S3 requests (PUT + GET) | $0.005 / 1k PUT | < 500 reqs | < $0.01 |
| Data transfer out (gradio.live tunnel) | $0.09 / GB | ~ 200 MB | $0.00 |
| Google Colab (free T4) | — | 0.27 h | $0.00 |
| **Total (approximate)** | | | **≈ $0.36** |

Fine-tuning costs zero dollars because it runs on free Colab. The AWS bill is dominated by EC2 inference time. If the demo window had been closed immediately after grading, total cost would have dropped below $0.10.

---

## 5 · Mark-rubric mapping

| Section | Rubric requirement | Where in this repo |
|---|---|---|
| §1 | Architecture diagram + data flow | `docs/CISC886_Group24_FinalReport.pdf` §1 (Figure 1) |
| §2 | New VPC, CIDR, IGW, SG rules with justification | `docs/CISC886_Group24_FinalReport.pdf` §2 (Figure 2) |
| §3 | Model + dataset documentation, leakage strategy | `docs/CISC886_Group24_FinalReport.pdf` §3 |
| §4 | EMR config + PySpark code + S3 output + 3 EDA + Terminated screenshot | `preprocessing/preprocess.py` + report §4 (Figures 3–8) |
| §5 | Notebook + hyperparameters + base-vs-tuned comparison | `training/banking_finetuning_v2.ipynb` + report §5 (Figures 9–16) |
| §6 | EC2 type/AMI + exact commands + runner screenshot + curl response | `deployment/` + report §6 (Figures 17–19) |
| §7 | Web UI auto-start + sample conversation | `deployment/app.py` + `deployment/banking-ui.service` + report §7 (Figures 20–23) |
| Repo | README with replication + cost table | this file |

---

## 6 · Authors

| Name | Student ID |
|---|---|
| Ossama Adel Moursy | 20595449 |
| Abdalrhman Ibrahim | 20596377 |
| Aya Said Abdallah Noah | 20596338 |

## 7 · License

This repository is released for academic evaluation under the terms of the underlying components' licenses (Qwen2.5: Apache-2.0; Bitext dataset: CDLA-Sharing 1.0).
