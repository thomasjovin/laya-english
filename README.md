# laya-english

Offline mirror of the **English** checkpoint of [Laya](https://huggingface.co/convaiinnovations/laya) by Convai Innovations (Apache-2.0), packaged so it can be used on a machine without Hugging Face access.

- Source: `convaiinnovations/laya`, root checkpoint, commit `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1`
- Encoder: ModernBERT-large (~421M params), max 512 tokens
- Multilingual and typed-decisions checkpoints are **not** included

## Contents

| Path | What |
|---|---|
| `model/` | Checkpoint dir: `rl_agent_config.json`, `encoder/`, `tokenizer/`, original model card (`MODEL_CARD.md`), eval results, helper scripts |
| `model/model.safetensors` | **Not in git** (843 MB > GitHub's 100 MB limit). Download from the [v1.0 release](https://github.com/thomasjovin/laya-english/releases/tag/v1.0) |
| `wheels/` | `laya-0.5.0` and `vegaml-0.17.2` wheels from PyPI |
| `classify_intents.py` | Offline intent-classification example |

## Setup on the offline machine

```bash
git clone https://github.com/thomasjovin/laya-english && cd laya-english

# Weights (843 MB) from the GitHub release
curl -L -o model/model.safetensors \
  https://github.com/thomasjovin/laya-english/releases/download/v1.0/model.safetensors
echo "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c  model/model.safetensors" | sha256sum -c

# Python >= 3.10. torch/transformers/etc. come from PyPI (or your internal mirror)
pip install wheels/laya-0.5.0-py3-none-any.whl wheels/vegaml-0.17.2-py3-none-any.whl
```

If `gh` is available: `gh release download v1.0 -R thomasjovin/laya-english -D model`.

## Usage

```bash
HF_HUB_OFFLINE=1 python classify_intents.py "I forgot my password" "where's my parcel"
```

```python
import laya
agent = laya.Agent("model", vegaml=False)   # local dir, classic encoder, no Hub calls
q = {"intent": {"type": "choice", "instructions": "What is the customer's intent?",
                "criteria": {"reset_password": "reset or recover password",
                             "track_order": "where is my order", "other": "anything else"}}}
print(laya.predict_tournament(agent, "where's my parcel", q)["answers"]["intent"])
```

Notes for many intent classes:

- `agent.predict` fits roughly 120 short labels in the 512-token budget, then raises `ValueError`. `laya.predict_tournament` (used in the example) splits labels into groups of 16, so hundreds of intents work; it costs ~4 s per message on CPU at 150 labels.
- The checkpoint's temperature for 11+ options is out of range and gets clamped (you'll see a `RuntimeWarning`), so confidences for large label sets are uncalibrated until you calibrate on your own labelled data (`agent.calibrate` / `agent.fit`, see `model/MODEL_CARD.md`).

## License

Apache-2.0, as published by Convai Innovations. See `model/MODEL_CARD.md` and the [upstream repo](https://github.com/NandhaKishorM/laya).
