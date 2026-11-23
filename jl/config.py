"""One config for every script.

The system prompt, model names and eval settings live in
judgment.config.json at the repo root. Nothing else may hard-code them:
a prompt that differs between training, deployment and the blind test is
a confound, and the test stops meaning anything.

Environment overrides (kept from v0): JL_SPECIALIST, JL_BASE, JL_OLLAMA.
"""

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "judgment.config.json")

_ENV = {
    "specialist": "JL_SPECIALIST",
    "base_model": "JL_BASE",
    "ollama_url": "JL_OLLAMA",
}


def load(path=CONFIG_PATH):
    with open(path) as f:
        cfg = json.load(f)
    for key, env in _ENV.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


def config_hash(cfg):
    """Short stable fingerprint of the config, for manifests."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
