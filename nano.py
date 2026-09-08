#!/usr/bin/env python3
"""Pinned, vendored Nano toolkit. No sibling checkout or package install required."""
import hashlib
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parent
lock = json.loads((root / 'tools/nano-standard.lock.json').read_text())
bundle = root / 'tools/nano-standard.pyz'
if hashlib.sha256(bundle.read_bytes()).hexdigest() != lock['sha256']:
    raise SystemExit('Nano toolkit digest mismatch; reinstall the pinned bundle')
sys.path.insert(0, str(bundle))
from nano_standard.cli import main
raise SystemExit(main(['--root', str(root), *sys.argv[1:]]))
