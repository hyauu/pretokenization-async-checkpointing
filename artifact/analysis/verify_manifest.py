"""Verify the delivered package before regenerating derived files (stdlib only)."""
from pathlib import Path
import hashlib
root = Path(__file__).resolve().parents[1]
expected = {}
for line in (root / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
    digest, name = line.split('  ', 1)
    p = (root / name).resolve()
    assert p.is_relative_to(root), name
    assert p.is_file(), ('missing', name)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, ('hash mismatch', name)
    expected[name] = digest
actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
          if p.is_file() and '__pycache__' not in p.parts and p.name != 'MANIFEST.sha256'}
assert actual == set(expected), ('unexpected files', sorted(actual - set(expected)))
print(f'PASS: {len(expected)} member SHA-256 identities verified.')
