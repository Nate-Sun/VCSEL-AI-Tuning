"""Verify a clean compact release; standard library only, no hardware/network."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import sys

IGNORED_DIRS = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv-ai-tuning'}


def main():
    root = Path(__file__).resolve().parent.parent
    if os.name == 'nt' and not str(root).startswith('\\\\?\\'):
        root = Path('\\\\?\\' + str(root))
    with (root / 'MANIFEST_SHA256.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    names = [row['path'] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate manifest paths')
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or ':' in name:
            raise ValueError('Unsafe path')

    def check(row):
        p = root / row['path']
        try:
            if p.is_symlink() or not p.is_file():
                return ('missing_or_link', row['path'])
            if p.stat().st_size != int(row['bytes']):
                return ('changed', row['path'])
            h = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != row['sha256'].lower():
                return ('changed', row['path'])
        except OSError:
            return ('unreadable', row['path'])
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = [result for result in pool.map(check, rows) if result]
    actual = set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for name in files:
            actual.add(Path(parent, name).relative_to(root).as_posix())
    unlisted = sorted(actual - set(names) - {'MANIFEST_SHA256.csv'})
    print(json.dumps({'checked': len(rows), 'errors': len(errors), 'unlisted': len(unlisted)}))
    for kind, path in errors:
        print(kind + ': ' + path)
    for path in unlisted:
        print('unlisted: ' + path)
    return 1 if errors or unlisted else 0


if __name__ == '__main__':
    sys.exit(main())
