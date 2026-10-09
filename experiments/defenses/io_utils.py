import hashlib,json,os
from pathlib import Path
def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)

def append(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def load_results(path):
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)  # Refuse silent reuse of a partial checkpoint.
            if row['key'] in result:
                raise RuntimeError('Duplicate completed attempt')
            result[row['key']] = row
    return result

def has_error(obj):
    if isinstance(obj, dict):
        return obj.get('infrastructure_error') is True or any(has_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_error(v) for v in obj)
    return False
