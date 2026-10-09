"""Serve pinned classifiers on localhost using the existing service environment."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = json.loads(Path(__file__).with_name('models.json').read_text())
    models = {}
    lock = threading.Lock()
    for name, spec in manifest.items():
        root = a.models / name / spec['revision']
        if json.loads((root / 'VERIFIED.json').read_text()) != spec:
            raise RuntimeError(f'Unverified classifier: {name}')
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(root, local_files_only=True).eval()
        bad_ids = [int(i) for i, label in model.config.id2label.items()
                   if label.upper() in {'INJECTION', 'JAILBREAK'}]
        if not bad_ids:
            raise RuntimeError(f'Unrecognized label mapping: {name}')
        models[name] = (tokenizer, model, bad_ids)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data):
            content = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, {'error': 'not found'})
            self.reply(200, {'ready': True, 'models': manifest, 'device': 'cpu'})

        def do_POST(self):
            if self.path != '/detect':
                return self.reply(404, {'error': 'not found'})
            try:
                length = int(self.headers['Content-Length'])
                if not 0 < length <= 2_000_000:
                    raise ValueError('Invalid payload length')
                data = json.loads(self.rfile.read(length))
                name, text = data['detector'], data['text']
                tokenizer, model, bad_ids = models[name]
                with lock, torch.inference_mode():
                    encoded = tokenizer(text, return_tensors='pt', padding=True,
                        truncation=True, max_length=512, stride=64,
                        return_overflowing_tokens=True)
                    encoded.pop('overflow_to_sample_mapping', None)
                    scores = []
                    for start in range(0, encoded['input_ids'].shape[0], 8):
                        batch = {k: v[start:start + 8] for k, v in encoded.items()}
                        probs = model(**batch).logits.softmax(dim=-1)
                        scores.extend(probs[:, bad_ids].sum(dim=-1).tolist())
                score = max(scores)
                self.reply(200, {'blocked': score > manifest[name]['threshold'],
                    'score': score, 'window_scores': scores, 'threshold': manifest[name]['threshold'],
                    'revision': manifest[name]['revision'], 'window_tokens': 512, 'overlap_tokens': 64})
            except Exception as exc:
                self.reply(500, {'error': type(exc).__name__ + ': ' + str(exc)})

        def log_message(self, format, *args):
            print(format % args, flush=True)

    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(json.dumps({'event': 'classifiers_ready', 'port': a.port}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
