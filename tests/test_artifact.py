"""CPU-only protocol/matrix/launcher checks; no model calls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ArtifactTests(unittest.TestCase):
    def test_six_configurations_only(self):
        models=json.loads((ROOT/'experiments/models.json').read_text())
        self.assertEqual(len(models),6)
        self.assertFalse(any('granite' in k or 'gemma' in k for k in models))

    def test_defense_protocol(self):
        cfg=json.loads((ROOT/'experiments/defenses/config.json').read_text())
        self.assertEqual(len(set(cfg['instance_ids'])),45)
        self.assertEqual(cfg['budget'],dict(max_messages=16,max_tool_iterations=4,max_runtime_seconds=90))
        self.assertEqual(cfg['defenses'],['none','protectai','promptguard','llm_detector'])
        self.assertEqual(cfg['seed_pairs'],[dict(graph_seed=i,role_seed=100+i) for i in range(3)])
        self.assertEqual(cfg['graphs'],['G(1,3)','G(3,3)','G33-P3'])

    def test_all_launcher_plans(self):
        models=json.loads((ROOT/'experiments/models.json').read_text())
        for model in models:
            for study,n in [('controlled',1890),('workflows',54),('defenses',540)]:
                with self.subTest(model=model,study=study):
                    output=subprocess.check_output([sys.executable,str(ROOT/'experiments/run.py'),study,'--model',model],text=True)
                    info=json.loads(output);self.assertFalse(info['execute']);self.assertEqual(info['full_cell_size'],n)
                    self.assertNotIn('--execute',info['command'])

    def test_immutable_rq4_condition(self):
        r=subprocess.run([sys.executable,str(ROOT/'experiments/run.py'),'defenses','--model','qwen3-14b','--condition','system-terminal'],capture_output=True)
        self.assertNotEqual(r.returncode,0)

    def test_serving_plan_paths(self):
        models=json.loads((ROOT/'experiments/models.json').read_text())
        for model in models:
            for study in ['controlled','defenses']:
                output=subprocess.check_output([sys.executable,str(ROOT/'experiments/serve.py'),'--study',study,'--model',model,'--model-dir','/diagnostic/model'],text=True)
                info=json.loads(output);self.assertFalse(info['execute']);self.assertEqual(info['context_tokens'],8192)
                for token in info['command']:
                    if token.startswith(str(ROOT)):self.assertTrue(Path(token.split(':')[0]).exists(),token)

if __name__=='__main__':unittest.main()
