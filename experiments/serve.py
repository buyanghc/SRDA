"""Print pinned serving settings; execute only with an explicit switch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODELS=json.loads((ROOT/'experiments/models.json').read_text())
MISTRAL_IMAGE='docker://vllm/vllm-openai@sha256:f2309d913a07da49ea20b2a694703f4cfcb5ad8e7437ec0f26145479ac01e002'

def make_command(a):
    profile=MODELS[a.model];runtime=ROOT/profile['runtime']
    if a.study=='defenses':runtime=ROOT/'runtimes/defenses'
    common=['serve',str(a.model_dir.resolve()),'--served-model-name',profile['model'],
            '--host','127.0.0.1','--port',str(a.port),'--max-model-len','8192',
            '--gpu-memory-utilization','0.90','--enable-auto-tool-choice']
    if 'ministral' in a.model:
        return ['apptainer','exec','--nv','--cleanenv','--bind',str(ROOT)+':'+str(ROOT)+':ro',
                '--bind',str(a.model_dir.resolve())+':'+str(a.model_dir.resolve())+':ro',MISTRAL_IMAGE,
                'vllm',*common,'--tokenizer-mode','mistral','--config-format','mistral',
                '--load-format','mistral','--tool-parser-plugin',str(runtime/'nci/plugins/mistral_json.py'),
                '--tool-call-parser','gatepath_mistral_json']
    if 'gpt-oss' in a.model:
        return [a.service_python,str(runtime/'nci/scripts/harmony_output_guard.py'),*common,
                '--dtype','bfloat16','--tool-call-parser','openai','--reasoning-parser','openai_gptoss',
                '--override-generation-config','{"temperature":0.0,"top_p":1.0,"top_k":-1}']
    if 'llama' in a.model:
        extra=['--dtype','bfloat16','--tool-call-parser','llama3_json','--chat-template',str(runtime/'nci/llama31_parallel_history.jinja')]
        extra+=['--override-generation-config','{"temperature":0.0,"top_p":1.0,"top_k":-1}']
        return [a.vllm,*common,*extra]
    extra=['--dtype','bfloat16','--reasoning-parser','qwen3','--tool-call-parser',
           'qwen3_coder' if a.model.startswith('qwen3.5') else 'hermes']
    if a.model.startswith('qwen3.5'):extra+=['--language-model-only']
    return [a.vllm,*common,*extra]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['controlled','workflows','defenses'],required=True)
    p.add_argument('--model',choices=list(MODELS),required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--port',type=int,default=8010)
    p.add_argument('--vllm',default='vllm');p.add_argument('--service-python',default=sys.executable)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    cmd=make_command(a)
    print(json.dumps({'command':cmd,'execute':a.execute,'context_tokens':8192},indent=2),flush=True)
    if a.execute:
        if not a.model_dir.is_dir():p.error('Model directory does not exist')
        subprocess.run([sys.executable,str(ROOT/'tools/verify.py'),'--sources-only'],check=True)
        if 'ministral' in a.model and os.environ.get('CUDA_VISIBLE_DEVICES'):
            position=cmd.index(MISTRAL_IMAGE)
            cmd[position:position]=['--env','CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES']]
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
