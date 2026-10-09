import asyncio, hashlib, json, time, traceback
from pathlib import Path
from io_utils import append, atomic_json, load_results, has_error
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'tools'))
from runtime import materialize
ROOT = materialize('defenses')
def make_plan(config, phase):
    instances = config['smoke_instance_ids'] if phase == 'smoke' else config['instance_ids']
    graphs = config['smoke_graphs'] if phase == 'smoke' else config['graphs']
    rows = []
    for i, (instance, graph) in enumerate((x, g) for x in instances for g in graphs):
        # Rotate order per paired case; never select cases from observed successes.
        names = config['defenses']
        for name in names[i % len(names):] + names[:i % len(names)]:
            rows.append({'key': f'{name}__{instance}__{graph}__g{config["graph_seed"]}r{config["role_seed"]}',
                         'instance_id': instance, 'graph': graph, 'defense': name,
                         'graph_seed': config['graph_seed'], 'role_seed': config['role_seed']})
    return rows

def summary(rows, records):
    groups = {}
    for name in dict.fromkeys(r['defense'] for r in rows):
        expected = [r for r in rows if r['defense'] == name]
        done = [records[r['key']] for r in expected if r['key'] in records]
        valid = [r for r in done if r['status'] == 'COMPLETED']
        groups[name] = {'planned': len(expected), 'recorded': len(done), 'valid': len(valid),
            'technical_errors': len(done) - len(valid),
            'attack_successes': sum(r['report']['attack_success'] for r in valid),
            'target_reached': sum(r['report']['target_agent_reached'] for r in valid),
            'episodes_with_input_block': sum(r['input_blocks'] > 0 for r in valid),
            'runtime_exhausted': sum(r['report']['runtime_budget_exhausted'] for r in valid),
            'wall_seconds': sum(r['elapsed_seconds'] for r in done),
            'agent_tokens': sum((r.get('report') or {}).get('usage', {}).get('total_tokens', 0) for r in done),
            'detector_tokens': sum(r['guard_usage'].get('total_tokens', 0) for r in done)}
    return {'planned': len(rows), 'recorded': len(records), 'groups': groups,
            'complete': len(records) == len(rows), 'denominator': 'all planned; technical errors exposed separately'}

async def run_one(row, config, model, args, folder):
    from gatepath.defenses import DefenseSession
    from gatepath.experiment_graphs import build_named_topology
    from gatepath.target_instances import load_formal_target_instances
    from gatepath.request_response_protocol import RequestResponseBudget, get_request_response_prompt_bundle
    from gatepath.request_response_runner import run_request_response_episode_async, _build_model_client, _openai_model_client_kwargs
    instances = {x.instance_id: x for x in load_formal_target_instances()}
    guard_class = DefenseSession
    if getattr(args, 'local_stop', False) and row['defense'] != 'none':
        from gatepath.local_stop_defenses import LocalStopDefenseSession
        guard_class = LocalStopDefenseSession
    guard = guard_class(row['defense'], trusted_goal='', base_url=args.base_url,
        model=model['model'], classifier_url=args.classifier_url,
        audit_sink=lambda event: append(folder / 'defense-audit.jsonl', {'episode_key': row['key'], **event}))
    started = time.monotonic()
    client = None
    report = None
    effective = {}
    try:
        client = _build_model_client(thinking_mode=model['thinking_mode'], reasoning_effort=model['reasoning_effort'],
            **_openai_model_client_kwargs(model=model['model'], base_url=args.base_url, api_key='EMPTY', temperature=0.0))
        effective = {k: client._create_args.get(k) for k in ('model', 'temperature', 'parallel_tool_calls', 'reasoning_effort', 'extra_body')}
        report = (await run_request_response_episode_async(
            instance=instances[row['instance_id']],
            topology=build_named_topology(row['graph'], seed=row['graph_seed']),
            prompts=get_request_response_prompt_bundle('en', version=config['prompt_version']),
            condition=config['condition'], budget=RequestResponseBudget(**config['budget']),
            episode_id=f'{args.campaign}-{row["key"]}', role_seed=row['role_seed'],
            model=model['model'], base_url=args.base_url, model_client=client,
            temperature=0.0, thinking_mode=model['thinking_mode'], reasoning_effort=model['reasoning_effort'],
            entry_agent_mode=config['entry_agent_mode'], response_visibility=config['response_visibility'],
            handoff_continuation=config['handoff_continuation'],
            defense=None if row['defense'] == 'none' else guard)).as_dict()
        if has_error(report) or guard.failure:
            raise RuntimeError('Technical execution/checker failure: ' + str(guard.failure))
        result = {'status': 'COMPLETED'}
    except Exception as exc:
        result = {'status': 'ERROR', 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if client is not None:
            await client.close()
    usage = {}
    for event in guard.events:
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[key] = usage.get(key, 0) + event.get('usage', {}).get(key, 0)
    return {**row, **result, 'campaign': args.campaign, 'report': report,
        'effective_client': effective, 'elapsed_seconds': time.monotonic() - started,
        'guard_usage': usage, 'input_checks': sum(x['event'] == 'input_check' for x in guard.events),
        'input_blocks': sum(x['event'] == 'input_check' and x.get('blocked', False) for x in guard.events),
        'guard_failure': guard.failure,
        'token_usage_incomplete': any(x.get('token_usage_incomplete', False) for x in guard.events)}
