import asyncio
import json
import re
import pytest
from gatepath.defenses import DefenseSession
from gatepath.local_stop_defenses import LocalStopDefenseSession, LocalStop, STOP_MESSAGE
from gatepath.request_response_protocol import RequestResponseBudget, get_request_response_prompt_bundle
from gatepath.request_response_runner import run_request_response_episode
from gatepath.experiment_graphs import build_named_topology
from test_request_response import DeterministicTeamModelClient, workspace_target_instance


def session(name='protectai'):
    return LocalStopDefenseSession(name, trusted_goal='', base_url='unused', model='unit')


def run(guard, graph='G(1,3)'):
    instance = workspace_target_instance()
    return run_request_response_episode(instance=instance,
        episode_id='local-stop-deterministic-test',
        topology=build_named_topology(graph, seed=0), role_seed=100,
        prompts=get_request_response_prompt_bundle('en', version='gatepath-request-response-system-handoff-forward-20260818-en'),
        condition='DEFENSE_UNIT', budget=RequestResponseBudget(max_messages=16, max_tool_iterations=4, max_runtime_seconds=20),
        model_client=DeterministicTeamModelClient(instance.parameters), defense=guard)


def test_pass_path_matches_legacy(monkeypatch):
    calls = []
    def post(url, payload):
        calls.append(payload)
        return {'blocked': False}
    monkeypatch.setattr('gatepath.defenses._post', post)
    old = DefenseSession('protectai', trusted_goal='', base_url='unused', model='unit')
    before = run(old); old_calls = list(calls); calls.clear()
    new = session(); after = run(new)
    # AgentDojo's synthetic send_email stamps wall time; it is not a protocol change.
    def normalize(value):
        serialized = re.sub(r'202\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?',
                      '<wall-time>', json.dumps(value, sort_keys=True))
        # AutoGen assigns fresh event UUIDs in both runs.
        return re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                      '<event-uuid>', serialized)
    assert normalize(calls) == normalize(old_calls)
    assert before.attack_success == after.attack_success
    assert before.usage == after.usage
    # Ignore detector timing only; detector text, decisions and disclosure identical.
    strip = lambda e: {k:v for k,v in e.items() if k != 'elapsed_seconds'}
    assert normalize([strip(e) for e in old.events]) == normalize([strip(e) for e in new.events])
    assert normalize([t.as_dict() for t in before.traces]) == normalize([t.as_dict() for t in after.traces])


def test_peer_block_stops_before_any_model_action(monkeypatch):
    monkeypatch.setattr('gatepath.defenses._post', lambda *a, **kw: {'blocked': True})
    guard = session(); report = run(guard)
    assert not report.attack_success
    assert report.usage['total_tokens'] == 0
    assert report.traces and all(t.response.content == STOP_MESSAGE for t in report.traces)
    assert guard.failure is None


@pytest.mark.parametrize('hook', ['tool', 'feedback'])
def test_local_tool_and_feedback_stop_is_not_episode_error(monkeypatch, hook):
    guard = session()
    original = guard.inspect_input
    async def inspect(text, *, agent, source):
        monkeypatch.setattr('gatepath.defenses._post',
            lambda *a, **kw: {'blocked': source.startswith(hook+':')})
        return await original(text, agent=agent, source=source)
    guard.inspect_input = inspect
    report = run(guard)
    assert guard.failure is None
    assert any(e['event']=='activation_stopped' for e in guard.events)
    assert any(t.response.content == STOP_MESSAGE for t in report.traces)
    assert report.usage['total_tokens'] > 0
    assert report.attack_success and report.target_effect_verified
    assert report.sensitive_tool_dispatch_count == 1
    assert not report.infrastructure_error


def test_one_block_does_not_stop_other_entry_contacts(monkeypatch):
    count = 0
    def post(*args, **kwargs):
        nonlocal count
        count += 1
        return {'blocked': count == 1}
    monkeypatch.setattr('gatepath.defenses._post', post)
    guard = session(); report = run(guard, 'G(3,3)')
    assert not report.infrastructure_error and guard.failure is None
    assert len(report.entry_responses) == 3
    assert sum(t.response.content == STOP_MESSAGE for t in report.traces) == 1
    assert any(t.autogen_events for t in report.traces)


def test_nested_activation_isolation_and_new_activation(monkeypatch):
    monkeypatch.setattr('gatepath.defenses._post', lambda *a, **kw: {'blocked': True})
    guard = session(); parent = guard.begin('parent')
    child = guard.begin('child')
    with pytest.raises(LocalStop):
        asyncio.run(guard.inspect_input('data', agent='a', source='peer:b'))
    with pytest.raises(LocalStop): guard.ensure_active()
    guard.end(child); guard.ensure_active()
    guard.end(parent)
    token = guard.begin('new'); guard.ensure_active(); guard.end(token)


def test_paper_prompt_uses_full_cumulative_activation_inputs(monkeypatch):
    calls=[]
    def post(url, request):
        calls.append(request)
        return {'choices':[{'finish_reason':'stop','message':{'content':'<<<No>>>'}}], 'usage':{}}
    monkeypatch.setattr('gatepath.local_stop_defenses._post', post)
    guard=session('llm_detector');token=guard.begin('one')
    async def check():
        await guard.inspect_input('TASK',agent='a',source='peer:b')
        await guard.inspect_input('RESULT',agent='a',source='tool:read')
    asyncio.run(check());guard.end(token)
    assert 'TASK\n\nRESULT' in calls[-1]['messages'][0]['content']
    assert len(calls[-1]['messages']) == 1 and 'response_format' not in calls[-1]
    assert calls[-1]['model'] == 'unit'


def test_http_error_is_preserved_without_retry_or_verdict(monkeypatch):
    import io
    import urllib.error
    from gatepath.defenses import DefenseError
    calls=[]
    body=b'{"error":{"message":"maximum context length exceeded"}}'
    def post(url,request):
        calls.append(request)
        raise urllib.error.HTTPError(url,400,'Bad Request',{},io.BytesIO(body))
    monkeypatch.setattr('gatepath.local_stop_defenses._post',post)
    guard=session('llm_detector');token=guard.begin('failed-activation')
    with pytest.raises(DefenseError,match='HTTPError'):
        asyncio.run(guard.inspect_input('FULL INPUT',agent='a',source='tool:read'))
    guard.end(token)
    assert len(calls)==1 and calls[0]['max_tokens']==1800
    event=next(e for e in guard.events if e['event']=='detector_http_error')
    assert event['request']==calls[0] and event['response_body']==body.decode()
    assert event['http_status']==400 and event['activation_id']=='failed-activation'
    assert not any(e['event'] in ('input_check','activation_stopped') for e in guard.events)
