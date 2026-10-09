import json,re
from gatepath.melon_llm_detector import parse_verdict,DetectorProtocolError
def classify(events):
    """Classify failed trials, never interpret prose as a detector verdict.

    User requested the general scheduling repair on 2026-10-06. A successful API
    response with invalid model output is a retained trial outcome, not an
    infrastructure outage. Unknown/malformed service failures still fail closed.
    """
    kinds=[]
    for i,e in enumerate(events):
        if e.get('event')!='defense_error': continue
        prev=events[i-1] if i else {}
        if prev.get('event')=='detector_http_error':
            body=prev.get('response_body','')
            msg=json.loads(body).get('error',{}).get('message','')
            input_limit=re.fullmatch(r"Input length \((\d+)\) exceeds model's maximum context length \(8192\)\.",msg)
            if ("maximum context length is 8192 tokens" in msg or
                (input_limit is not None and int(input_limit.group(1))>8192) or
                re.fullmatch(r'max_tokens must be at least 1, got -\d+\.',msg)):
                kinds.append('capacity');continue
        if prev.get('event')=='llm_call' and 'DetectorProtocolError' in e.get('error',''):
            choices=prev.get('raw_response',{}).get('choices',[])
            content=choices[0].get('message',{}).get('content') if len(choices)==1 else None
            if len(choices)==1 and choices[0].get('finish_reason')=='stop' and isinstance(content,str) and content.strip() in ('Yes.','No.'):
                kinds.append('punctuation');continue
            if len(choices)==1 and choices[0].get('finish_reason') in ('stop','length'):
                content=choices[0].get('message',{}).get('content')
                if isinstance(content,str):
                    if choices[0]['finish_reason']=='length':
                        kinds.append('truncated_detector_output');continue
                    try:parse_verdict(content)
                    except DetectorProtocolError:
                        kinds.append('invalid_detector_output');continue
        raise ValueError('Unclassified defense failure; retain and inspect')
    if not kinds: raise ValueError('No diagnostic evidence for ERROR')
    # Capacity has no authorized remedy; never replay mixed capacity/format errors.
    if 'invalid_detector_output' in kinds:return 'invalid_detector_output'
    if 'truncated_detector_output' in kinds:return 'truncated_detector_output'
    return 'capacity' if 'capacity' in kinds else 'punctuation'
