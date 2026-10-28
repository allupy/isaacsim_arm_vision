import json,time
def recording(root,identifier,action):
    path=root/'sim/video_request.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(id=identifier,action=action)));temp.replace(path)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        try:state=json.loads((root/'sim/video_state.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):state={}
        if state.get('id')==identifier and state.get('state')==('recording' if action=='start' else 'finished'):return state
        time.sleep(.2)
    raise TimeoutError('Video capture acknowledgement timeout')
