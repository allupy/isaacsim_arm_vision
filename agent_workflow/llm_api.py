"""OpenAI-compatible HTTP transport; secrets are supplied at runtime only."""
import json,urllib.request

class APIPlannerProvider:
    def __init__(self,base_url,model,key):
        self.base_url=base_url.rstrip('/');self.model=model;self.key=key;self.last_response=None
        if not self.base_url.startswith('https://'):raise ValueError('HTTPS endpoint required')
    def request(self,path,data=None):
        req=urllib.request.Request(self.base_url+path,data=None if data is None else json.dumps(data).encode(),
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
    def models(self):return [x['id'] for x in self.request('/models').get('data',[])]
    def __call__(self,instruction,schema):
        prompt=('Translate the user instruction into a JSON array of ordered robot skills. Each item has exactly '
            'target, destination, skill. Apples map to apple, cups to cup, 水果区 to fruit_zone, 杯子区 to cup_zone. '
            '盘子 maps to plate; 杯架 maps to cup_stand. Do not invent targets, coordinates or additional actions. Unsupported or ambiguous requests return []. '
            'Return JSON only, without markdown. Allowed values: '+json.dumps(schema))
        reply=self.request('/chat/completions',dict(model=self.model,temperature=0,max_tokens=1000,
            messages=[dict(role='system',content=prompt),dict(role='user',content=instruction)]))
        content=reply['choices'][0]['message']['content'].strip()
        if content.startswith('```'):
            content='\n'.join(content.splitlines()[1:-1])
        plan=json.loads(content)
        self.last_response=dict(model=reply.get('model',self.model),usage=reply.get('usage'),plan=plan)
        return plan

if __name__=='__main__':
    import sys
    provider=APIPlannerProvider('https://api.llm.ustc.edu.cn/v1','',sys.stdin.readline().strip())
    print(json.dumps(provider.models()),flush=True)
