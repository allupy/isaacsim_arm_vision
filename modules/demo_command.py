"""Small, explicit command vocabulary for the grasp demonstration."""
import re

TARGETS = {
    'red cube': ('红色方块', '红块', 'red cube', 'red block'),
    'green cube': ('绿色方块', '绿块', 'green cube', 'green block'),
    'blue cube': ('蓝色方块', '蓝块', 'blue cube', 'blue block'),
    'cup': ('杯子', '水杯', 'cup', 'mug'),
    'banana': ('香蕉', 'banana'),
    'apple': ('苹果', 'apple'),
}


def parse_command(text):
    lowered = text.lower()
    matches = [name for name, aliases in TARGETS.items()
               if any((alias in lowered if not alias.isascii() else
                       re.search(r'\b' + re.escape(alias) + r'\b', lowered)) for alias in aliases)]
    if len(matches) != 1:
        raise ValueError('Demo commands must name exactly one supported target: ' + str(matches))
    target = matches[0]
    return dict(instruction=text, target=target, destination='marked placement zone',
                skill='pick_and_place', parser='explicit demo vocabulary',
                display=f'Pick the {target}; place in the marked zone')
