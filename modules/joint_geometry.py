"""Choose the closest equivalent revolute angle without crossing model limits."""
import math

def nearest_equivalent(value, reference, lower, upper):
    first = math.ceil((lower-value)/(2*math.pi))
    last = math.floor((upper-value)/(2*math.pi))
    if first > last:
        raise ValueError('No equivalent angle inside joint limits')
    return min((value+2*math.pi*k for k in range(first,last+1)),key=lambda q:abs(q-reference))
