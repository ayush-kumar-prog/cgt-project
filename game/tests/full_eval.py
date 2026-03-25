"""Comprehensive evaluation suite."""
import gc, numpy as np
class Leader:
    _subclass_registry = {}
    def __init__(s, name, engine): s.name = name; s.engine = engine
    @classmethod
    def cleanup_old_subclasses(c):
        for s in list(c.__subclasses__()):
            if s.__name__ in c._subclass_registry: del c._subclass_registry[s.__name__]
        gc.collect()
    @classmethod
    def update_subclass_registry(c):
        c.cleanup_old_subclasses()
        c._subclass_registry = {s.__name__: s for s in c.__subclasses__()}
    def new_price(s, date): pass
    def start_simulation(s): pass
    def end_simulation(s): pass
    def get_price_from_date(s, date): return s.engine.exposed_get_price(date)

exec(open('/app/leaders_code.py').read())
from engine import Engine
import constants as cnst

def all_sub(cls):
    r = []
    for s in cls.__subclasses__():
        r.append(s); r.extend(all_sub(s))
    return r

def run(lcls, mk, mode):
    e = Engine(); e.connect(Leader, lcls.__name__, mk)
    e.leader = lcls(lcls.__name__, e)
    r = e.main_loop(101, 130, mode=mode)
    return r, sum((x[1]-x[3])*(100-5*x[1]+3*x[2]) for x in r)
