"""Runs inside Docker. Tests leaders against real engine."""
import gc, numpy as np

class Leader:
    _subclass_registry = {}
    def __init__(self, name, engine):
        self.name = name; self.engine = engine
    @classmethod
    def cleanup_old_subclasses(cls):
        for sub in list(cls.__subclasses__()):
            if sub.__name__ in cls._subclass_registry:
                del cls._subclass_registry[sub.__name__]
        gc.collect()
    @classmethod
    def update_subclass_registry(cls):
        cls.cleanup_old_subclasses()
        cls._subclass_registry = {s.__name__: s for s in cls.__subclasses__()}
    def new_price(self, date): pass
    def start_simulation(self): pass
    def end_simulation(self): pass
    def get_price_from_date(self, date):
        return self.engine.exposed_get_price(date)

exec(open('/app/leaders_code.py').read())
try:
    exec(open('/app/benchmarks.py').read())
except FileNotFoundError:
    pass

from engine import Engine
import constants as cnst
Leader.update_subclass_registry()
leaders = {s.__name__: s for s in Leader.__subclasses__()}
print(f"Leaders: {list(leaders.keys())}")
for mk in ['MK1', 'MK2', 'MK3']:
    for nm, cls in leaders.items():
        e = Engine(); e.connect(Leader, nm, mk)
        e.leader = cls(nm, e)
        r = e.main_loop(101, 130, mode=cnst.Mode.TEST)
        p = sum((x[1]-x[3])*(100-5*x[1]+3*x[2]) for x in r)
        uLs = [x[1] for x in r]
        print(f"  {nm:20s} vs {mk}: {p:>10.0f}  uL=[{min(uLs):.1f},{max(uLs):.1f}]")
