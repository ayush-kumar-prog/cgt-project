"""Stress test: mock followers with varied parameters."""
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

class MockEngine:
    def __init__(self, follower_fn, noise, seed=42, trend=0.0):
        self.fn = follower_fn
        self.noise = noise
        self.trend = trend
        self.rng = np.random.RandomState(seed)
        self.prices = {}
        for t in range(1, 101):
            uL = 1.72 + self.rng.random() * 0.18
            uF = self.fn(uL) + trend * t + self.rng.normal(0, noise)
            self.prices[t] = (uL, uF)

    def exposed_get_price(self, date):
        return self.prices.get(date, (0.0, 0.0))
