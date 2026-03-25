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

def run_mock(name, fn, noise, trend=0.0, ub=float('inf')):
    eng = MockEngine(fn, noise, trend=trend)
    cls = [s for s in Leader.__subclasses__() if s.__name__=='AdaptiveLeader'][0]
    ldr = cls(cls.__name__, eng); ldr.UPPER_BOUND = ub
    ldr.engine = eng; ldr.start_simulation()
    tot, opt = 0.0, 0.0
    for t in range(101, 131):
        uL = ldr.new_price(t)
        uF = fn(uL) + trend*t + eng.rng.normal(0, noise)
        eng.prices[t] = (uL, uF)
        tot += (uL-1)*(100-5*uL+3*uF)
        bp = max((u-1)*(100-5*u+3*(fn(u)+trend*t))
                 for u in np.arange(1, min(ub,50)+.5, .5))
        opt += bp
    pct = tot/opt*100 if opt > 0 else 0
    print(f'  {name:<35s} {tot:>9.0f}/{opt:>9.0f} = {pct:>5.1f}%')

print('=== STRESS TEST: MK4/5/6 APPROXIMATIONS ===')
# MK1-like variants (linear, different slopes)
run_mock('MK1 (baseline)',      lambda u: 2.21+0.74*u, 0.55)
run_mock('MK4 (steeper slope)', lambda u: 1.0+1.2*u,  0.55)
run_mock('MK4 (gentler slope)', lambda u: 3.0+0.3*u,  0.55)
run_mock('MK4 (high noise)',    lambda u: 2.21+0.74*u, 1.5)
run_mock('MK4 (low intercept)', lambda u: 0.5+0.74*u, 0.55)

print('\n=== MK2-LIKE (with time trends) ===')
run_mock('MK2 (baseline)',      lambda u: 1.69+0.80*u, 0.26, trend=0.029)
run_mock('MK5 (slower trend)',  lambda u: 1.69+0.80*u, 0.26, trend=0.01)
run_mock('MK5 (faster trend)',  lambda u: 1.69+0.80*u, 0.26, trend=0.06)
run_mock('MK5 (neg trend)',     lambda u: 5.0+0.80*u,  0.26, trend=-0.02)
run_mock('MK5 (diff slope)',    lambda u: 2.5+0.5*u,   0.30, trend=0.02)

print('\n=== MK3-LIKE (nonlinear, bounded [1,15]) ===')
run_mock('MK3 (baseline sqrt)', lambda u: 0.44+0.69*np.sqrt(u), 0.50, ub=15)
run_mock('MK6 (smaller sqrt)',  lambda u: 0.2+0.5*np.sqrt(u),   0.30, ub=15)
run_mock('MK6 (log)',           lambda u: 0.5+0.8*np.log(u),    0.50, ub=15)
run_mock('MK6 (linear low)',    lambda u: 1.0+0.1*u,            0.40, ub=15)

print('\n=== ADVERSARIAL EDGE CASES ===')
run_mock('Zero slope (constant)',  lambda u: 3.5,              0.30)
run_mock('Negative slope',         lambda u: 8.0-0.3*u,        0.50)
run_mock('Very high noise',        lambda u: 2.0+0.7*u,        2.0)
run_mock('Very steep slope',       lambda u: 0.5+2.0*u,        0.50)
