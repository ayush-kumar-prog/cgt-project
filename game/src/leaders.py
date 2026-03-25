"""Stackelberg pricing game leaders."""
import numpy as np


class AdaptiveLeader(Leader):
    """Adaptive leader with exploration + RLS."""

    UPPER_BOUND = float('inf')

    def __init__(self, name, engine):
        super().__init__(name, engine)
        self.hist_uL = []
        self.hist_uF = []
        self.alpha = 0.0
        self.beta = 0.0
        self.sigma2 = 1.0
        self.P = np.eye(2) * 100.0
        self.lam = 0.97
        self.all_uL = []
        self.all_uF = []

    def start_simulation(self):
        for t in range(1, 101):
            uL, uF = self.get_price_from_date(t)
            self.hist_uL.append(uL)
            self.hist_uF.append(uF)
        self.all_uL = list(self.hist_uL)
        self.all_uF = list(self.hist_uF)
        self._fit_ols()

    def _fit_ols(self):
        uL = np.array(self.all_uL)
        uF = np.array(self.all_uF)
        X = np.column_stack([np.ones_like(uL), uL])
        theta = np.linalg.lstsq(X, uF, rcond=None)[0]
        self.alpha, self.beta = theta[0], theta[1]
        res = uF - X @ theta
        self.sigma2 = max(np.var(res), 1e-6)
        self.P = np.linalg.inv(X.T @ X / self.sigma2 + np.eye(2) * 0.001)

    def _optimal_price(self):
        a, b = self.alpha, self.beta
        denom = 10 - 6 * b
        if denom < 0.5:
            uL = 50.0
        else:
            uL = (105 + 3 * a - 3 * b) / denom
        uL = max(1.01, min(uL, self.UPPER_BOUND))
        uF_pred = a + b * uL
        if 100 - 5 * uL + 3 * uF_pred < 5:
            uL = (95 + 3 * uF_pred) / 5
        return max(1.01, min(uL, self.UPPER_BOUND))
