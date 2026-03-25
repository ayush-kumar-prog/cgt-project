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

    def _rls_update(self, uL, uF):
        x = np.array([1.0, uL])
        Px = self.P @ x
        gain = Px / (self.lam + x @ Px)
        err = uF - (self.alpha + self.beta * uL)
        theta = np.array([self.alpha, self.beta])
        theta += gain * err
        self.alpha, self.beta = theta[0], theta[1]
        self.P = (self.P - np.outer(gain, x @ self.P)) / self.lam
        self.all_uL.append(uL)
        self.all_uF.append(uF)

    def new_price(self, date):
        if date > 101:
            prev_uL, prev_uF = self.get_price_from_date(date - 1)
            self._rls_update(prev_uL, prev_uF)
        if date == 101:
            return min(5.0, self.UPPER_BOUND)
        if date == 102:
            prev_uL, prev_uF = self.get_price_from_date(101)
            self._rls_update(prev_uL, prev_uF)
            hist_mean = np.mean(self.hist_uF)
            slope_est = (prev_uF - hist_mean) / (5.0 - np.mean(self.hist_uL))
            if slope_est > 0.3:
                return min(18.0, self.UPPER_BOUND)
            self._fit_ols()
        return self._optimal_price()
