import pandas as pd
import numpy as np
from .harmonics import harmonic_basis

class PeriodicVAR:
    # ---------- 1. Settings (chosen by you, never changed by fit) ----------
    def __init__(self, p=5, n_harm=2, noise="monthly"):
        """Store hyperparameters only. No computation here."""
        self.p = p
        self.n_harm = n_harm
        self.noise = noise
    
    # ---------- 2. Fitting ----------
    def fit(self, data):
        """data: DataFrame, continuous daily DatetimeIndex.
        Sets the learned attributes (trailing underscore) and returns self."""
        lag_matrix, Y = self._lag_matrix(data,self.p)
        design_matrix = self._design(lag_matrix, self.n_harm)

        coefs = np.linalg.lstsq(design_matrix.values, Y.values, rcond=None)[0]
        self.coef_table_ = pd.DataFrame(coefs, index = design_matrix.columns, columns=Y.columns)

        resid = Y - design_matrix.values @ coefs
        self.resid_ = pd.DataFrame(data=resid, index=Y.index, columns=Y.columns)

        self.columns_ = list(Y.columns)
        self.k_ = len(self.columns_)
        self.nobs_ = len(Y)
        self.last_obs_ = data.iloc[-self.p:]

        self.regressors_ = list(lag_matrix.columns)
        self.y_ = Y

        self.sigma_ = {}
        self.chol_ = {}
        for m in range(1, 13):
            r = self.resid_[self.resid_.index.month == m]
            self.sigma_[m] = np.cov(r.values.T, bias=True)
            self.chol_[m] = np.linalg.cholesky(self.sigma_[m])

        return self


    # Private helpers used by fit (leading underscore = internal)
    def _lag_matrix(self, df, lag): 
        LAG_MATRIX = df.copy()
        cols_origin = df.columns
        for i in range(lag):
            delay = df.shift(i+1)
            col_names = delay.columns
            for col in col_names:
                LAG_MATRIX[f'{col}_lag_{i+1}'] = delay[f'{col}'].values
        LAG_MATRIX = LAG_MATRIX.iloc[lag:]
        Y = LAG_MATRIX[cols_origin]
        LAG_MATRIX = LAG_MATRIX.drop(columns=cols_origin)
        LAG_MATRIX.insert(0, "const", 1.0)

        return LAG_MATRIX, Y

    def _design(self, lag_matrix, n_harmonics): 

        basis = harmonic_basis(lag_matrix.index.dayofyear, n_harmonics=n_harmonics)

        cols = {}
        for col in lag_matrix.columns:
            for key, h in basis.items():
                cols[f"{col}_{key}"] = lag_matrix[col].values * h

        return pd.DataFrame(cols, index=lag_matrix.index)
    
    # ---------- 3. Inspecting the fitted model ----------
    def coefs_at(self, doy):
        """Ordinary-VAR coefficients for one day of year:
        DataFrame with rows = regressors (const, lags) and columns = equations."""  
        basis = harmonic_basis(doy, n_harmonics=self.n_harm)
        regs = self.regressors_

        coefs_at_dict = {}

        for reg in regs:
            total = 0
            for h in basis:
                total = total + self.coef_table_.loc[f"{reg}_{h}"] * basis[h][0]

            coefs_at_dict[reg] = total

        return pd.DataFrame(coefs_at_dict).T

    def stability(self):
        """Max companion eigenvalue for each day of year (Series, index 1..366)."""

        k, p = self.k_, self.p
        max_roots = {}                                   # day -> value

        for d in range(1,367):
            table = self.coefs_at(d).drop("const")

            A_list = []
            for l in range(1, p+1):
                rows = [f"{c}_lag_{l}" for c in self.columns_]
                A_list.append(table.loc[rows].values.T)

            # Step 3: the companion matrix
            C = np.zeros((k * p, k * p))
            C[:k, :] = np.hstack(A_list)
            C[k:, :-k] = np.eye(k * (p - 1))

            # Step 4: largest |eigenvalue|
            max_roots[d] = np.abs(np.linalg.eigvals(C)).max()

        return pd.Series(max_roots, name="max_root")

    def loglike(self):
        k = self.k_
        months = self.resid_.index.month
        ll = 0.0
        for m in range(1, 13):
            n_m = (months == m).sum()
            logdet = np.linalg.slogdet(self.sigma_[m])[1]       # log of the determinant
            ll += -0.5 * n_m * (k * np.log(2 * np.pi) + logdet + k)
        return ll

    def bic(self):
        return -2 * self.loglike() + self.n_params() * np.log(self.nobs_)

    def r2(self, adjusted=True):
        """R² per equation (one-step-ahead, in-sample)."""
        ss_res = (self.resid_ ** 2).sum()
        ss_tot = ((self.y_ - self.y_.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        if adjusted:
            n, q = self.nobs_, len(self.coef_table_)    # q = regressors per equation
            r2 = 1 - (1 - r2) * (n - 1) / (n - q)
        return r2
    
    def summary(self): ...

    def n_params(self):
        n_cov = 12 * self.k_ * (self.k_ + 1) // 2
        return self.coef_table_.size + n_cov
    
    # ---------- 4. Using the fitted model ----------
    def forecast(self, history, steps):
        """Deterministic forecast, no noise."""

    def simulate(self, n_years, start="2001-01-01", seed=None, burn=365):
        """Stochastic simulation, returns a DataFrame with dates."""
        k, p = self.k_, self.p

        # 1. Dates: burn-in days before `start`, then n_years of output
        n_out = int(round(n_years * 365.25))
        first = pd.Timestamp(start) - pd.Timedelta(days=burn)
        dates = pd.date_range(first, periods=burn + n_out, freq="D")

        # 2. The lookup book: coefficients for each day of year (computed once)
        tables = {d: self.coefs_at(d).values for d in range(1, 367)}

        # 3. Random numbers
        rng = np.random.default_rng(seed)

        # 4. Storage: one row per day, plus p rows at the top for the starting values
        z = np.zeros((p + len(dates), k))
        z[:p] = self.last_obs_.values

        # 5. Walk through the simulated days in order
        for s, date in enumerate(dates):
            t = p + s                                   # row in z for this day

            # the past p simulated days (+ the 1 for the intercept)
            lag_row = np.concatenate([[1.0]] + [z[t - l] for l in range(1, p + 1)])

            # predictable part: look up today's coefficients by day of year
            deterministic = lag_row @ tables[date.dayofyear]

            # random kick, with this month's size and correlations
            noise = self.chol_[date.month] @ rng.standard_normal(k)

            # today's value, stored so tomorrow can use it as "yesterday"
            z[t] = deterministic + noise

        # 6. Drop the starting rows and the burn-in, attach the dates
        return pd.DataFrame(z[p + burn:], index=dates[burn:], columns=self.columns_)