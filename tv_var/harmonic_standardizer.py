import pandas as pd
import numpy as np

from .harmonics import harmonic_basis

class HarmonicStandardizer:
    def __init__(self, n_harmonics=3):
        """Store the setting only."""
        self.n_harmonics = n_harmonics

    def _design(self, dates):
        basis = harmonic_basis(dates.dayofyear, n_harmonics=self.n_harmonics)
        return pd.DataFrame(basis, index=dates)

    def fit(self, df):
        """Learn the harmonic coefficients for the seasonal mean and std. Return self."""
        X = self._design(df.index) 
        beta_mu = np.linalg.lstsq(X.values, df.values, rcond=None)[0]
        self.beta_mu_ = pd.DataFrame(beta_mu, index=X.columns, columns=df.columns)

        fitted_mean = X.values @ beta_mu
        anomaly = df.values - fitted_mean
        beta_var =  np.linalg.lstsq(X.values, anomaly**2, rcond=None)[0]
        self.beta_var_ = pd.DataFrame(beta_var, index=X.columns, columns=df.columns)

        self.columns_  = list(df.columns)

        return self
    
    def mu_sd(self, dates):
        """Seasonal mean and std (arrays) for given dates"""
        X = self._design(dates)[self.beta_mu_.index]  

        fitted_mean     = X.values @ self.beta_mu_.values
        fitted_variance = X.values @ self.beta_var_.values

        if np.min(fitted_variance) <= 0:
            raise ValueError("Fitted variance is not positive for some dates.")
        
        std = np.sqrt(fitted_variance)

        return fitted_mean, std

    def transform(self, df):
        """pcs_df -> z_df."""
        mu, sd = self.mu_sd(df.index)
        return pd.DataFrame((df.values - mu) / sd, index=df.index, columns=self.columns_)
    
    def inverse_transform(self, z_df):
        """z_df -> pcs_df (same index and column names)."""
        mu, sd = self.mu_sd(z_df.index)
        return pd.DataFrame((z_df.values * sd) + mu, index=z_df.index, columns=self.columns_)