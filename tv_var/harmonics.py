import numpy as np

def harmonic_basis(dayofyear, n_harmonics):
    """Harmonic basis for day(s) of year: {'base', 'cos_1', 'sin_1', ..., 'cos_H', 'sin_H'}."""

    dayofyear = np.atleast_1d(np.asarray(dayofyear, dtype=float))
    dayofyear_frac = (dayofyear-1)/365.25

    ones = np.ones(len(dayofyear))

    dict_harmonics = {'base':ones}

    for i in range(n_harmonics):

        dict_harmonics[f'cos_{i+1}'] = np.cos(2 * np.pi * dayofyear_frac * (i+1))
        dict_harmonics[f'sin_{i+1}'] = np.sin(2 * np.pi * dayofyear_frac * (i+1))

    return dict_harmonics
