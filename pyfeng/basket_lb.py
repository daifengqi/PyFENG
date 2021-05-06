import scipy.stats as spst
import numpy as np
import warnings
from . import opt_abc as opt
from . import multiasset as ma


class BsmBasket1BmTest(opt.OptABC):
    """
    Multiasset BSM model for pricing basket/Spread options when all asset prices are driven by a single Brownian motion (BM).
    """

    def __init__(self, sigma, weight=None, intr=0.0, divr=0.0, is_fwd=False):
        """
        Args:
            sigma: model volatilities of `n_asset` assets. (n_asset, )
            weight: asset weights, If None, equally weighted as 1/n_asset
                If scalar, equal weights of the value
                If 1-D array, uses as it is. (n_asset, )
            intr: interest rate (domestic interest rate)
            divr: vector of dividend/convenience yield (foreign interest rate) 0-D or (n_asset, ) array
            is_fwd: if True, treat `spot` as forward price. False by default.
        """

        sigma = np.atleast_1d(sigma)
        self.n_asset = len(sigma)
        if weight is None:
            self.weight = np.ones(self.n_asset) / self.n_asset
        elif np.isscalar(weight):
            self.weight = np.ones(self.n_asset) * weight
        else:
            assert len(weight) == self.n_asset
            self.weight = np.array(weight)
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd)

    @staticmethod
    def root(fac, std, strike):
        """
        Calculate the root x of f(x) = sum(fac * exp(std*x)) - strike = 0 using Newton's method
        Each fac and std should have the same signs so that f(x) is a monotonically increasing function.
        fac: factor to the exponents. (n_asset, ) or (n_strike, n_asset). Asset takes the last dimension.
        std: total standard variance. (n_asset, )
        strike: strike prices. scalar or (n_asset, )
        """

        assert np.all(fac * std >= 0.0)
        log = np.min(fac) > 0  # Basket if log=True, spread if otherwise.
        scalar_output = np.isscalar(np.sum(fac * std, axis=-1) - strike)
        strike = np.atleast_1d(strike)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_k = np.where(strike > 0, np.log(strike), 1)

            # Initial guess with linearlized assmption
            x = (strike - np.sum(fac, axis=-1)) / np.sum(fac * std, axis=-1)
            if log:
                np.fmin(x, np.amin(np.log(strike[:, None] / fac) / std, axis=-1), out=x)
            else:
                np.clip(x, -3, 3, out=x)

        # Test x=-9 and 9 for min/max values.
        y_max = np.exp(9 * std)
        y_min = np.sum(fac / y_max, axis=-1) - strike
        y_max = np.sum(fac * y_max, axis=-1) - strike

        x[y_min >= 0] = -np.inf
        x[y_max <= 0] = np.inf
        ind = ~((y_min >= 0) | (y_max <= 0))

        if np.all(~ind):
            return x[0] if scalar_output else x

        for k in range(32):
            y_vec = fac * np.exp(std * x[ind, None])
            y = np.log(np.sum(y_vec, axis=-1)) - log_k[ind] if log else np.sum(y_vec, axis=-1) - strike[ind]
            dy = np.sum(std * y_vec, axis=-1) / np.sum(y_vec, axis=-1) if log else np.sum(std * y_vec, axis=-1)
            x[ind] -= y / dy
            if len(y) == 0:
                print(ind, y_vec, y)
            y_err_max = np.amax(np.abs(y))
            if y_err_max < BsmBasket1BmTest.IMPVOL_TOL:
                break

        if y_err_max > BsmBasket1BmTest.IMPVOL_TOL:
            warn_msg = f'root did not converge within {k} iterations: max error = {y_err_max}'
            warnings.warn(warn_msg, Warning)

        return x[0] if scalar_output else x

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        assert fwd.shape[-1] == self.n_asset

        fwd_basket = fwd * self.weight
        sigma_std = self.sigma * np.sqrt(texp)
        cp = np.array(cp)
        d2 = -cp * self.root(fwd_basket * np.exp(-sigma_std**2/2), sigma_std, strike)

        if np.isscalar(d2):
            d1 = d2 + cp*sigma_std
        else:
            d1 = d2[:, None] + np.atleast_1d(cp)[:, None]*sigma_std

        price = np.sum(fwd_basket*spst.norm.cdf(d1), axis=-1)
        price -= strike * spst.norm.cdf(d2)
        price *= cp*df
        return price


class BsmBasketLowerBound(ma.NormBasket):

    def V_func(self, fwd):
        """
        Generate factor matrix V

        Args:
            fwd: forward rates of basket

        Returns:
            V matrix
        """
        gg = self.weight * fwd
        gg /= np.linalg.norm(gg)

        # equation 22，generate Q_1 and V_1
        Q1 = self.chol_m.T @ gg / np.sqrt(gg @ self.cov_m @ gg)  # in py, this is a row vector
        V1 = self.chol_m @ Q1
        V1 = V1[:, None]

        # obtain full V
        e1 = np.zeros_like(self.sigma)
        e1[0] = 1
        v = (Q1 - e1) / np.linalg.norm(Q1 - e1)
        v = v[:, None]
        R = np.eye(self.n_asset) - 2 * v @ v.T

        # singular value decomposition
        U, D, Q = np.linalg.svd(self.chol_m @ R[:, 1:], full_matrices=False)
        V = np.hstack((V1, U @ np.diag(D)))

        return V

    def price(self, strike, spot, texp, cp=1):
        fwd, df, _ = self._fwd_factor(spot, texp)
        V = self.V_func(fwd)
        sigma1 = V[:, 0]
        m = BsmBasket1BmTest(sigma1, is_fwd=True)
        price = m.price(strike, fwd, texp)
        return price
