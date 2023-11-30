from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pymc as pm

from pymc_experimental.model_builder import ModelBuilder


class LinearModel(ModelBuilder):
    def __init__(
        self,
        model_config: Dict = None,
        sampler_config: Dict = None,
        fit_method: str = "mcmc",
        nsamples: int = 100,
    ):
        self.nsamples = nsamples
        super().__init__(model_config, sampler_config, fit_method)

    """
    This class is an implementation of a linear regression model in PYMC.

    The regression model is as follows:

    y = a X + b + ε

    The parameters to be estimated are the slope a, the intercept b, and
    the observation error ε.

    The slope $a$ is a vector of length matching the number of columns of X.
    The prior for the slope is a normal distribution with a given location and scale.

    The intercept b is an optional scalar value. The prior for the intercept is
    a normal distribution with a given location and scale. If the prior for the
    intercept is set to `None` instead, then the value of the intercept is fixed at 0
    (i.e. the intercept is not included in the model).

    The observation error ε is a scalar value. The prior for the observation error is
    a normal distribution with a given location and scale.
    """

    _model_type = "LinearModel"
    version = "0.2"

    @staticmethod
    def get_default_model_config():
        return {
            "intercept": {"loc": 0, "scale": 10},
            "slope": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @staticmethod
    def get_default_sampler_config():
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

    @property
    def _serializable_model_config(self) -> Dict:
        return self.model_config

    @property
    def output_var(self):
        return "y_hat"

    def build_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Build the PyMC model.

        Returns
        -------
        None

        Examples
        --------
        >>> self.build_model()
        >>> assert self.model is not None
        >>> assert isinstance(self.model, pm.Model)
        >>> assert "intercept" in self.model.named_vars
        >>> assert "slope" in self.model.named_vars
        >>> assert "σ_model_fmc" in self.model.named_vars
        >>> assert "y_model" in self.model.named_vars
        >>> assert "y_hat" in self.model.named_vars
        >>> assert self.output_var == "y_hat"
        """
        cfg = self.model_config

        n_input_vars = X.shape[1]

        # Data array size can change but number of dimensions must stay the same.
        with pm.Model() as self.model:
            x = pm.MutableData("x", np.zeros((1, 1)), dims=["observation", "input_var"])
            y_data = pm.MutableData("y_data", np.zeros((1,)), dims="observation")

            # priors
            if cfg["intercept"] is None:
                intercept = 0
            else:
                intercept = pm.Normal(
                    "intercept", cfg["intercept"]["loc"], sigma=cfg["intercept"]["scale"]
                )
            slope = pm.Normal(
                "slope",
                cfg["slope"]["loc"],
                sigma=cfg["slope"]["scale"],
                dims="input_var",
                shape=n_input_vars,
            )
            obs_error = pm.HalfNormal("σ_model_fmc", cfg["obs_error"])

            # Model
            y_model = pm.Deterministic("y_model", intercept + slope @ x.T, dims="observation")

            # observed data
            y_hat = pm.Normal(
                "y_hat",
                y_model,
                sigma=obs_error,
                observed=y_data,
                dims="observation",
            )

        self._data_setter(X, y)

    def _data_setter(self, X: pd.DataFrame, y: Optional[Union[pd.DataFrame, pd.Series]] = None):
        with self.model:
            pm.set_data({"x": X}, coords={"observation": X.index})
            if y is not None:
                pm.set_data({"y_data": y.squeeze()}, coords={"observation": y.index})
            else:
                pm.set_data({"y_data": np.full(len(X), np.nan)}, coords={"observation": X.index})

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: pd.Series
    ) -> None:
        """
        Generate model data for linear regression.

        Parameters
        ----------
        nsamples : int, optional
            The number of samples to generate. Default is 100.
        data : np.ndarray, optional
            An optional data array to add noise to.

        Returns
        -------
        tuple
            A tuple of two np.ndarrays representing the feature matrix and target vector, respectively.

        Examples
        --------
        >>> import numpy as np
        >>> x, y = cls.generate_model_data()
        >>> assert isinstance(x, np.ndarray)
        >>> assert isinstance(y, np.ndarray)
        >>> assert x.shape == (100, 1)
        >>> assert y.shape == (100,)
        """
        # Ensure output Series is named appropriately
        y = y.rename(self.output_var)
        self.X, self.y = X, y
