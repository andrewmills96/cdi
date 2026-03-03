import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Sequence, Union, Dict, Optional
import warnings
import scipy as sp

# Constants
DAYS_PER_YEAR = 365.0
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]
DEFAULT_LABEL = "Def"


# Helpers
def calc_dt(dates: Union[pd.DatetimeIndex, Sequence[datetime]], val_date: datetime) -> np.ndarray:
    """Calculate year fractions between a sequence of dates from a valuation date"""

    # check dates are in the right format and calculate the number of days from val_date
    if isinstance(dates, pd.DatetimeIndex):
        delta_days = (dates - pd.Timestamp(val_date)).days
    else:
        dates = pd.to_datetime(dates)
        delta_days = (dates - pd.Timestamp(val_date)).days

    dt = np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR
    return dt

def fit_to_shape(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Trim and pad an array to shape with zeros."""
    if len(shape) == 3 and a.ndim == 2:
        a = np.expand_dims(a, axis=-1)
    trimmed = a[tuple(slice(0, min(a.shape[i], shape[i])) for i in range(len(shape)))]
    pad_widths = [(0, max(0, shape[i] - trimmed.shape[i])) for i in range(len(shape))]
    while len(pad_widths) < trimmed.ndim:
        pad_widths.append((0, 0))
    return np.pad(trimmed, pad_widths, mode='constant')

def total_returns(pvs: np.ndarray, cashflows: np.ndarray, fwds: np.ndarray, a0: float):
    """Calculates the annual total return for a portfolio of bonds"""
    returns = np.empty((pvs.shape[0], pvs.shape[1]), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:, 0] = (pvs[:, 0] + cashflows[:, 0]) / a0 - 1
        returns[:, 1:] = (pvs[:, 1:] + cashflows[:, 1:]) / pvs[:, :-1] - 1

    returns = np.where(np.isfinite(returns), returns, fwds)

    if returns.shape != (pvs.shape[0], pvs.shape[1]):
        raise ValueError(f"Unexpected returns shape {returns.shape}")
    if np.isnan(returns).any():
        raise ValueError("NaN values in returns array")

    return returns

def map_spreads(ratings: np.ndarray, spread_map: Dict[str, float]) -> np.ndarray:
    """
    Vectorized mapping of rating label arrays to spread floats.
    Uses pd.factorize to avoid per-element Python lookups.
    ratings: any shape ndarray of rating strings.
    Returns: same shape ndarray of floats.
    """
    flat = ratings.ravel()
    codes, uniques = pd.factorize(flat, sort=False)
    mapped = np.fromiter(
        (spread_map.get(u, 0.0) for u in uniques),
        dtype=float,
        count=len(uniques)
    )
    return mapped[codes].reshape(ratings.shape)

def compute_maturity_flags(cashflows: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask (n_years x n_bonds) that is True for years
    up to and including each bond's final non-zero cashflow year.

    Uses np.flip() rather than [::-1] slicing to ensure a contiguous array
    is passed to np.argmax, avoiding stride-related issues on some numpy builds.
    """
    n_years, n_bonds = cashflows.shape
    # Flip along axis=0 (time axis) into a fresh contiguous array, then find
    # the first non-zero row — that corresponds to the last non-zero in the original.
    flipped      = np.flip(cashflows, axis=0)                                # contiguous copy
    has_cashflow = cashflows.any(axis=0)                                     # (n_bonds,)
    last_cf_idx  = np.where(
        has_cashflow,
        n_years - 1 - np.argmax(flipped != 0, axis=0),
        -1
    )                                                                        # (n_bonds,)
    year_indices = np.arange(n_years)[:, np.newaxis]                         # (n_years, 1)
    return year_indices <= last_cf_idx[np.newaxis, :]                        # (n_years, n_bonds)


# Rates
class Rates:
    """Class to hold a yield curve."""

    def __init__(self, yields: np.ndarray, dates: List):
        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")
        self.yields = pd.Series(np.asarray(yields, dtype=float), index=pd.DatetimeIndex(dates))

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Linearly interpolate yields to specified dates."""
        combined = self.yields.index.union(dates).sort_values()
        return self.yields.reindex(combined).interpolate(method="time").reindex(dates)

    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Linearly interpolate and then calculate the implied forward rates between each date in dates
        """
        # Interpolate
        y = self.interpolate(dates)

        # Calculate forward rates
<<<<<<< HEAD
        t = calc_dt(pd.DatetimeIndex(y.index), val_date)
        acc = np.power(1.0 + np.asarray(y.values, dtype=float), t)
=======
        t = calc_dt(y.index, val_date)
        acc = np.power(1.0 + y.values, t)
>>>>>>> b915299a24f972d1aee59a47871cad61ad35f44d
        dt = np.diff(t)
        fwds = np.empty_like(y.values)
        fwds[0] = y.iloc[0]
        fwds[1:] = (acc[1:] / acc[:-1]) ** (1.0 / dt) - 1.0
        return pd.Series(fwds, index=y.index, name="fwds")


# Risk Model
class TransitionMatrix:
    """
    Class for holding a credit rating transition matrix.
    Matrix must be defined as a square matrix with labels.
    """

    def __init__(self, tmatrix: np.ndarray, labels: list[str]):

        # Check Shape of ratings and labels match
        if tmatrix.ndim != 2 or tmatrix.shape[0] != tmatrix.shape[1]:
            raise IndexError("Transition matrix must be square and 2-dimensional")
        if len(labels) != tmatrix.shape[0]:
            raise IndexError("Labels length must match matrix dimension")

        # Get Transition Matrix Details
        self.tmatrix = tmatrix
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)  # cumulative probabilities for np.searchsorted

        # Labels
        self.labels = np.array(labels)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

        # Check transitions sum to 1
        bad_rows = np.where(~np.isclose(self.cum_tmatrix[:, -1], 1.0))[0]
        if bad_rows.size:
            raise ValueError(f"Transition probabilities for {self.labels[bad_rows]} do not sum to 1")

    def indices_to_labels(self, indices: np.ndarray) -> np.ndarray:
        return self.labels[indices]

    def transitions(self, pX: np.ndarray, ratings_map: pd.Series) -> np.ndarray:
        """
        Vectorised ratings migration.

        Parameters
        ----------
        pX : (n_sim, n_years, n_issuers) CDF(N(0,1)) draws
        ratings_map : Series of initial ratings labels (length n_issuers)

        Returns
        -------
        transitions : (n_sim, n_years+1, n_issuers) ratings label array
        """
        n_sim, n_years, n_issuers = pX.shape
        initial = np.array([self.label_to_idx[r] for r in ratings_map])       # (n_issuers,)

        # Result array (indices): include t=0 initial ratings
        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial[np.newaxis, :]

        # Calculate transitions for each year
        for t in range(n_years):
            # current ratings
            cur = res[:, t, :]                                                 # (n_sim, n_issuers)

            # cumulative tmatrix for current ratings
            cum = self.cum_tmatrix[cur]                                        # (n_sim, n_issuers, n_states)

            # probabilities at t
            pX_t = pX[:, t, :, np.newaxis]                                      # (n_sim, n_issuers, 1)

            # check where pX_t places on cum tmatrix to identify transition
            # clip guards against pX == 1.0 producing an out-of-bounds index
            res[:, t + 1, :] = np.clip(
                np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1
            ).astype(np.int32)                                                     # (n_sim, n_issuers)

        # convert back to labels
        return self.indices_to_labels(res)                                     # (n_sim, n_years+1, n_issuers)

    def fundamental_matrix(self):
        """
        Calculate the fundamental matrix for a transition matrix,
        which shows the expected number of visits to each state before default.

        The transition matrix (P) can be broken down into a matrix of transient states (Q),
        a matrix of default probabilities (R) and an absorbing state matrix (D):
        P = [Q  R]
            [O  D]
        """

        P = self.tmatrix

        # Separate non-absorbing states (Q)
        Q = P[:-1, :-1]

        # Fundamental Matrix (Expected time in transient states before default)
        # N = (I - Q)^-1
        N = np.linalg.inv((np.eye(len(Q)) - Q))

        labels = self.labels[:-1]

        return pd.DataFrame(N, index = labels, columns = labels)

    def time_to_default(self):
        """
        Calculate the expected time to default for each rating.
        Expected time to default = fundamental_matrix @ 1
        """

        # Get fundamental matrix
        N = self.fundamental_matrix().values

        # Expected time to default: t = N @ 1
        t = N @ np.ones(len(N))

        return pd.Series(t, index=self.labels[:-1], name = 'Time to Default')

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt = ".1%"))


class SimulationResult:
    """Container for CreditRiskModel outputs."""
    def __init__(self, E, S, I, X, pX, transitions, n_sim, n_years):
        self.E = E
        self.S = S
        self.I = I
        self.X = X
        self.pX = pX
        self.transitions = transitions
        self.n_sim = n_sim
        self.n_years = n_years


class CreditRiskModel:
    """
    Monte Carlo factor-based Credit Risk Model based on the BRS Credit VaR Model.
    """

    def __init__(
            self,
            transition_matrix: TransitionMatrix,
            rho_e: float,
            rho_s: np.ndarray,
            issuer_ids: list,
            sector_map: pd.Series,
            ratings_map: pd.Series
    ):
        self.transition_matrix = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        self.sector_map = sector_map
        self.ratings_map = ratings_map
        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validate consistency of inputs for the credit risk model."""

        # --- Type checks ---
        if not isinstance(self.sector_map, pd.Series):
            raise TypeError("sector_map must be a pandas Series.")

        if not isinstance(self.ratings_map, pd.Series):
            raise TypeError("ratings_map must be a pandas Series.")

        if not isinstance(self.rho_s, np.ndarray):
            raise TypeError("rho_s must be a numpy array.")

        if not np.isscalar(self.rho_e):
            raise TypeError("rho_e must be a scalar.")

        # --- Sector correlation consistency ---
        n_issuer_sectors = self.sector_map.nunique()
        n_sectors = len(self.rho_s)

        if n_sectors > n_issuer_sectors:
            warnings.warn(
                f"{n_sectors} sector correlations are defined but only "
                f"{n_issuer_sectors} sectors exist in sector_map. "
                f"Only the first {n_issuer_sectors} will be used.",
                UserWarning
            )

        if np.any(self.rho_e > self.rho_s):
            raise ValueError("rho_s must be greater than rho_e")

    def run(self,  n_sim: int, n_years: int):

        n_issuers = len(self.ratings_map)
        n_issuer_sectors = self.sector_map.nunique()

        # Random draws
        E = np.random.normal(size=(n_sim, n_years))
        S = np.random.normal(size=(n_sim, n_years, n_issuer_sectors))
        I = np.random.normal(size=(n_sim, n_years, n_issuers))

        rho_s_i = self.rho_s[self.sector_map]
        X = np.sqrt(self.rho_e) * E[:, :, np.newaxis] + \
            np.sqrt(rho_s_i - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map] +\
            np.sqrt(1 - rho_s_i)[np.newaxis, np.newaxis, :] * I

        pX = sp.stats.norm.cdf(X)

        # Calculate ratings transitions
        transitions = self.transition_matrix.transitions(pX, self.ratings_map)

        # format transitions into a dataframe to retain issuer id info.
        transitions = pd.DataFrame(
            transitions.reshape(n_sim * (n_years+1), n_issuers),
            index=pd.MultiIndex.from_product(
                [range(n_sim), range(n_years + 1)],
                names=["sim", "year"]
            ),
            columns=self.issuer_ids
        )

        return SimulationResult(E, S, I, X, pX, transitions, n_sim, n_years)


# Assets
class Issuers:
    """Container class for bond issuer details."""
    def __init__(
        self,
        ids: List[str],
        ratings: pd.Series,
        sectors: pd.Series,
        names: Optional[pd.Series] = None
    ):
        self.ids = ids
        self.ratings = ratings
        self.sectors = sectors
        self.names = names


class BondSimulationResult:
    """Output of Bonds.run_sim()."""
    def __init__(self, transitions, received_cashflows, recovery_payments,
                 total_cashflows, pvs, dates, bond_ids):
        self.transitions = transitions
        self.received_cashflows = received_cashflows
        self.recovery_payments = recovery_payments
        self.total_cashflows = total_cashflows
        self.pvs = pvs
        self.dates = dates
        self.bond_ids = bond_ids

    def to_dataframe(self, by_bond: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(
            {'rating': self.transitions.reshape(-1),
             'cashflow': self.total_cashflows.reshape(-1),
             'pv': self.pvs.reshape(-1)},
            index=pd.MultiIndex.from_product(
                [range(self.total_cashflows.shape[0]), self.dates, self.bond_ids],
                names=["scenario", "date", "bond_id"],
            ),
        ).reset_index()
        if not by_bond:
            return df.groupby(['scenario', 'date']).sum().drop(columns='bond_id').reset_index()
        return df


class Bonds:

    """Bond universe: cashflows, issuer links, and simulation."""

    def __init__(
        self,
        ids: list,
        issuer_ids: pd.Series,
        recoveries: pd.Series,
        cashflows: pd.DataFrame,
        issuers: Issuers,
        descriptions: pd.Series | None = None
    ):
        self.ids = ids
        self.issuer_ids = issuer_ids
        self.recoveries = recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        self.descriptions = descriptions
        self.validate_inputs()

    def validate_inputs(self) -> None:
        """Validate internal consistency of bond inputs."""

        # --- ids ---
        if not isinstance(self.ids, (list, tuple)):
            raise TypeError("ids must be a list or tuple.")

        if len(self.ids) == 0:
            raise ValueError("ids cannot be empty.")

        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique.")

        # --- Series alignment ---
        for name, s in {
            "issuer_ids": self.issuer_ids,
            "recoveries": self.recoveries,
        }.items():
            if not isinstance(s, pd.Series):
                raise TypeError(f"{name} must be a pandas Series.")

            if not s.index.isin(self.ids).all():
                raise ValueError(f"{name} index must match bond ids.")

        # --- Recoveries ---
        if ((self.recoveries < 0) | (self.recoveries > 1)).any():
            raise ValueError("recoveries must be between 0 and 1.")

        # --- Cashflows ---
        cf = self.cashflows

        if not isinstance(cf, pd.DataFrame):
            raise TypeError("cashflows must be a pandas DataFrame.")

        # Ensure DatetimeIndex (or convertible date column)
        if not isinstance(cf.index, pd.DatetimeIndex):
            date_col = next((c for c in cf.columns if c.lower() == "date"), None)
            if date_col is None:
                raise ValueError("cashflows must have a DatetimeIndex or a 'date' column.")
            cf = cf.set_index(pd.to_datetime(cf[date_col], errors="raise")).drop(columns=date_col)

        # Ensure cashflow columns contain all bond ids
        missing_cf = set(self.ids) - set(cf.columns)
        if missing_cf:
            raise ValueError(f"cashflows is missing columns for bond ids: {missing_cf}")

        # Ensure numeric cashflows
        if not cf.apply(pd.api.types.is_numeric_dtype).all():
            raise TypeError("all cashflow columns must be numeric.")

        # Sort index and match columns to bond id order.
        self.cashflows = cf[self.ids].sort_index()

        # --- Issuers ---
        missing_issuers = set(self.issuer_ids.values) - set(self.issuers.ids)
        if missing_issuers:
            raise ValueError(f"issuer_ids missing in issuers object: {missing_issuers}")

    def year_end_cashflows(self, val_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Aggregates cashflows to year-end dates.
        Moves intra-month dates to month-end first (bonds often pay on the 1st),
        then resamples to calendar year-end.
        """
        cf = self.cashflows.resample("ME").sum()
        if val_date is not None:
            cf = cf.loc[cf.index > val_date]
        return cf.resample("YE").sum()

    def pv(self, val_date: pd.Timestamp, rates: Rates, spread_map: Dict[str, float]) -> pd.Series:
        """Present value of each bond at val_date using current market rates and spreads."""

        cfs = self.year_end_cashflows(val_date)
        dt = calc_dt(pd.DatetimeIndex(cfs.index), val_date)

        base_yields = np.asarray(rates.interpolate(pd.DatetimeIndex(cfs.index)).values, dtype=float)                # (T,)
        current_ratings = self.issuer_ids.map(self.issuers.ratings).reindex(self.ids)
        spreads = map_spreads(np.asarray(current_ratings.values)[np.newaxis, :], spread_map).squeeze(0)  # (n_bonds,)

        # Discount factors: (T, n_bonds)
        dfs = (1 + base_yields[:, np.newaxis] + spreads[np.newaxis, :]) ** (-dt[:, np.newaxis])
        prices = (cfs.values * dfs).sum(axis=0)

        return pd.Series(prices, index=self.ids)

    def _run_sim_pv(
            self,
            cashflows: np.ndarray,
            bond_transitions: np.ndarray,
            spread_map: dict,
            yields: np.ndarray,
            dt: np.ndarray
        ) -> np.ndarray:

        # Shape
        n_years_cf, n_bonds = cashflows.shape
        n_sim, n_years_sim, n_bonds = bond_transitions.shape

        # Take min n_years to return output for
        n_years = min(n_years_cf, n_years_sim)

        # Check that cashflows and transitions contain the same number of bonds
        assert cashflows.shape[1] == bond_transitions.shape[2], "n_bonds inconsistent"

        # Since all spreads come from a finite set of rating labels,
        # we can precompute a PV table of shape (n_unique_spreads, n_years, n_bonds)

        # Get unique spread values
        unique_spreads, spread_inverse = np.unique(map_spreads(bond_transitions, spread_map), return_inverse=True)
        n_unique = len(unique_spreads)

        # Create a matrix of dts for each year in the simulation
        dtime_mat = dt[:, np.newaxis] - dt[:n_years][np.newaxis, :]          # (n_years_cf, n_years)
        future = dtime_mat > 0                                               # cashflow is after val year

        # Create a matrix of (1 + yields)
        yields = np.asarray(yields, dtype=float)
        yield_mat = (1 + yields)[:, np.newaxis] * np.ones((1, n_years))   # (n_years_cf, n_years)

        # Build PV table: (n_unique, n_years, n_bonds)
        pv_table = np.zeros((n_unique, n_years, n_bonds))
        for si, s in enumerate(unique_spreads):
            # Discount factors: shape (n_years_cf, n_years)
            dfs_mat = np.zeros_like(dtime_mat)
            dfs_mat[future] = (yield_mat[future] + s) ** (-dtime_mat[future])

            # Sum over all cashflow dates for each valuation year
            pv_table[si] = dfs_mat.T @ cashflows                             # (n_years, n_bonds)

        # Map each (sim, t, bond) to its spread index then look up PV
        spread_idx = spread_inverse.reshape(n_sim, n_years, n_bonds)         # (n_sim, n_years, n_bonds)
        t_idx = np.arange(n_years)[np.newaxis, :, np.newaxis]
        b_idx = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        pvs = pv_table[spread_idx, t_idx, b_idx]                      # (n_sim, n_years, n_bonds)

        return pvs

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):

        """Project cashflows and revalue bonds across all scenarios."""

        # 1. Project cashflows

        # Raw cashflows and shape parameters
        cashflows_df = self.year_end_cashflows(val_date)
        cashflows = cashflows_df.values                                    # (n_years, n_bonds)
        dates = cashflows_df.index
        n_years_cf = cashflows.shape[0]
        n_bonds = cashflows.shape[1]
        n_sim = sim_results.n_sim
        n_years_sim = sim_results.n_years
        n_years = min(n_years_sim, n_years_cf)      # Simulation result will cover min(cashflow years, sim years)

        # Map issuer transitions to bonds, drop year-0 (initial ratings).
        bond_transitions = (
            sim_results.transitions
            .loc[sim_results.transitions.index.get_level_values("year") != 0, self.issuer_ids]
            .set_axis(self.ids, axis=1)
            .to_numpy()
            .reshape(n_sim, n_years_sim, n_bonds)
        )
        bond_transitions = bond_transitions[:, :n_years, :]                  # (n_sim, n_years, n_bonds)

        # --- Maturity flags: (n_years_cf, n_bonds) ---
        not_matured = compute_maturity_flags(cashflows)

        # --- Default flags ---
        is_defaulted  = (bond_transitions == DEFAULT_LABEL)                  # (n_sim, n_years, n_bonds)
        not_defaulted = ~is_defaulted

        # First default event per (sim, bond): True only in the first year of default
        # cumsum trick: cumsum==1 flags the transition year from non-default to default
        first_default = (np.cumsum(is_defaulted, axis=1) == 1) & is_defaulted  # (n_sim, n_years, n_bonds)

        # --- Cashflow arrays ---
        cf_slice = cashflows[np.newaxis, :n_years, :]                        # (1, n_years, n_bonds)

        # Received cashflows: only when bond is not in default
        received_cashflows = cf_slice * not_defaulted                        # (n_sim, n_years, n_bonds)

        # Recovery payments: face * recovery, paid in the first default year, only if not matured
        recoveries = np.asarray(self.recoveries.values, dtype=float)
        recovery_payments = (
            first_default
            * recoveries[np.newaxis, np.newaxis, :]
            * not_matured[np.newaxis, :n_years, :]
        )                                                                     # (n_sim, n_years, n_bonds)

        # Total Cashflows
        total_cashflows = received_cashflows + recovery_payments              # (n_sim, n_years, n_bonds)

        # 2. Calculate PVs

        yields = np.asarray(rates.interpolate(pd.DatetimeIndex(dates)).values, dtype=float)                              # (n_years_cf,)
        dt = calc_dt(pd.DatetimeIndex(dates), val_date)                                         # (n_years_cf,)
        pvs = self._run_sim_pv(cashflows, bond_transitions, spread_map, yields, dt)

        # Zero out PV for bonds that have defaulted (default is absorbing — PV = 0 from first default year onward)
        ever_defaulted = np.cumsum(is_defaulted, axis=1) > 0                 # (n_sim, n_years, n_bonds)
        pvs = pvs * ~ever_defaulted

        # Return
        return BondSimulationResult(
            transitions = bond_transitions,
            received_cashflows = received_cashflows,
            recovery_payments = recovery_payments,
            total_cashflows = total_cashflows,
            pvs = pvs,
            dates = dates[:n_years],
            bond_ids = self.ids
        )


# Liabilities
class Liabilities:
    """Container for liability cashflows."""

    def __init__(self, cashflows: pd.Series, dates: List):
        # Check dates and cashflows are the same length
        if len(cashflows) != len(dates):
            raise ValueError("Cashflows and dates length mismatch.")

        # Store cashflows as series
        self.cashflows = pd.Series(cashflows, index = pd.to_datetime(dates))

    def pv(self, val_date: datetime, rates: Union[Rates, float], shift=0.0, timeline = False, n_years: Optional[float] = None) -> Union[float, np.ndarray]:
        """
        Calculate PV of liabilities for a specified date and discount rate.
        """

        # Get cashflows beyond val date
        cashflows = self.cashflows.loc[self.cashflows.index > val_date]
        dates = cashflows.index
        dt = calc_dt(pd.DatetimeIndex(dates), val_date)

        if not isinstance(rates, (int, float)):
            # rates is a Rates object
            yields_int = rates.interpolate(pd.DatetimeIndex(dates))
            net_yield = np.asarray(yields_int.values, dtype=float) + shift
        else:
<<<<<<< HEAD
            # rates is a scalar
            net_yield = np.full(len(cashflows), float(rates) + shift, dtype=float)
=======
            yields_int = rates.interpolate(dates)
            net_yield = (yields_int + shift).values
>>>>>>> b915299a24f972d1aee59a47871cad61ad35f44d

        if timeline:
            cf_values = np.asarray(cashflows.values, dtype=float)
            pv = np.array([
                    np.sum(cf_values[i + 1:, np.newaxis] * ((1 + net_yield[i+1:, np.newaxis])**-(dt[i + 1:] - dt[i])))
                    for i in range(len(dates))
            ])
            pv = pv[:n_years] if n_years is not None else pv
        else:
            cf_values = np.asarray(cashflows.values, dtype=float)
            pv = (cf_values * (1 + net_yield) ** -dt).sum()

        return pv

    def __str__(self):
        return pd.DataFrame(self.cashflows).to_markdown(floatfmt='.0f')


# CDI
def allocate_bond_sim(bond_sim: BondSimulationResult, allocation: pd.Series, shape: tuple, target: str = 'cashflows'):
    """Calculate cashflows or pvs for a bond sim result based on a defined allocation."""

    # Ids and Notionals
    ids = allocation.index.to_list()
    allocations = allocation.values

    # map bond_id -> index in cashflows
    bond_idx = {bid: i for i, bid in enumerate(bond_sim.bond_ids)}

    # indices of the bonds you want (in the correct order)
    selected_idx = [bond_idx[id] for id in ids]

    # get cashflows or pvs
    if target == 'pvs':
        nominal = bond_sim.pvs[:, :, selected_idx]
    elif target == 'cashflows':
        nominal = bond_sim.total_cashflows[:, :, selected_idx]
    else:
        raise TypeError('Invalid Option Selected')

    net = nominal * allocations
    total = net.sum(axis=2)

    return fit_to_shape(total, shape)


class CDISimulationResult:
    """Output container for CDI sim."""
    def __init__(self, config, bond_sim, cdi_sim):
        self.config = config
        self.bond_sim = bond_sim
        self.cdi_sim = cdi_sim


class FoxConfig:
    """Config file for the Fox mandate"""
    def __init__(
            self,
            heubeck_liabilities,
            r_gaap,
            r_ifrs,
            cmbp_margin,
            asset_buffer,
            mortality_buffer,
            fee,
            performance_cap,
            additional_payment_year: int = 10,
            performance_payment_year: int = 25
        ):
        self.heubeck_liabilities=heubeck_liabilities
        self.r_gaap=r_gaap
        self.r_ifrs=r_ifrs
        self.cmbp_margin=cmbp_margin
        self.asset_buffer=asset_buffer
        self.mortality_buffer=mortality_buffer
        self.fee=fee
        self.performance_cap=performance_cap
        self.additional_payment_year=additional_payment_year
        self.performance_payment_year=performance_payment_year

    def to_dict(self):
        return {
            "heubeck_liabilities": self.heubeck_liabilities,
            "r_gaap":self.r_gaap,
            "r_ifrs":self.r_ifrs,
            "cmbp_margin":self.cmbp_margin,
            "asset_buffer":self.asset_buffer,
            "mortality_buffer":self.mortality_buffer,
            "fee":self.fee,
            "performance_cap":self.performance_cap,
            "additional_payment_year":self.additional_payment_year,
            "performance_payment_year":self.performance_payment_year
        }


class CDIMandate_Fox:
    """
    Class implementing the Fox CDI mandate.
    """
    def __init__(
        self,
        liabilities: Liabilities,
        cash: float,
        bonds: Bonds,
        cdi_allocation: pd.Series,
        cmbp_allocation: pd.Series,
        config: FoxConfig
    ):
        self.liabilities=liabilities
        self.cash=cash
        self.bonds=bonds
        self.cdi_allocation=cdi_allocation
        self.cmbp_allocation=cmbp_allocation
        self.config = config

    def run(self, val_date: datetime, rates: Rates, spread_map:  dict, sim_results: SimulationResult) -> CDISimulationResult:
        """
        Run the Fox CDI simulation.
        """
        # Simulation Shape
        max_years = sim_results.n_years
        n_sim = sim_results.n_sim

        ### Compute liability metrics. ###
        #  These are deterministic so can be calcualted up front
        liab_series = self.liabilities.cashflows.loc[val_date:][:max_years]
        liab_cashflows = np.asarray(liab_series.values, dtype=float)
        dates  = pd.DatetimeIndex(liab_series.index)
        T = len(dates)
        dt = calc_dt(dates, val_date)

        # PV timelines
        liab_pv_gaap = np.asarray(self.liabilities.pv(val_date, self.config.r_gaap, timeline=True, n_years=max_years), dtype=float)
        liab_pv_ifrs = np.asarray(self.liabilities.pv(val_date, self.config.r_ifrs, timeline=True, n_years=max_years), dtype=float)

        # Meltdown liabilities
        liab_cashflows_arr = np.asarray(liab_cashflows, dtype=float)
        cum_liab_cfs = liab_cashflows_arr.cumsum()
        meltdown_liabs = np.maximum(
            0.0,
            (self.config.heubeck_liabilities - cum_liab_cfs) * self.config.mortality_buffer
            * ((1 + self.config.r_gaap) ** dt)
        )

        # Next 2 liabilities
<<<<<<< HEAD
        full_cfs = np.asarray(self.liabilities.cashflows.loc[val_date:].values, dtype=float)
=======
        full_cfs = self.liabilities.cashflows.loc[val_date:].values
>>>>>>> b915299a24f972d1aee59a47871cad61ad35f44d
        if len(full_cfs) < T + 2:
            raise ValueError(
                f"Liability cashflows too short: need at least {T + 2} entries from val_date, got {len(full_cfs)}."
            )
        next2 = full_cfs[1 : T + 1] + full_cfs[2 : T + 2]

        ### Calculate Asset Values ###
        # Price bonds at val_date and calculate expected cashflows
        bond_prices = self.bonds.pv(pd.Timestamp(val_date), rates, spread_map)

        # market value of CDI and CMBP portfolios at val date
        cdi_t0 = (self.cdi_allocation * bond_prices).sum()
        cmbp_t0 = (self.cmbp_allocation * bond_prices).sum()

        # total assets at val date
        total_assets_0 = cdi_t0 + self.cash

        # Get expected (no default) cashflows for CDI portfolio
        raw_bond_cfs = self.bonds.year_end_cashflows(val_date)
        exp_cdi_cf = np.asarray((raw_bond_cfs * self.cdi_allocation).sum(axis=1).values, dtype=float)
        if len(exp_cdi_cf) < T:
            exp_cdi_cf = np.pad(exp_cdi_cf, (0, T - len(exp_cdi_cf)), mode='constant')
        else:
            exp_cdi_cf = exp_cdi_cf[:T]

        ## Calculate starting HGB Gap
        day0_hgb_gap = (self.config.heubeck_liabilities * self.config.mortality_buffer - (total_assets_0 + self.config.asset_buffer))

        ## Run stochastic bond simulation for all scenarios
        bond_sim  = self.bonds.run_sim(val_date, rates, spread_map, sim_results)

        # Calculate CDI and CMBP portfolio cashflows and PVs from bond sim.
        shape = (n_sim, T)
        asset_cfs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'cashflows')
        asset_pvs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'pvs')
        cmbp_cfs = allocate_bond_sim(bond_sim, self.cmbp_allocation, shape, 'cashflows')
        cmbp_pvs = allocate_bond_sim(bond_sim, self.cmbp_allocation, shape, 'pvs')

        # Calculate annual total returns of CMBP portfolio, which are used for the bund comparator.
        fwds = np.asarray(rates.calc_fwds(val_date, dates).values, dtype=float)
        yields = np.asarray(rates.interpolate(dates).values, dtype=float)
        bt = total_returns(cmbp_pvs, cmbp_cfs, fwds, cmbp_t0)

        ### Run asset-liability waterfall ###

        # create empty 3d arrays for results we want to store
        cash_arr = np.zeros(shape)
        assets_arr = np.zeros(shape)
        fee_arr = np.zeros(shape)
        hgb_gap_arr = np.zeros(shape)
        hgb_payment_arr = np.zeros(shape)
        perf_payment_arr = np.zeros(shape)
        additional_payment_arr = np.zeros(shape)
        bund_comparator_arr = np.zeros(shape)
        net_asset_ret = np.zeros(shape)

        # Asset buffer we can calculate up front
        asset_buffer_arr = np.array([self.config.asset_buffer*((1 + self.config.r_ifrs)**(t+1)) if t < 10 else 0.0 for t in range(T)])

        # create vectors to hold point in time values at set to required starting values
        cash_t = np.full(n_sim, self.cash)
        assets_t = np.full(n_sim, total_assets_0)
        hgb_gap_t = np.full(n_sim, day0_hgb_gap)
        bund_t = np.full(n_sim, total_assets_0)
        cum_hgb_payment = np.zeros(n_sim)

        for t in range(T):
            # Store current asset value to calculate returns at end
            prev_assets_t = assets_t

            # Calcualte fee
            fee_t  = self.config.fee * (cash_t * (1 + fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])

            # Cash after asset cashflow, liability payment and fee.
            cash_t = cash_t * (1 + fwds[t]) + asset_cfs[:, t] - liab_cashflows[t] - fee_t

            # updated total assets at t
            assets_t = cash_t + asset_pvs[:, t]

            # calculate updated hgb gap and make payment if needed
            hgb_gap_t = np.clip(meltdown_liabs[t] - (assets_t + asset_buffer_arr[t]), 0.0, hgb_gap_t)
            hgb_pay_t = np.clip(next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb_payment)

            # add payment to cash (and assets) balance
            cash_t += hgb_pay_t
            assets_t += hgb_pay_t
            cum_hgb_payment += hgb_pay_t

            # Additional payment from client at the configured year if the scheme is underfunded.
            additional_payment_t = 0.0
            if t == self.config.additional_payment_year - 1:
<<<<<<< HEAD
                additional_payment_t = np.clip(np.minimum(np.asarray(liab_pv_gaap, dtype=float)[t] - assets_t, 1.1 * np.asarray(liab_pv_ifrs, dtype=float)[t] - assets_t), 0.0, asset_buffer_arr[t])
=======
                additional_payment_t = np.clip(np.minimum(liab_pv_gaap[t] - assets_t, 1.1 * liab_pv_ifrs[t] - assets_t), 0.0, asset_buffer_arr[t])
>>>>>>> b915299a24f972d1aee59a47871cad61ad35f44d
                cash_t += additional_payment_t
                assets_t += additional_payment_t

            # Performance guarantee (bund comparator)
            bund_t = bund_t * (1 + bt[:, t] + self.config.cmbp_margin) - liab_cashflows[t] + additional_payment_t
            performance_payment = np.clip(bund_t - assets_t, 0.0, self.config.performance_cap) if t == self.config.performance_payment_year - 1 else 0.0

            # Net asset return (excl. liability payment and additional injection)
            net_ret_t = (assets_t + liab_cashflows[t] - additional_payment_t) / prev_assets_t - 1.0

            # Store
            fee_arr[:, t] = fee_t
            cash_arr[:, t] = cash_t
            assets_arr[:, t] = assets_t
            hgb_gap_arr[:, t] = hgb_gap_t
            hgb_payment_arr[:, t] = hgb_pay_t
            perf_payment_arr[:, t] = performance_payment
            additional_payment_arr[:, t] = additional_payment_t
            bund_comparator_arr[:, t] = bund_t
            net_asset_ret[:, t] = net_ret_t

        ### Prepare output df ###
        tile = lambda x: np.tile(x, (n_sim, 1))
        results = {
            "dt": tile(dt),
            "liab_cashflow": tile(liab_cashflows),
            "liab_pv_gaap": tile(liab_pv_gaap),
            "liab_pv_ifrs": tile(liab_pv_ifrs),
            "meltdown_liabilities": tile(meltdown_liabs),
            "next_2_liabs": tile(next2),
            "bund_fwds": tile(fwds),
            "bund_yield": tile(yields),
            "expected_cdi_cashflow": tile(exp_cdi_cf),
            "asset_cashflow": asset_cfs,
            "remaining_asset_pv": asset_pvs,
            "cmbp_cashflow": cmbp_cfs,
            "cmbp_pv": cmbp_pvs,
            "bt": bt,
            "fee": fee_arr,
            "cash": cash_arr,
            "assets": assets_arr,
            "net_asset_return": net_asset_ret,
            "hgb_gap": hgb_gap_arr,
            "hgb_payment": hgb_payment_arr,
            "bund_comparator": bund_comparator_arr,
            "performance_payment": perf_payment_arr,
            "additional_payment": additional_payment_arr,
        }
        index = pd.MultiIndex.from_product([np.arange(n_sim), dates], names=["scenario", "date"])
        df = pd.DataFrame({k: v.reshape(-1) for k, v in results.items()}, index=index).reset_index()

        # Calculate some additional metrics
        df["funding_level_gaap"] = df["assets"] / df["liab_pv_gaap"]
        df["funding_level_ifrs"] = df["assets"] / df["liab_pv_ifrs"]
        df["net_bt_return"] = df["bt"] + self.config.cmbp_margin

        return CDISimulationResult(config=self.config, bond_sim=bond_sim, cdi_sim=df)