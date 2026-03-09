import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Sequence, Union, Dict, Optional
import warnings
import scipy as sp

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
DAYS_PER_YEAR = 365.0
RATINGS_ORDER = [
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def'
]
DEFAULT_LABEL = "Def"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def calc_dt(dates: Union[pd.DatetimeIndex, Sequence[datetime]], val_date: datetime) -> np.ndarray:
    """Year fractions from *val_date* to each element of *dates*."""
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    delta_days = (dates - pd.Timestamp(val_date)).days
    return np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR


def fit_to_shape(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Trim / zero-pad *a* so it matches *shape*."""
    if len(shape) == 3 and a.ndim == 2:
        a = np.expand_dims(a, axis=-1)
    trimmed = a[tuple(slice(0, min(a.shape[i], shape[i])) for i in range(len(shape)))]
    pad_widths = [(0, max(0, shape[i] - trimmed.shape[i])) for i in range(len(shape))]
    while len(pad_widths) < trimmed.ndim:
        pad_widths.append((0, 0))
    return np.pad(trimmed, pad_widths, mode='constant')

def map_spreads_chunked(ratings: np.ndarray, spread_map: Dict[str, float], chunk_size: int = 1000) -> np.ndarray:
    """
    Chunked, vectorised rating-label → spread lookup.

    Parameters
    - ratings: ndarray of rating labels (any shape). Typically dtype=object/str.
    - spread_map: mapping from rating label -> spread.
    - chunk_size: number of elements processed per chunk (flattened).
    - default: value used when a rating is missing from `spread_map`.
    - out: optional preallocated output array (same shape as `ratings`).

    Returns
    - ndarray of spreads with same shape as `ratings`.
    """
    flat = np.asarray(ratings).ravel()

    out_arr = np.empty(flat.shape, dtype=float)

    n = flat.size
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = flat[start:stop]
        # Factorise per chunk to avoid a single massive codes/uniques allocation.
        codes, uniques = pd.factorize(chunk, sort=False)
        mapped = np.fromiter(
            (spread_map.get(u, 0.0) for u in uniques),
            dtype=float,
            count=len(uniques),
        )
        out_arr[start:stop] = mapped[codes]

    return out_arr.reshape(ratings.shape)

def map_spreads(ratings: np.ndarray, spread_map: Dict[str, float]) -> np.ndarray:
    """Vectorised rating-label → spread lookup."""
    flat = ratings.ravel()
    codes, uniques = pd.factorize(flat, sort=False)
    mapped = np.fromiter(
        (spread_map.get(u, 0.0) for u in uniques),
        dtype=float, count=len(uniques),
    )
    return mapped[codes].reshape(ratings.shape)


def compute_maturity_flags(cashflows: np.ndarray) -> np.ndarray:
    """Boolean mask (n_years × n_bonds): True up to and including the last non-zero cashflow year."""
    n_years, n_bonds = cashflows.shape
    flipped = np.flip(cashflows, axis=0)
    has_cashflow = cashflows.any(axis=0)
    last_cf_idx = np.where(has_cashflow, n_years - 1 - np.argmax(flipped != 0, axis=0), -1)
    year_indices = np.arange(n_years)[:, np.newaxis]
    return year_indices <= last_cf_idx[np.newaxis, :]


# ---------------------------------------------------------------------------
#  Rates
# ---------------------------------------------------------------------------

class Rates:
    """Holds and interpolates a yield curve."""

    def __init__(self, yields: np.ndarray, dates: List):
        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")
        self.yields = pd.Series(np.asarray(yields, dtype=float), index=pd.DatetimeIndex(dates))

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        combined = self.yields.index.union(dates).sort_values()
        return self.yields.reindex(combined).interpolate(method="time").reindex(dates)

    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex) -> pd.Series:
        """Implied forward rates between consecutive *dates*."""
        y = self.interpolate(dates)
        t = calc_dt(pd.DatetimeIndex(y.index), val_date)
        acc = np.power(1.0 + np.asarray(y.values, dtype=float), t)
        dt = np.diff(t)
        fwds = np.empty_like(y.values)
        fwds[0] = y.iloc[0]
        fwds[1:] = (acc[1:] / acc[:-1]) ** (1.0 / dt) - 1.0
        return pd.Series(fwds, index=y.index, name="fwds")


# ---------------------------------------------------------------------------
#  Credit-risk model
# ---------------------------------------------------------------------------

class TransitionMatrix:
    """Square credit-rating transition matrix with labelled states."""

    def __init__(self, tmatrix: np.ndarray, labels: list[str]):
        if tmatrix.ndim != 2 or tmatrix.shape[0] != tmatrix.shape[1]:
            raise IndexError("Transition matrix must be square and 2-dimensional")
        if len(labels) != tmatrix.shape[0]:
            raise IndexError("Labels length must match matrix dimension")

        self.tmatrix = tmatrix
        self.cum_tmatrix = np.cumsum(tmatrix, axis=1)
        self.labels = np.array(labels)
        self.label_to_idx = {l: i for i, l in enumerate(labels)}

        bad_rows = np.where(~np.isclose(self.cum_tmatrix[:, -1], 1.0))[0]
        if bad_rows.size:
            raise ValueError(f"Transition probabilities for {self.labels[bad_rows]} do not sum to 1")

    def indices_to_labels(self, indices: np.ndarray) -> np.ndarray:
        return self.labels[indices]

    def _step(self, cur: np.ndarray, pX_t: np.ndarray) -> np.ndarray:
        """Single migration step: map current rating indices + uniform draws → new indices."""
        cum = self.cum_tmatrix[cur]
        return np.clip(
            np.sum(pX_t[:, :, np.newaxis] > cum if pX_t.ndim == 2 else pX_t > cum, axis=-1),
            0, self.tmatrix.shape[0] - 1,
        ).astype(np.int32)

    def transitions(self, pX: np.ndarray, ratings_map: pd.Series) -> np.ndarray:
        """
        Vectorised ratings migration from a common initial state.
        pX: (n_sim, n_years, n_issuers) CDF(N(0,1)) draws.
        Returns: (n_sim, n_years+1, n_issuers) rating-label array.
        """
        n_sim, n_years, n_issuers = pX.shape
        initial = np.array([self.label_to_idx[r] for r in ratings_map])

        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial[np.newaxis, :]

        for t in range(n_years):
            cum = self.cum_tmatrix[res[:, t, :]]
            pX_t = pX[:, t, :, np.newaxis]
            res[:, t + 1, :] = np.clip(
                np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1
            ).astype(np.int32)

        return self.indices_to_labels(res)

    def transitions_from_matrix(self, pX: np.ndarray, initial_indices: np.ndarray) -> np.ndarray:
        """
        Same as *transitions* but with per-simulation initial ratings.
        initial_indices: (n_sim, n_issuers) int array.
        """
        n_sim, n_years, n_issuers = pX.shape

        initial_indices_int = np.array([[self.label_to_idx[r] for r in initial_indices[s]] for s in range(n_sim)])
        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial_indices_int

        for t in range(n_years):
            cum = self.cum_tmatrix[res[:, t, :]]
            pX_t = pX[:, t, :, np.newaxis]
            res[:, t + 1, :] = np.clip(
                np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1
            ).astype(np.int32)

        return self.indices_to_labels(res)

    def fundamental_matrix(self) -> pd.DataFrame:
        Q = self.tmatrix[:-1, :-1]
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        labels = self.labels[:-1]
        return pd.DataFrame(N, index=labels, columns=labels)

    def time_to_default(self) -> pd.Series:
        N = self.fundamental_matrix().values
        t = N @ np.ones(len(N))
        return pd.Series(t, index=self.labels[:-1], name='Time to Default')

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt=".1%"))


class SimulationResult:
    """Container for CreditRiskModel outputs."""
    def __init__(self, E, S, I, X, pX, transitions, n_sim, n_years, rho_e=None, rho_s=None):
        self.E = E
        self.S = S
        self.I = I
        self.X = X
        self.pX = pX
        self.transitions = transitions
        self.n_sim = n_sim
        self.n_years = n_years
        self.rho_e = rho_e
        self.rho_s = rho_s


class CreditRiskModel:
    """Monte-Carlo factor-based credit-risk model (BRS Credit VaR)."""

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        rho_e: float,
        rho_s: np.ndarray,
        issuer_ids: list,
        sector_map: pd.Series,
        ratings_map: pd.Series,
    ):
        self.transition_matrix = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        self.sector_map = sector_map
        self.ratings_map = ratings_map
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.sector_map, pd.Series):
            raise TypeError("sector_map must be a pandas Series.")
        if not isinstance(self.ratings_map, pd.Series):
            raise TypeError("ratings_map must be a pandas Series.")
        if not isinstance(self.rho_s, np.ndarray):
            raise TypeError("rho_s must be a numpy array.")
        if not np.isscalar(self.rho_e):
            raise TypeError("rho_e must be a scalar.")

        n_issuer_sectors = self.sector_map.nunique()
        n_sectors = len(self.rho_s)
        if n_sectors > n_issuer_sectors:
            warnings.warn(
                f"{n_sectors} sector correlations defined but only {n_issuer_sectors} sectors in sector_map.",
                UserWarning,
            )
        if np.any(self.rho_e > self.rho_s):
            raise ValueError("rho_s must be >= rho_e")

    def _run_pX(self, n_sim, n_years, n_issuers, sector_map) -> np.ndarray:

        n_sectors = sector_map.nunique()
        E = np.random.normal(size=(n_sim, n_years))
        S = np.random.normal(size=(n_sim, n_years, n_sectors))
        I = np.random.normal(size=(n_sim, n_years, n_issuers))
        X = (
            np.sqrt(self.rho_e) * E[:, :, np.newaxis]
            + np.sqrt(self.rho_s[sector_map] - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, sector_map]
            + np.sqrt(1 - self.rho_s[sector_map])[np.newaxis, np.newaxis, :] * I
        )
        pX = sp.stats.norm.cdf(X)

        return pX

    def run_matrix(self, n_sim, n_years, n_issuers, sector_map, matrix: np.ndarray) -> SimulationResult:
         """Run the full simulation and return all outputs."""

         pX = self._run_pX(n_sim, n_years, n_issuers, sector_map)
         transitions = self.transition_matrix.transitions_from_matrix(pX, matrix)
         transitions = pd.DataFrame(
             transitions.reshape(n_sim * (n_years + 1), n_issuers),
             index=pd.MultiIndex.from_product(
                 [range(n_sim), range(n_years + 1)], names=["sim", "year"]
             ),
             columns=self.issuer_ids,
         )
         return SimulationResult(
             E=None, S=None, I=None, X=None, pX=pX, transitions=transitions, n_sim=n_sim, n_years=n_years
         )

    def run(self, n_sim: float, n_years: int) -> SimulationResult:

        n_sim = int(n_sim)
        n_issuers = len(self.ratings_map)
        n_sectors = self.sector_map.nunique()

        E = np.random.normal(size=(n_sim, n_years))
        S = np.random.normal(size=(n_sim, n_years, n_sectors))
        I = np.random.normal(size=(n_sim, n_years, n_issuers))

        rho_s_i = self.rho_s[self.sector_map]
        X = (
            np.sqrt(self.rho_e) * E[:, :, np.newaxis]
            + np.sqrt(rho_s_i - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map]
            + np.sqrt(1 - rho_s_i)[np.newaxis, np.newaxis, :] * I
        )
        pX = sp.stats.norm.cdf(X)

        transitions = self.transition_matrix.transitions(pX, self.ratings_map)
        transitions = pd.DataFrame(
            transitions.reshape(n_sim * (n_years + 1), n_issuers),
            index=pd.MultiIndex.from_product(
                [range(n_sim), range(n_years + 1)], names=["sim", "year"]
            ),
            columns=self.issuer_ids,
        )
        return SimulationResult(E, S, I, X, pX, transitions, n_sim, n_years, self.rho_e, self.rho_s)


# ---------------------------------------------------------------------------
#  Assets
# ---------------------------------------------------------------------------

class Issuers:
    """Container for bond-issuer details."""
    def __init__(self, ids: List[str], ratings: pd.Series, sectors: pd.Series,
                 names: Optional[pd.Series] = None):
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

    def __init__(self, ids: list, issuer_ids: pd.Series, recoveries: pd.Series,
                 cashflows: pd.DataFrame, issuers: Issuers):
        self.ids = ids
        self.issuer_ids = issuer_ids
        self.recoveries = recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.ids, (list, tuple)):
            raise TypeError("ids must be a list or tuple.")
        if len(self.ids) == 0:
            raise ValueError("ids cannot be empty.")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("ids must be unique.")

        for name, s in {"issuer_ids": self.issuer_ids, "recoveries": self.recoveries}.items():
            if not isinstance(s, pd.Series):
                raise TypeError(f"{name} must be a pandas Series.")
            if not s.index.isin(self.ids).all():
                raise ValueError(f"{name} index must match bond ids.")

        if ((self.recoveries < 0) | (self.recoveries > 1)).any():
            raise ValueError("recoveries must be between 0 and 1.")

        cf = self.cashflows
        if not isinstance(cf, pd.DataFrame):
            raise TypeError("cashflows must be a pandas DataFrame.")
        if not isinstance(cf.index, pd.DatetimeIndex):
            date_col = next((c for c in cf.columns if c.lower() == "date"), None)
            if date_col is None:
                raise ValueError("cashflows must have a DatetimeIndex or a 'date' column.")
            cf = cf.set_index(pd.to_datetime(cf[date_col], errors="raise")).drop(columns=date_col)
        missing_cf = set(self.ids) - set(cf.columns)
        if missing_cf:
            raise ValueError(f"cashflows missing columns for: {missing_cf}")
        if not cf.apply(pd.api.types.is_numeric_dtype).all():
            raise TypeError("All cashflow columns must be numeric.")
        self.cashflows = cf[self.ids].sort_index()

        missing_issuers = set(self.issuer_ids.values) - set(self.issuers.ids)
        if missing_issuers:
            raise ValueError(f"issuer_ids missing in issuers: {missing_issuers}")

    def year_end_cashflows(self, val_date: Optional[datetime] = None) -> pd.DataFrame:
        cf = self.cashflows.resample("ME").sum()
        if val_date is not None:
            cf = cf.loc[cf.index > val_date]
        return cf.resample("YE").sum()

    def pv(self, val_date: pd.Timestamp, rates: Rates, spread_map: Dict[str, float]) -> pd.Series:
        """Present value of each bond at *val_date*."""
        cfs = self.year_end_cashflows(val_date)
        dt = calc_dt(pd.DatetimeIndex(cfs.index), val_date)
        base_yields = np.asarray(rates.interpolate(pd.DatetimeIndex(cfs.index)).values, dtype=float)
        current_ratings = self.issuer_ids.map(self.issuers.ratings).reindex(self.ids)
        spreads = map_spreads(np.asarray(current_ratings.values)[np.newaxis, :], spread_map).squeeze(0)
        dfs = (1 + base_yields[:, np.newaxis] + spreads[np.newaxis, :]) ** (-dt[:, np.newaxis])
        prices = (cfs.values * dfs).sum(axis=0)
        return pd.Series(prices, index=self.ids)

    def _run_sim_pv(self, cashflows: np.ndarray, bond_transitions: np.ndarray, spread_map: dict, yields: np.ndarray, dt: np.ndarray) -> np.ndarray:
        n_years_cf, n_bonds = cashflows.shape
        n_sim, n_years_sim, _ = bond_transitions.shape
        n_years = min(n_years_cf, n_years_sim)
        assert cashflows.shape[1] == bond_transitions.shape[2]

        unique_spreads, spread_inverse = np.unique(
            map_spreads_chunked(bond_transitions, spread_map), return_inverse=True
        )
        n_unique = len(unique_spreads)

        dtime_mat = dt[:, np.newaxis] - dt[:n_years][np.newaxis, :]
        future = dtime_mat > 0
        yield_mat = (1 + yields)[:, np.newaxis] * np.ones((1, n_years))

        pv_table = np.zeros((n_unique, n_years, n_bonds))
        for si, s in enumerate(unique_spreads):
            dfs_mat = np.zeros_like(dtime_mat)
            dfs_mat[future] = (yield_mat[future] + s) ** (-dtime_mat[future])
            pv_table[si] = dfs_mat.T @ cashflows

        spread_idx = spread_inverse.reshape(n_sim, n_years, n_bonds)
        t_idx = np.arange(n_years)[np.newaxis, :, np.newaxis]
        b_idx = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        return pv_table[spread_idx, t_idx, b_idx]

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> BondSimulationResult:
        """Project cashflows and revalue bonds across all scenarios."""

        cashflows_df = self.year_end_cashflows(val_date)
        cashflows = cashflows_df.values
        dates = cashflows_df.index
        n_years_cf, n_bonds = cashflows.shape
        n_sim = sim_results.n_sim
        n_years_sim = sim_results.n_years
        n_years = min(n_years_sim, n_years_cf)

        # Map issuer transitions → bonds, drop year-0
        bond_transitions = (
            sim_results.transitions
            .loc[sim_results.transitions.index.get_level_values("year") != 0, self.issuer_ids]
            .set_axis(self.ids, axis=1)
            .to_numpy()
            .reshape(n_sim, n_years_sim, n_bonds)
        )[:, :n_years, :]

        not_matured = compute_maturity_flags(cashflows)

        is_defaulted = (bond_transitions == DEFAULT_LABEL)
        not_defaulted = ~is_defaulted
        first_default = (np.cumsum(is_defaulted, axis=1) == 1) & is_defaulted

        cf_slice = cashflows[np.newaxis, :n_years, :]
        received_cashflows = cf_slice * not_defaulted
        recoveries = np.asarray(self.recoveries.values, dtype=float)
        recovery_payments = (
            first_default * recoveries[np.newaxis, np.newaxis, :] * not_matured[np.newaxis, :n_years, :]
        )
        total_cashflows = received_cashflows + recovery_payments

        yields = np.asarray(rates.interpolate(pd.DatetimeIndex(dates)).values, dtype=float)
        dt = calc_dt(pd.DatetimeIndex(dates), val_date)
        pvs = self._run_sim_pv(cashflows, bond_transitions, spread_map, yields, dt)

        # Default is absorbing – PV = 0 from first default onward
        ever_defaulted = np.cumsum(is_defaulted, axis=1) > 0
        pvs = pvs * ~ever_defaulted

        return BondSimulationResult(
            transitions=bond_transitions,
            received_cashflows=received_cashflows,
            recovery_payments=recovery_payments,
            total_cashflows=total_cashflows,
            pvs=pvs,
            dates=dates[:n_years],
            bond_ids=self.ids,
        )


# ---------------------------------------------------------------------------
#  Liabilities
# ---------------------------------------------------------------------------

class Liabilities:
    def __init__(self, cashflows: pd.Series, dates: List):
        if len(cashflows) != len(dates):
            raise ValueError("Cashflows and dates length mismatch.")
        self.cashflows = pd.Series(cashflows, index=pd.to_datetime(dates))

    def pv(self, val_date: datetime, rates: Union[Rates, float], shift=0.0,
           timeline=False, n_years: Optional[float] = None) -> Union[float, np.ndarray]:
        cashflows = self.cashflows.loc[self.cashflows.index > val_date]
        dates = cashflows.index
        dt = calc_dt(pd.DatetimeIndex(dates), val_date)

        if isinstance(rates, (int, float)):
            net_yield = np.full(len(cashflows), float(rates) + shift, dtype=float)
        else:
            net_yield = np.asarray(rates.interpolate(pd.DatetimeIndex(dates)).values, dtype=float) + shift

        cf_values = np.asarray(cashflows.values, dtype=float)

        if timeline:
            pv = np.array([
                np.sum(cf_values[i + 1:] * ((1 + net_yield[i + 1:]) ** -(dt[i + 1:] - dt[i])))
                for i in range(len(dates))
            ])
            return pv[:n_years] if n_years is not None else pv
        else:
            return (cf_values * (1 + net_yield) ** -dt).sum()

    def __str__(self):
        return pd.DataFrame(self.cashflows).to_markdown(floatfmt='.0f')


# ---------------------------------------------------------------------------
#  CDI mandate helpers
# ---------------------------------------------------------------------------

def allocate_bond_sim(bond_sim: BondSimulationResult, allocation: pd.Series,
                      shape: tuple, target: str = 'cashflows') -> np.ndarray:
    """Extract portfolio-level cashflows or PVs from a bond simulation by allocation."""
    ids = allocation.index.to_list()
    allocs = allocation.values
    bond_idx = {bid: i for i, bid in enumerate(bond_sim.bond_ids)}
    selected = [bond_idx[id] for id in ids]

    if target == 'pvs':
        nominal = bond_sim.pvs[:, :, selected]
    elif target == 'cashflows':
        nominal = bond_sim.total_cashflows[:, :, selected]
    else:
        raise TypeError('target must be "cashflows" or "pvs"')

    return fit_to_shape((nominal * allocs).sum(axis=2), shape)

# ---------------------------------------------------------------------------
#  CDI result containers
# ---------------------------------------------------------------------------

class CDISimulationResult:
    """Output container for CDIMandate_Fox.run()."""

    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, config: "FoxConfig", schedule: dict, bond_sim: BondSimulationResult, cdi_sim: pd.DataFrame, sim_results: SimulationResult):
        self.config = config
        self.schedule = schedule
        self.bond_sim = bond_sim
        self.cdi_sim = cdi_sim
        self.sim_results = sim_results

    def _discount_factors(self) -> pd.Series:
        df = self.cdi_sim
        return (1 + df["bund_yield"]) ** (-df["dt"])

    def _pv_by_scenario(self, column: str) -> pd.Series:
        return (self.cdi_sim[column] * self._discount_factors()).groupby(self.cdi_sim["scenario"]).sum()

    def pv_hgb_payments(self) -> pd.Series:
        return self._pv_by_scenario("hgb_payment").rename("pv_hgb_payment")

    def pv_performance_payments(self) -> pd.Series:
        return self._pv_by_scenario("performance_payment").rename("pv_performance_payment")

    def pv_total_obligations(self) -> pd.Series:
        return (self.pv_hgb_payments() + self.pv_performance_payments()).rename("pv_total_obligations")

    def pv_fees(self) -> pd.Series:
        return self._pv_by_scenario("fee").rename("pv_fee")

    def pv_net(self) -> pd.Series:
        return (self.pv_fees() - self.pv_total_obligations()).rename("pv_net")

    def obligation_metrics(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        qs = quantiles or self._DEFAULT_QUANTILES

        pv_hgb  = self.pv_hgb_payments()
        pv_perf = self.pv_performance_payments()
        pv_tot  = self.pv_total_obligations()
        pv_fee  = self.pv_fees()
        pv_net  = self.pv_net()

        scenarios = pd.DataFrame({
            pv_hgb.name:  pv_hgb,
            pv_perf.name: pv_perf,
            pv_tot.name:  pv_tot,
            pv_fee.name:  pv_fee,
            pv_net.name:  pv_net,
        })

        summary = scenarios.quantile(qs).T
        summary.insert(0, "mean", scenarios.mean())

        prob = pd.DataFrame(
            {"mean": [(pv_hgb > 0).mean(), (pv_perf > 0).mean(), (pv_net < 0).mean()]},
            index=["prob_hgb_trigger", "prob_perf_trigger", "prob_net_negative"],
        )
        for q in qs:
            prob[q] = np.nan

        return pd.concat([summary, prob])

    def write_output(self, output_location, folder_name):
        import json

        target_dir = output_location / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Config with added correlation parameters
        config_dict = self.config.to_dict()
        config_dict['timestamp '] = pd.Timestamp.now().isoformat()
        config_dict['rho_s'] = list(self.sim_results.rho_s)
        config_dict['rho_e'] = self.sim_results.rho_e
        config_dict['n_sim'] = self.sim_results.n_sim
        config_dict['n_years'] = self.sim_results.n_years
        with open(output_location / f"{folder_name}/config.json", "w", encoding="utf-8") as f: json.dump(config_dict, f, indent=2, sort_keys=True)

        # Static results
        dict = self.schedule
        det_df = pd.DataFrame(dict).drop(columns=['T'])
        det_df.to_csv(output_location / f"{folder_name}/static_results.csv", index=False)

        # Bond simulation results
        selected_cols = [
            'date',
            'scenario',
            'remaining_asset_pv',
            'asset_cashflow',
            'cash',
            'assets',
            'hgb_gap',
            'additional_payment',
            'fee',
            'hgb_payment',
            'bund_comparator',
            'performance_payment'
        ]
        df = self.cdi_sim[selected_cols]
        df.to_csv(output_location / f"{folder_name}/cdi_sim.csv", index=False)


class ICAAPResult:
    """Container for ICAAP nested Monte-Carlo capital results."""

    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, pv_hgb_t1, pv_perf_t1, pv_fee_t1, icaap_df: Optional[pd.DataFrame] = None):
        self.pv_hgb_t1  = pv_hgb_t1
        self.pv_perf_t1 = pv_perf_t1
        self.pv_fee_t1 = pv_fee_t1
        self.icaap_df = icaap_df

    def summary(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        qs = quantiles or self._DEFAULT_QUANTILES
        scenarios = pd.DataFrame({
            "pv_hgb_t1": self.pv_hgb_t1,
            "pv_perf_t1": self.pv_perf_t1,
            "pv_fee_t1": self.pv_fee_t1
        })
        tbl = scenarios.quantile(qs).T
        tbl.insert(0, "mean", scenarios.mean())
        return tbl


# ---------------------------------------------------------------------------
#  Fox mandate configuration & simulation
# ---------------------------------------------------------------------------

class FoxConfig:
    def __init__(
        self,
        val_date: datetime,
        heubeck_liabilities,
        r_gaap,
        r_ifrs,
        cmbp_margin,
        asset_buffer,
        mortality_buffer,
        fee,
        performance_cap,
        cash,
        cdi_t0,
        additional_payment_year: int = 10,
        performance_payment_year: int = 25
    ):
        self.val_date = val_date
        self.heubeck_liabilities = heubeck_liabilities
        self.r_gaap = r_gaap
        self.r_ifrs = r_ifrs
        self.cmbp_margin = cmbp_margin
        self.asset_buffer = asset_buffer
        self.mortality_buffer = mortality_buffer
        self.fee = fee
        self.performance_cap = performance_cap
        self.additional_payment_year = additional_payment_year
        self.performance_payment_year = performance_payment_year
        self.cash = cash
        self.cdi_t0 = cdi_t0

    def to_dict(self):
        return vars(self)


class CDIMandate_Fox:
    """Fox CDI mandate: asset-liability waterfall with performance guarantee."""

    def __init__(self, liabilities: Liabilities, cash: float, bonds: Bonds,
                 cdi_allocation: pd.Series, cmbp_allocation: pd.Series,
                 config: FoxConfig):
        self.liabilities = liabilities
        self.cash = cash
        self.bonds = bonds
        self.cdi_allocation = cdi_allocation
        self.cmbp_allocation = cmbp_allocation
        self.config = config

    def _calc_deterministic(self, val_date: datetime, rates: Rates, n_years: int) -> dict:
        """Pre-compute all deterministic arrays needed by the waterfall."""
        self.config = self.config

        # Liability series
        liab_series = self.liabilities.cashflows.loc[val_date:][:n_years]
        liab_cashflows = np.asarray(liab_series.values, dtype=float)
        dates = pd.DatetimeIndex(liab_series.index)
        T = len(dates)
        dt = calc_dt(dates, val_date)

        # Liability PV timelines
        liab_pv_gaap = np.asarray(self.liabilities.pv(val_date, self.config.r_gaap, timeline=True, n_years=n_years), dtype=float)
        liab_pv_ifrs = np.asarray(self.liabilities.pv(val_date, self.config.r_ifrs, timeline=True, n_years=n_years), dtype=float)

        # Meltdown liabilities
        cum_liab = liab_cashflows.cumsum()
        meltdown_liabs = np.maximum(
            0.0,
            (self.config.heubeck_liabilities - cum_liab) * self.config.mortality_buffer * ((1 + self.config.r_gaap) ** dt),
        )

        # Next-2-year liabilities
        full_cfs = np.asarray(self.liabilities.cashflows.loc[val_date:].values, dtype=float)
        if len(full_cfs) < T + 2:
            raise ValueError(f"Liability cashflows too short: need {T + 2} from val_date, got {len(full_cfs)}.")
        next2 = full_cfs[1 : T + 1] + full_cfs[2 : T + 2]

        # Rates
        fwds   = np.asarray(rates.calc_fwds(val_date, dates).values, dtype=float)
        yields = np.asarray(rates.interpolate(dates).values, dtype=float)
        df = (1 + yields) ** (-dt)

        # Asset buffer (accrues at IFRS rate for first 10 years, then 0)
        asset_buffer = np.array([
            self.config.asset_buffer * ((1 + self.config.r_ifrs) ** (t + 1)) if t < 10 else 0.0
            for t in range(T)
        ])

        # Expected (no-default) CDI cashflows
        raw = self.bonds.year_end_cashflows(val_date)
        exp = np.asarray((raw[self.cdi_allocation.index] * self.cdi_allocation).sum(axis=1).values, dtype=float)
        exp_cdi_cf = np.pad(exp, (0, max(0, T - len(exp))))[:T]

        return {
            "date": dates,
            "T": T,
            "dt": dt,
            "yields": yields,
            "fwds": fwds,
            "df": df,
            "liab_cashflows": liab_cashflows,
            "liab_pv_gaap": liab_pv_gaap,
            "liab_pv_ifrs": liab_pv_ifrs,
            "meltdown_liabs": meltdown_liabs,
            "next2": next2,
            "asset_buffer": asset_buffer,
            "exp_cdi_cf": exp_cdi_cf
        }

    def _calc_bt(self, val_date: datetime, rates: Rates, schedule: dict) -> tuple[np.ndarray, float]:
        """
        CMBP total returns are deterministic because all CMBP bonds carry the
        AAAA rating, which is absorbing (no migration, no default).
        Returns (bt, cmbp_t0) where bt has shape (T,).
        """


        raw_cfs = self.bonds.year_end_cashflows(val_date)
        cmbp_ids = self.cmbp_allocation.index

        dates_full = pd.DatetimeIndex(raw_cfs.index)
        dt_full = calc_dt(dates_full, val_date)
        yields_full = np.asarray(rates.interpolate(dates_full).values, dtype=float)
        n_full = len(dates_full)
        T = schedule["T"]

        # Portfolio cashflows per year (full horizon, may extend beyond T)
        cmbp_cfs = np.asarray((raw_cfs[cmbp_ids] * self.cmbp_allocation).sum(axis=1).values, dtype=float)

        # Market value at val_date (AAAA spread = 0)
        cmbp_t0 = float((cmbp_cfs * (1 + yields_full) ** (-dt_full)).sum())

        # PV at each year-end within the simulation horizon
        pvs = np.zeros(T)
        for t in range(T):
            mask = np.arange(n_full) > t
            dtime = dt_full[mask] - dt_full[t]
            pvs[t] = (cmbp_cfs[mask] * (1 + yields_full[mask]) ** (-dtime)).sum()

        # Annual total returns
        cf_T = cmbp_cfs[:T]
        bt = np.empty(T)
        bt[0] = (pvs[0] + cf_T[0]) / cmbp_t0 - 1 if cmbp_t0 > 0 else schedule["fwds"][0]
        for t in range(1, T):
            bt[t] = (pvs[t] + cf_T[t]) / pvs[t - 1] - 1 if pvs[t - 1] > 0 else schedule["fwds"][t]

        return bt

    def run(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> CDISimulationResult:
        """Value assets and simulate CDI waterfall under each scenario in sim_results."""

        # Extract dimensions
        n_sim = sim_results.n_sim
        max_years = sim_results.n_years

        # Pre-calculate all deterministic schedule components (liabilities, rates, buffers, etc.)
        sched = self._calc_deterministic(val_date, rates, max_years)
        T = sched["T"]
        shape = (n_sim, T)

        # Initial asset valuation at t=0 (deterministic): PV of bond portfolio + cash
        bond_prices = self.bonds.pv(pd.Timestamp(val_date), rates, spread_map)
        cdi_t0 = (self.cdi_allocation * bond_prices).sum()
        total_assets_0 = cdi_t0 + self.cash

        # Initial HGB gap at t=0: difference between Heubeck liabilities and available assets (including buffer)
        day0_hgb_gap = self.config.heubeck_liabilities * self.config.mortality_buffer - (total_assets_0 + self.config.asset_buffer)

        # Stochastic bond simulation: CDI portfolio cashflows & PVs
        bond_sim  = self.bonds.run_sim(val_date, rates, spread_map, sim_results)
        asset_cfs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'cashflows')
        asset_pvs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'pvs')

        # Deterministic CMBP returns
        bt = self._calc_bt(val_date, rates, sched)

        # Initial values at t=0
        cash_t = np.full(n_sim, self.cash)
        assets_t = np.full(n_sim, total_assets_0)
        hgb_gap_t = np.full(n_sim, day0_hgb_gap)
        bund_t = np.full(n_sim, total_assets_0)
        cum_hgb_payment = np.zeros(n_sim)

        # Initialize empty arrays to store waterfall components across all scenarios and time steps
        cash_arr = np.zeros(shape)
        assets_arr = np.zeros(shape)
        fee_arr = np.zeros(shape)
        hgb_gap_arr = np.zeros(shape)
        hgb_payment_arr = np.zeros(shape)
        perf_payment_arr = np.zeros(shape)
        additional_pay_arr = np.zeros(shape)
        bund_comparator_arr = np.zeros(shape)
        net_asset_ret = np.zeros(shape)

        for t in range(T):
            # Store previous assets for return calculation before updating
            prev_assets = assets_t.copy()

            # Calculate fee as a percentage of total asset value (cash + cdi cashflow + PV of remaining assets) before liability payments
            fee_t = self.config.fee * (cash_t * (1 + sched["fwds"][t]) + asset_cfs[:, t] + asset_pvs[:, t])

            # Update cash and assets after receiving asset cashflows, paying liabilities, and fees
            cash_t = cash_t * (1 + sched["fwds"][t]) + asset_cfs[:, t] - sched["liab_cashflows"][t] - fee_t

            # Update assets to reflect cash changes and remaining asset PVs
            assets_t = cash_t + asset_pvs[:, t]

            # Calculate HGB gap and payment as difference between meltdown liabilities and assets + buffer
            hgb_gap_t = np.clip(sched["meltdown_liabs"][t] - (assets_t + sched["asset_buffer"][t]), 0.0, hgb_gap_t)

            # Pay shortfall up to HGB gap if assets go below the next-2-year liabilities
            hgb_pay_t = np.clip(sched["next2"][t] - assets_t, 0.0, hgb_gap_t - cum_hgb_payment)
            cash_t += hgb_pay_t
            assets_t += hgb_pay_t
            cum_hgb_payment += hgb_pay_t

            # Additional payment from client to bring GAAP funding level up to 100% or IFRS up to 110%, capped by year 10 asset buffer.
            add_t = 0.0
            if t == self.config.additional_payment_year - 1:
                add_t = np.clip(
                    np.minimum(sched["liab_pv_gaap"][t] - assets_t, 1.1 * sched["liab_pv_ifrs"][t] - assets_t),
                    0.0, sched["asset_buffer"][t],
                )
                cash_t += add_t
                assets_t += add_t

            # Update bund comparator: apply bt return minus liabilities plus any additional payment
            bund_t = bund_t * (1 + bt[t] + self.config.cmbp_margin) - sched["liab_cashflows"][t] + add_t

            # Performance payment at performance_payment_year if assets underperform the bund comparator, capped by the performance cap
            perf_t = (
                np.clip(bund_t - assets_t, 0.0, self.config.performance_cap)
                if t == self.config.performance_payment_year - 1 else 0.0
            )

            # Calculate net asset return for analysis
            net_ret_t = (assets_t + sched["liab_cashflows"][t] - add_t) / prev_assets - 1.0

            # Store results for this time step
            fee_arr[:, t] = fee_t
            cash_arr[:, t] = cash_t
            assets_arr[:, t] = assets_t
            hgb_gap_arr[:, t] = hgb_gap_t
            hgb_payment_arr[:, t] = hgb_pay_t
            perf_payment_arr[:, t] = perf_t
            additional_pay_arr[:, t] = add_t
            bund_comparator_arr[:, t] = bund_t
            net_asset_ret[:, t] = net_ret_t

        # Output DataFrame with MultiIndex (scenario, date) and all relevant columns
        tile = lambda x: np.tile(x, (n_sim, 1))
        results = {
            "dt": tile(sched["dt"]),
            "liab_cashflow": tile(sched["liab_cashflows"]),
            "liab_pv_gaap": tile(sched["liab_pv_gaap"]),
            "liab_pv_ifrs": tile(sched["liab_pv_ifrs"]),
            "meltdown_liabilities": tile(sched["meltdown_liabs"]),
            "next_2_liabs": tile(sched["next2"]),
            "bund_fwds": tile(sched["fwds"]),
            "bund_yield": tile(sched["yields"]),
            "expected_cdi_cashflow": tile(sched["exp_cdi_cf"]),
            "bt": tile(bt),
            "asset_cashflow": asset_cfs,
            "remaining_asset_pv": asset_pvs,
            "fee": fee_arr,
            "cash": cash_arr,
            "assets": assets_arr,
            "net_asset_return": net_asset_ret,
            "hgb_gap": hgb_gap_arr,
            "hgb_payment": hgb_payment_arr,
            "bund_comparator": bund_comparator_arr,
            "performance_payment": perf_payment_arr,
            "additional_payment": additional_pay_arr,
        }
        index = pd.MultiIndex.from_product([np.arange(n_sim), sched["date"]], names=["scenario", "date"])
        df = pd.DataFrame({k: v.reshape(-1) for k, v in results.items()}, index=index).reset_index()

        # Additional calculated columns
        df["funding_level_gaap"] = df["assets"] / df["liab_pv_gaap"]
        df["funding_level_ifrs"] = df["assets"] / df["liab_pv_ifrs"]
        df["net_bt_return"] = df["bt"] + self.config.cmbp_margin

        return CDISimulationResult(config=self.config, schedule = sched, bond_sim=bond_sim, cdi_sim=df, sim_results=sim_results)

    def run_icaap(
            self,
            val_date: datetime,
            rates: Rates,
            spread_map: dict,
            cr_model: CreditRiskModel,
            n_outer: int = 100,
            n_inner: int = 100,
            n_years: int = 25,
            chunk_size: int = 25
        ) -> ICAAPResult:
            """
            Nested Monte-Carlo for the 1-year distribution of PV(obligations).
            Outer simulation: n_outer 1-year credit scenarios → fund state at t=1.
            Inner simulation: from each outer state, n_inner paths over the
                remaining mandate life → conditional E[PV(obligations)].
            Returns:
            - ICAAPResult: Summary statistics (pv_hgb_t1, pv_perf_t1, etc.)
            - DataFrame: Detailed inner waterfall results with MultiIndex (outer_sim, inner_sim, date)
            """

            # Extract dimensions
            n_years_inner = n_years - 1
            cfg = self.config

            # Get determistic parameters for inner simulation starting from t=1 for each outer scenario.
            sched = self._calc_deterministic(val_date, rates, n_years)
            bt = self._calc_bt(val_date, rates, sched)
            T_inner = sched["T"] - 1
            inner_dates = sched["date"][1:]
            inner_val_date = sched["date"][0]
            inner_fwds = sched["fwds"][1:]
            inner_yields = sched["yields"][1:]
            inner_liab_cfs = sched["liab_cashflows"][1:]
            inner_meltdown = sched["meltdown_liabs"][1:]
            inner_next2 = sched["next2"][1:]
            inner_liab_pv_gaap = sched["liab_pv_gaap"][1:]
            inner_liab_pv_ifrs = sched["liab_pv_ifrs"][1:]
            inner_asset_buffer = sched["asset_buffer"][1:]
            inner_dt = sched["dt"][1:] - sched["dt"][0]
            inner_df = (1 + inner_yields) ** (-inner_dt)

            # Run outer cdi simulation for year 1
            outer_cr = cr_model.run(n_outer, 1)
            outer_cdi = self.run(val_date, rates, spread_map, outer_cr)

            # Run risk model for each scenario of issuer ratings at t=1.
            sector_map = self.bonds.issuers.sectors
            n_issuers = len(self.bonds.issuers.ids)

            year1_ratings = outer_cr.transitions.query("year == 1").values

            N_all = n_outer * n_inner

            asset_cfs = np.zeros((N_all, T_inner))
            asset_pvs = np.zeros((N_all, T_inner))
            for c_start in range(0, n_inner, chunk_size):
                c_end = min(c_start + chunk_size, n_inner)
                n_c = c_end - c_start
                N_c = n_c * n_outer

                print(f"Processing {n_c} scenarios {c_start} to {c_end} of {n_inner}")

                year1_ratings_rep = np.repeat(year1_ratings, n_c, axis=0)
                # print(year1_ratings_rep.shape)
                inner_sim_c = cr_model.run_matrix(N_c, n_years_inner, n_issuers, sector_map, year1_ratings_rep)
                # Inner bond sim
                inner_bond_sim_c = self.bonds.run_sim(inner_val_date, rates, spread_map, inner_sim_c)

                # Fix double-recovery for bonds already defaulted in the outer year
                outer_bond_ratings = outer_cdi.bond_sim.transitions[:, 0, :]
                already_defaulted = (outer_bond_ratings == DEFAULT_LABEL)
                already_def_all = np.repeat(already_defaulted, n_c, axis=0)
                inner_bond_sim_c.recovery_payments[:, 0, :] *= ~already_def_all
                inner_bond_sim_c.total_cashflows = inner_bond_sim_c.received_cashflows + inner_bond_sim_c.recovery_payments

                # get cashflows and pvs for all scenarios
                shape_c = (N_c, T_inner)
                asset_cfs_c = allocate_bond_sim(inner_bond_sim_c, self.cdi_allocation, shape_c, "cashflows")
                asset_pvs_c = allocate_bond_sim(inner_bond_sim_c, self.cdi_allocation, shape_c, "pvs")

                asset_cfs[c_start*n_outer:c_end*n_outer, :] = asset_cfs_c
                asset_pvs[c_start*n_outer:c_end*n_outer, :] = asset_pvs_c

                del inner_bond_sim_c, inner_sim_c, year1_ratings_rep, already_def_all, already_defaulted

            # Free up memory
            del  year1_ratings, outer_cr, outer_bond_ratings

            # ── Inner Waterfall ──
            # get results of outer sim to use as starting points for inner sim
            df1 = outer_cdi.cdi_sim
            cash_outer = df1["cash"].values
            hgb_gap_outer = df1["hgb_gap"].values
            cum_hgb_outer = df1["hgb_payment"].values
            bund_outer = df1["bund_comparator"].values
            assets_outer = df1["assets"].values

            cash_t = np.repeat(np.asarray(cash_outer.astype(float)), n_inner)
            assets_t = np.repeat(np.asarray(assets_outer.astype(float)), n_inner)
            hgb_gap_t = np.repeat(np.asarray(hgb_gap_outer.astype(float)), n_inner)
            cum_hgb = np.repeat(np.asarray(cum_hgb_outer.astype(float)), n_inner)
            bund_t = np.repeat(np.asarray(bund_outer.astype(float)), n_inner)

            shape_all = (N_all, T_inner)
            # Empty arrays to store results
            hgb_arr = np.zeros(shape_all)
            cash_arr = np.zeros(shape_all)
            assets_arr = np.zeros(shape_all)
            fee_arr = np.zeros(shape_all)
            hgb_gap_arr = np.zeros(shape_all)
            hgb_payment_arr = np.zeros(shape_all)
            perf_payment_arr_c = np.zeros(shape_all)
            additional_pay_arr = np.zeros(shape_all)
            bund_comparator_arr = np.zeros(shape_all)
            net_asset_ret_arr = np.zeros(shape_all)
            perf_payment_arr = np.zeros(N_all)

            # year in which performance payment is made
            perf_inner_t = cfg.performance_payment_year - 2

            # Run waterfall for each time step
            for t in range(T_inner):
                # Store previous assets for return calculation before updating
                prev_assets_t = assets_t.copy()

                # Calculate fee as a percentage of total asset value (cash + cdi cashflow + PV of remaining assets) before liability payments
                fee_t = cfg.fee * (cash_t * (1 + inner_fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])

                # Update cash and assets after receiving asset cashflows, paying liabilities, and fees
                cash_t = cash_t * (1 + inner_fwds[t]) + asset_cfs[:, t] - inner_liab_cfs[t] - fee_t

                # Update assets to reflect cash changes and remaining asset PVs
                assets_t = cash_t + asset_pvs[:, t]

                # Calculate HGB gap and payment as difference between meltdown liabilities and assets + buffer
                hgb_gap_t = np.clip(inner_meltdown[t] - (assets_t + inner_asset_buffer[t]), 0.0, hgb_gap_t)

                # Pay shortfall up to HGB gap if assets go below the next-2-year liabilities
                hgb_pay_t = np.clip(inner_next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb)
                cash_t += hgb_pay_t
                assets_t += hgb_pay_t
                cum_hgb += hgb_pay_t

                # Additional payment from client at year 10 to bring GAAP funding level up to 100% or IFRS up to 110%, capped by asset buffer.
                add_t = 0.0
                if t == cfg.additional_payment_year - 2:
                    add_t = np.clip(
                        np.minimum(inner_liab_pv_gaap[t] - assets_t, 1.1 * inner_liab_pv_ifrs[t] - assets_t),
                        0.0,
                        inner_asset_buffer[t],
                    )
                    cash_t += add_t
                    assets_t += add_t

                # Bund comparator evolution
                bund_t = bund_t * (1 + bt[t + 1] + cfg.cmbp_margin) - inner_liab_cfs[t] + add_t

                # Performance payment at terminal year
                if t == perf_inner_t:
                    perf_payment_arr = np.clip(bund_t - assets_t, 0.0, cfg.performance_cap)
                else:
                    perf_payment_arr = np.zeros(N_all)

                # Net asset return for analysis
                net_ret_t = (assets_t + inner_liab_cfs[t] - add_t) / prev_assets_t - 1.0

                # Store arrays
                fee_arr[:, t] = fee_t
                cash_arr[:, t] = cash_t
                assets_arr[:, t] = assets_t
                hgb_arr[:, t] = hgb_pay_t
                hgb_gap_arr[:, t] = hgb_gap_t
                hgb_payment_arr[:, t] = hgb_pay_t
                perf_payment_arr_c[:, t] = perf_payment_arr
                additional_pay_arr[:, t] = add_t
                bund_comparator_arr[:, t] = bund_t
                net_asset_ret_arr[:, t] = net_ret_t

            # Discounted Performance Payment and HGB payment for all paths
            pv_perf_path = perf_payment_arr * inner_df[perf_inner_t] if (0 <= perf_inner_t <= T_inner) else np.zeros(N_all)
            pv_hgb_path = (hgb_arr * inner_df[np.newaxis, :]).sum(axis=1)
            pv_fee_path = (fee_arr * inner_df[np.newaxis, :]).sum(axis=1)

            # Conditional distribution of HGB PV and performance payment PV
            pv_hgb_cond = pv_hgb_path.reshape(n_outer, n_inner).mean(axis=1)
            pv_perf_cond = pv_perf_path.reshape(n_outer, n_inner).mean(axis=1)
            pv_fee_cond = pv_fee_path.reshape(n_outer, n_inner).mean(axis=1)

            # Build detailed DataFrame (all results, no chunking)
            tile = lambda x: np.tile(x, (N_all, 1))
            results = {
                "dt": tile(inner_dt),
                "liab_cashflow": tile(inner_liab_cfs),
                "liab_pv_gaap": tile(inner_liab_pv_gaap),
                "liab_pv_ifrs": tile(inner_liab_pv_ifrs),
                "meltdown_liabilities": tile(inner_meltdown),
                "next_2_liabs": tile(inner_next2),
                "bund_fwds": tile(inner_fwds),
                "bund_yield": tile(inner_yields),
                'remaining_asset_pv': asset_pvs,
                'asset_cashflow': asset_cfs,
                "fee": fee_arr,
                "cash": cash_arr,
                "assets": assets_arr,
                "net_asset_return": net_asset_ret_arr,
                "hgb_gap": hgb_gap_arr,
                "hgb_payment": hgb_payment_arr,
                "bund_comparator": bund_comparator_arr,
                "performance_payment": perf_payment_arr_c,
                "additional_payment": additional_pay_arr,
            }

            # Create MultiIndex for DataFrame: (outer_sim, inner_sim, date)
            outer_sims = np.repeat(np.arange(n_outer), n_inner * T_inner)
            inner_sims = np.tile(np.repeat(np.arange(n_inner), T_inner), n_outer)
            dates_tiled = np.tile(inner_dates, N_all)
            index = pd.MultiIndex.from_arrays(
                [outer_sims, inner_sims, dates_tiled],
                names=["outer_sim", "inner_sim", "date"]
            )

            # Output dataframe
            icaap_df = pd.DataFrame(
                {k: v.reshape(-1) for k, v in results.items()},
                index=index
            ).reset_index()

            # Additional calculated columns
            icaap_df["funding_level_gaap"] = icaap_df["assets"] / icaap_df["liab_pv_gaap"]
            icaap_df["funding_level_ifrs"] = icaap_df["assets"] / icaap_df["liab_pv_ifrs"]
            icaap_df["net_bt_return"] = icaap_df["bund_yield"] + cfg.cmbp_margin

            return ICAAPResult(
                pv_hgb_t1=pv_hgb_cond,
                pv_perf_t1=pv_perf_cond,
                pv_fee_t1=pv_fee_cond,
                icaap_df=icaap_df
            )
