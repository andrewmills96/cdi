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

    def transitions_from_state(self, pX: np.ndarray, initial_indices: np.ndarray) -> np.ndarray:
        """
        Same as *transitions* but with per-simulation initial ratings.
        initial_indices: (n_sim, n_issuers) int array.
        """
        n_sim, n_years, n_issuers = pX.shape

        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial_indices

        for t in range(n_years):
            cum = self.cum_tmatrix[res[:, t, :]]
            pX_t = pX[:, t, :, np.newaxis]
            res[:, t + 1, :] = np.clip(
                np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1
            ).astype(np.int32)

        return self.indices_to_labels(res)

    def fundamental_matrix(self):
        Q = self.tmatrix[:-1, :-1]
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        labels = self.labels[:-1]
        return pd.DataFrame(N, index=labels, columns=labels)

    def time_to_default(self):
        N = self.fundamental_matrix().values
        t = N @ np.ones(len(N))
        return pd.Series(t, index=self.labels[:-1], name='Time to Default')

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt=".1%"))


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

    def run(self, n_sim: int, n_years: int) -> "SimulationResult":
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
        return SimulationResult(E, S, I, X, pX, transitions, n_sim, n_years)


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
                 cashflows: pd.DataFrame, issuers: Issuers,
                 descriptions: pd.Series | None = None):
        self.ids = ids
        self.issuer_ids = issuer_ids
        self.recoveries = recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        self.descriptions = descriptions
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

    def _run_sim_pv(self, cashflows: np.ndarray, bond_transitions: np.ndarray,
                    spread_map: dict, yields: np.ndarray, dt: np.ndarray) -> np.ndarray:
        n_years_cf, n_bonds = cashflows.shape
        n_sim, n_years_sim, _ = bond_transitions.shape
        n_years = min(n_years_cf, n_years_sim)
        assert cashflows.shape[1] == bond_transitions.shape[2]

        unique_spreads, spread_inverse = np.unique(
            map_spreads(bond_transitions, spread_map), return_inverse=True
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

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict,
                sim_results: SimulationResult) -> BondSimulationResult:
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


class _Schedule:
    """Pre-computed deterministic arrays used by the CDI waterfall."""
    liab_cashflows: np.ndarray
    dates: pd.DatetimeIndex
    T: int
    dt: np.ndarray
    liab_pv_gaap: np.ndarray
    liab_pv_ifrs: np.ndarray
    meltdown_liabs: np.ndarray
    next2: np.ndarray
    fwds: np.ndarray
    yields: np.ndarray
    asset_buffer: np.ndarray
    exp_cdi_cf: np.ndarray


# ---------------------------------------------------------------------------
#  CDI result containers
# ---------------------------------------------------------------------------

class CDISimulationResult:
    """Output container for CDIMandate_Fox.run()."""

    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, config: "FoxConfig", bond_sim: BondSimulationResult, cdi_sim: pd.DataFrame):
        self.config = config
        self.bond_sim = bond_sim
        self.cdi_sim = cdi_sim

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


class ICAAPResult:
    """Container for ICAAP nested Monte-Carlo capital results."""

    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, pv_hgb_t1, pv_perf_t1, pv0, df_1y, n_outer, n_inner, icaap_df: Optional[pd.DataFrame] = None):
        self.pv_hgb_t1  = pv_hgb_t1
        self.pv_perf_t1 = pv_perf_t1
        self.pv0  = pv0
        self.df_1y = df_1y
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.icaap_df = icaap_df

    @property
    def pv_total_t1(self):
        return self.pv_hgb_t1 + self.pv_perf_t1

    @property
    def pv_total_discounted(self):
        return self.pv_total_t1 * self.df_1y

    @property
    def delta_pv(self):
        return self.pv_total_discounted - self.pv0

    def var(self, confidence: float = 0.995) -> float:
        return float(np.quantile(self.delta_pv, confidence))

    def tvar(self, confidence: float = 0.995) -> float:
        d = self.delta_pv
        threshold = np.quantile(d, confidence)
        tail = d[d >= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(threshold)

    def summary(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        qs = quantiles or self._DEFAULT_QUANTILES
        scenarios = pd.DataFrame({
            "pv_hgb_t1":           self.pv_hgb_t1,
            "pv_perf_t1":          self.pv_perf_t1,
            "pv_total_t1":         self.pv_total_t1,
            "pv_total_discounted": self.pv_total_discounted,
            "delta_pv":            self.delta_pv,
        })
        tbl = scenarios.quantile(qs).T
        tbl.insert(0, "mean", scenarios.mean())

        cap = pd.DataFrame(
            {"mean": [self.var(), self.tvar(), self.pv0]},
            index=["capital_var_995", "capital_tvar_995", "pv0_baseline"],
        )
        for q in qs:
            cap[q] = np.nan
        return pd.concat([tbl, cap])


# ---------------------------------------------------------------------------
#  Fox mandate configuration & simulation
# ---------------------------------------------------------------------------

class FoxConfig:
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
        performance_payment_year: int = 25,
    ):
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

    # ------------------------------------------------------------------
    #  Shared helpers
    # ------------------------------------------------------------------

    def _build_schedule(self, val_date: datetime, rates: Rates, n_years: int) -> _Schedule:
        """Pre-compute all deterministic arrays needed by the waterfall."""
        s = _Schedule()
        cfg = self.config

        # Liability series
        liab_series = self.liabilities.cashflows.loc[val_date:][:n_years]
        s.liab_cashflows = np.asarray(liab_series.values, dtype=float)
        s.dates = pd.DatetimeIndex(liab_series.index)
        s.T = len(s.dates)
        s.dt = calc_dt(s.dates, val_date)

        # Liability PV timelines
        s.liab_pv_gaap = np.asarray(self.liabilities.pv(val_date, cfg.r_gaap, timeline=True, n_years=n_years), dtype=float)
        s.liab_pv_ifrs = np.asarray(self.liabilities.pv(val_date, cfg.r_ifrs, timeline=True, n_years=n_years), dtype=float)

        # Meltdown liabilities
        cum_liab = s.liab_cashflows.cumsum()
        s.meltdown_liabs = np.maximum(
            0.0,
            (cfg.heubeck_liabilities - cum_liab) * cfg.mortality_buffer * ((1 + cfg.r_gaap) ** s.dt),
        )

        # Next-2-year liabilities
        full_cfs = np.asarray(self.liabilities.cashflows.loc[val_date:].values, dtype=float)
        if len(full_cfs) < s.T + 2:
            raise ValueError(f"Liability cashflows too short: need {s.T + 2} from val_date, got {len(full_cfs)}.")
        s.next2 = full_cfs[1 : s.T + 1] + full_cfs[2 : s.T + 2]

        # Rates
        s.fwds   = np.asarray(rates.calc_fwds(val_date, s.dates).values, dtype=float)
        s.yields = np.asarray(rates.interpolate(s.dates).values, dtype=float)

        # Asset buffer (accrues at IFRS rate for first 10 years, then 0)
        s.asset_buffer = np.array([
            cfg.asset_buffer * ((1 + cfg.r_ifrs) ** (t + 1)) if t < 10 else 0.0
            for t in range(s.T)
        ])

        # Expected (no-default) CDI cashflows
        raw = self.bonds.year_end_cashflows(val_date)
        exp = np.asarray((raw[self.cdi_allocation.index] * self.cdi_allocation).sum(axis=1).values, dtype=float)
        s.exp_cdi_cf = np.pad(exp, (0, max(0, s.T - len(exp))))[:s.T]

        return s

    def _deterministic_bt(self, val_date: datetime, rates: Rates,
                          schedule: _Schedule) -> tuple[np.ndarray, float]:
        """
        CMBP total returns are deterministic because all CMBP bonds carry the
        AAAA rating, which is absorbing (no migration, no default).
        Returns (bt, cmbp_t0) where bt has shape (T,).
        """
        raw_cfs = self.bonds.year_end_cashflows(val_date)
        cmbp_ids = self.cmbp_allocation.index

        # Portfolio cashflows per year (full horizon, may extend beyond T)
        cmbp_cfs = np.asarray(
            (raw_cfs[cmbp_ids] * self.cmbp_allocation).sum(axis=1).values, dtype=float
        )

        dates_full = pd.DatetimeIndex(raw_cfs.index)
        dt_full    = calc_dt(dates_full, val_date)
        yields_full = np.asarray(rates.interpolate(dates_full).values, dtype=float)
        n_full = len(dates_full)
        T = schedule.T

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
        bt[0] = (pvs[0] + cf_T[0]) / cmbp_t0 - 1 if cmbp_t0 > 0 else schedule.fwds[0]
        for t in range(1, T):
            bt[t] = (pvs[t] + cf_T[t]) / pvs[t - 1] - 1 if pvs[t - 1] > 0 else schedule.fwds[t]

        return bt, cmbp_t0

    # ------------------------------------------------------------------
    #  Main simulation
    # ------------------------------------------------------------------

    def run(self, val_date: datetime, rates: Rates, spread_map: dict,
            sim_results: SimulationResult) -> CDISimulationResult:
        n_sim = sim_results.n_sim
        max_years = sim_results.n_years
        cfg = self.config

        # Deterministic schedules
        sched = self._build_schedule(val_date, rates, max_years)
        T = sched.T
        shape = (n_sim, T)

        # Deterministic CMBP returns (AAAA bonds never default)
        bt, _ = self._deterministic_bt(val_date, rates, sched)

        # Portfolio market values at val_date
        bond_prices = self.bonds.pv(pd.Timestamp(val_date), rates, spread_map)
        cdi_t0 = (self.cdi_allocation * bond_prices).sum()
        total_assets_0 = cdi_t0 + self.cash
        day0_hgb_gap = cfg.heubeck_liabilities * cfg.mortality_buffer - (total_assets_0 + cfg.asset_buffer)

        # Stochastic bond simulation → CDI portfolio cashflows & PVs
        bond_sim  = self.bonds.run_sim(val_date, rates, spread_map, sim_results)
        asset_cfs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'cashflows')
        asset_pvs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'pvs')

        # ── Waterfall ──
        cash_arr             = np.zeros(shape)
        assets_arr           = np.zeros(shape)
        fee_arr              = np.zeros(shape)
        hgb_gap_arr          = np.zeros(shape)
        hgb_payment_arr      = np.zeros(shape)
        perf_payment_arr     = np.zeros(shape)
        additional_pay_arr   = np.zeros(shape)
        bund_comparator_arr  = np.zeros(shape)
        net_asset_ret        = np.zeros(shape)

        cash_t          = np.full(n_sim, self.cash)
        assets_t        = np.full(n_sim, total_assets_0)
        hgb_gap_t       = np.full(n_sim, day0_hgb_gap)
        bund_t          = np.full(n_sim, total_assets_0)
        cum_hgb_payment = np.zeros(n_sim)

        for t in range(T):
            prev_assets = assets_t.copy()

            fee_t = cfg.fee * (cash_t * (1 + sched.fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])
            cash_t = cash_t * (1 + sched.fwds[t]) + asset_cfs[:, t] - sched.liab_cashflows[t] - fee_t
            assets_t = cash_t + asset_pvs[:, t]

            # HGB gap check & top-up
            hgb_gap_t = np.clip(sched.meltdown_liabs[t] - (assets_t + sched.asset_buffer[t]), 0.0, hgb_gap_t)
            hgb_pay_t = np.clip(sched.next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb_payment)
            cash_t   += hgb_pay_t
            assets_t += hgb_pay_t
            cum_hgb_payment += hgb_pay_t

            # Additional injection at configured year
            add_t = 0.0
            if t == cfg.additional_payment_year - 1:
                add_t = np.clip(
                    np.minimum(sched.liab_pv_gaap[t] - assets_t, 1.1 * sched.liab_pv_ifrs[t] - assets_t),
                    0.0, sched.asset_buffer[t],
                )
                cash_t   += add_t
                assets_t += add_t

            # Bund comparator (bt is deterministic → scalar broadcast)
            bund_t = bund_t * (1 + bt[t] + cfg.cmbp_margin) - sched.liab_cashflows[t] + add_t

            # Performance payment at terminal year
            perf_t = (
                np.clip(bund_t - assets_t, 0.0, cfg.performance_cap)
                if t == cfg.performance_payment_year - 1 else 0.0
            )

            net_ret_t = (assets_t + sched.liab_cashflows[t] - add_t) / prev_assets - 1.0

            fee_arr[:, t]            = fee_t
            cash_arr[:, t]           = cash_t
            assets_arr[:, t]         = assets_t
            hgb_gap_arr[:, t]        = hgb_gap_t
            hgb_payment_arr[:, t]    = hgb_pay_t
            perf_payment_arr[:, t]   = perf_t
            additional_pay_arr[:, t] = add_t
            bund_comparator_arr[:, t] = bund_t
            net_asset_ret[:, t]      = net_ret_t

        # ── Output DataFrame ──
        tile = lambda x: np.tile(x, (n_sim, 1))
        results = {
            "dt":                    tile(sched.dt),
            "liab_cashflow":         tile(sched.liab_cashflows),
            "liab_pv_gaap":          tile(sched.liab_pv_gaap),
            "liab_pv_ifrs":          tile(sched.liab_pv_ifrs),
            "meltdown_liabilities":  tile(sched.meltdown_liabs),
            "next_2_liabs":          tile(sched.next2),
            "bund_fwds":             tile(sched.fwds),
            "bund_yield":            tile(sched.yields),
            "expected_cdi_cashflow": tile(sched.exp_cdi_cf),
            "bt":                    tile(bt),
            "asset_cashflow":        asset_cfs,
            "remaining_asset_pv":    asset_pvs,
            "fee":                   fee_arr,
            "cash":                  cash_arr,
            "assets":                assets_arr,
            "net_asset_return":      net_asset_ret,
            "hgb_gap":              hgb_gap_arr,
            "hgb_payment":           hgb_payment_arr,
            "bund_comparator":       bund_comparator_arr,
            "performance_payment":   perf_payment_arr,
            "additional_payment":    additional_pay_arr,
        }
        index = pd.MultiIndex.from_product([np.arange(n_sim), sched.dates], names=["scenario", "date"])
        df = pd.DataFrame({k: v.reshape(-1) for k, v in results.items()}, index=index).reset_index()

        df["funding_level_gaap"] = df["assets"] / df["liab_pv_gaap"]
        df["funding_level_ifrs"] = df["assets"] / df["liab_pv_ifrs"]
        df["net_bt_return"]      = df["bt"] + cfg.cmbp_margin

        return CDISimulationResult(config=cfg, bond_sim=bond_sim, cdi_sim=df)

    # ------------------------------------------------------------------
    #  ICAAP capital (nested Monte-Carlo)
    # ------------------------------------------------------------------

    def icaap_capital(
        self,
        val_date: datetime,
        rates: Rates,
        spread_map: dict,
        cr_model: CreditRiskModel,
        base_result: CDISimulationResult,
        n_outer: int = 500,
        n_inner: int = 500,
        chunk_size: int = 50,
    ) -> ICAAPResult:
        """
        Nested Monte-Carlo for the 1-year distribution of PV(obligations).

        Outer simulation: n_outer 1-year credit scenarios → fund state at t=1.
        Inner simulation: from each outer state, n_inner paths over the
            remaining mandate life → conditional E[PV(obligations)].
        The distribution across outer scenarios gives VaR / TVaR.

        Returns:
        - ICAAPResult: Summary statistics (pv_hgb_t1, pv_perf_t1, etc.)
        - DataFrame: Detailed inner waterfall results with MultiIndex (outer_sim, inner_sim, date)
        """
        n_years = base_result.bond_sim.total_cashflows.shape[1]
        if n_years < 2:
            raise ValueError("ICAAP requires n_years >= 2.")
        n_years_inner = n_years - 1
        cfg = self.config

        # ── Deterministic schedules ──
        sched = self._build_schedule(val_date, rates, n_years)
        T = sched.T
        bt, _ = self._deterministic_bt(val_date, rates, sched)

        # Inner-year slices (year 2+ of the waterfall)
        T_inner            = T - 1
        inner_fwds         = sched.fwds[1:]
        inner_yields       = sched.yields[1:]
        inner_liab_cfs     = sched.liab_cashflows[1:]
        inner_meltdown     = sched.meltdown_liabs[1:]
        inner_next2        = sched.next2[1:]
        inner_liab_pv_gaap = sched.liab_pv_gaap[1:]
        inner_liab_pv_ifrs = sched.liab_pv_ifrs[1:]
        inner_asset_buffer = sched.asset_buffer[1:]
        inner_dt           = sched.dt[1:] - sched.dt[0]
        inner_df           = (1 + inner_yields) ** (-inner_dt)

        perf_inner_t = cfg.performance_payment_year - 2
        has_perf     = 0 <= perf_inner_t < T_inner

        # ── Outer 1-year simulation ──
        outer_cr  = cr_model.run(n_outer, 1)
        outer_cdi = self.run(val_date, rates, spread_map, outer_cr)

        df1 = outer_cdi.cdi_sim
        cash_outer    = df1["cash"].values
        hgb_gap_outer = df1["hgb_gap"].values
        cum_hgb_outer = df1["hgb_payment"].values
        bund_outer    = df1["bund_comparator"].values

        # Issuer ratings at t=1
        year1_mask    = outer_cr.transitions.index.get_level_values("year") == 1
        year1_ratings = outer_cr.transitions.loc[year1_mask].values

        outer_bond_ratings = outer_cdi.bond_sim.transitions[:, 0, :]
        already_defaulted  = (outer_bond_ratings == DEFAULT_LABEL)

        tm = cr_model.transition_matrix
        initial_indices_all = np.array([
            [tm.label_to_idx[r] for r in year1_ratings[s]]
            for s in range(n_outer)
        ])

        # ── Factor-model coefficients (constant across chunks) ──
        n_issuers    = len(cr_model.issuer_ids)
        n_sectors    = cr_model.sector_map.nunique()
        rho_s_i      = cr_model.rho_s[cr_model.sector_map]
        inner_val_date = sched.dates[0]

        sqrt_rho_e   = np.sqrt(cr_model.rho_e)
        sqrt_rho_s_e = np.sqrt(rho_s_i - cr_model.rho_e)
        sqrt_1_rho_s = np.sqrt(1 - rho_s_i)

        pv_hgb_cond  = np.empty(n_outer)
        pv_perf_cond = np.empty(n_outer)

        # Accumulate detailed results
        all_results = []

        # ── Chunked inner simulation ──
        for c_start in range(0, n_outer, chunk_size):
            c_end = min(c_start + chunk_size, n_outer)
            n_c   = c_end - c_start
            N_c   = n_c * n_inner

            # Credit draws
            E       = np.random.normal(size=(N_c, n_years_inner))
            S       = np.random.normal(size=(N_c, n_years_inner, n_sectors))
            I_noise = np.random.normal(size=(N_c, n_years_inner, n_issuers))

            X = (
                sqrt_rho_e * E[:, :, np.newaxis]
                + sqrt_rho_s_e[np.newaxis, np.newaxis, :] * S[:, :, cr_model.sector_map]
                + sqrt_1_rho_s[np.newaxis, np.newaxis, :] * I_noise
            )
            pX = sp.stats.norm.cdf(X)

            init_chunk      = np.repeat(initial_indices_all[c_start:c_end], n_inner, axis=0)
            inner_trans_arr = tm.transitions_from_state(pX, init_chunk)

            inner_trans_df = pd.DataFrame(
                inner_trans_arr.reshape(N_c * (n_years_inner + 1), n_issuers),
                index=pd.MultiIndex.from_product(
                    [range(N_c), range(n_years_inner + 1)], names=["sim", "year"]
                ),
                columns=cr_model.issuer_ids,
            )
            inner_sim = SimulationResult(E, S, I_noise, X, pX, inner_trans_df, N_c, n_years_inner)

            # Inner bond sim (CDI allocation only)
            inner_bond_sim = self.bonds.run_sim(inner_val_date, rates, spread_map, inner_sim)

            # Fix double-recovery for bonds already defaulted in the outer year
            already_def_c = np.repeat(already_defaulted[c_start:c_end], n_inner, axis=0)
            inner_bond_sim.recovery_payments[:, 0, :] *= ~already_def_c
            inner_bond_sim.total_cashflows = inner_bond_sim.received_cashflows + inner_bond_sim.recovery_payments

            shape_c   = (N_c, T_inner)
            asset_cfs = allocate_bond_sim(inner_bond_sim, self.cdi_allocation, shape_c, "cashflows")
            asset_pvs = allocate_bond_sim(inner_bond_sim, self.cdi_allocation, shape_c, "pvs")

            del inner_bond_sim, inner_trans_arr, inner_trans_df, inner_sim
            del E, S, I_noise, X, pX

            # ── Inner waterfall ──
            cash_t    = np.repeat(np.asarray(cash_outer[c_start:c_end].astype(float)),    n_inner)
            hgb_gap_t = np.repeat(np.asarray(hgb_gap_outer[c_start:c_end].astype(float)), n_inner)
            cum_hgb   = np.repeat(np.asarray(cum_hgb_outer[c_start:c_end].astype(float)), n_inner)
            bund_t    = np.repeat(np.asarray(bund_outer[c_start:c_end].astype(float)), n_inner)

            # Collect waterfall arrays for this chunk
            hgb_arr              = np.zeros(shape_c)
            cash_arr_c           = np.zeros(shape_c)
            assets_arr_c         = np.zeros(shape_c)
            fee_arr_c            = np.zeros(shape_c)
            hgb_gap_arr_c        = np.zeros(shape_c)
            hgb_payment_arr_c    = np.zeros(shape_c)
            perf_payment_arr_c   = np.zeros(shape_c)
            additional_pay_arr_c = np.zeros(shape_c)
            bund_comparator_arr_c = np.zeros(shape_c)
            net_asset_ret_c      = np.zeros(shape_c)
            perf_payment_arr     = np.zeros(N_c)

            for t in range(T_inner):
                orig_t = t + 1
                prev_cash_t = cash_t.copy()

                fee_t = cfg.fee * (cash_t * (1 + inner_fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])
                cash_t = cash_t * (1 + inner_fwds[t]) + asset_cfs[:, t] - inner_liab_cfs[t] - fee_t
                assets_t = cash_t + asset_pvs[:, t]

                hgb_gap_t = np.clip(inner_meltdown[t] - (assets_t + inner_asset_buffer[t]), 0.0, hgb_gap_t)
                hgb_pay_t = np.clip(inner_next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb)
                cash_t   += hgb_pay_t
                assets_t += hgb_pay_t
                cum_hgb  += hgb_pay_t

                add_t = 0.0
                if orig_t == cfg.additional_payment_year - 1:
                    add_t = np.clip(
                        np.minimum(inner_liab_pv_gaap[t] - assets_t, 1.1 * inner_liab_pv_ifrs[t] - assets_t),
                        0.0, inner_asset_buffer[t],
                    )
                    cash_t   += add_t
                    assets_t += add_t

                # Bund comparator evolution
                bund_t = bund_t * (1 + bt[t + 1] + cfg.cmbp_margin) - inner_liab_cfs[t] + add_t

                # Performance payment at terminal year
                if t == perf_inner_t:
                    perf_payment_arr = np.clip(bund_t - assets_t, 0.0, cfg.performance_cap)

                net_ret_t = (assets_t + inner_liab_cfs[t] - add_t) / prev_cash_t - 1.0

                # Store arrays
                fee_arr_c[:, t]             = fee_t
                cash_arr_c[:, t]            = cash_t
                assets_arr_c[:, t]          = assets_t
                hgb_gap_arr_c[:, t]         = hgb_gap_t
                hgb_payment_arr_c[:, t]     = hgb_pay_t
                perf_payment_arr_c[:, t]    = (perf_payment_arr if t == perf_inner_t else 0.0)
                additional_pay_arr_c[:, t]  = add_t
                bund_comparator_arr_c[:, t] = bund_t
                net_asset_ret_c[:, t]       = net_ret_t

            # Aggregate performance payment PV for this chunk
            if has_perf:
                pv_perf_path = perf_payment_arr * inner_df[perf_inner_t]
            else:
                pv_perf_path = np.zeros(N_c)

            pv_hgb_path = (hgb_arr * inner_df[np.newaxis, :]).sum(axis=1)

            # Store outer scenario aggregates
            pv_hgb_cond[c_start:c_end]  = pv_hgb_path.reshape(n_c, n_inner).mean(axis=1)
            pv_perf_cond[c_start:c_end] = pv_perf_path.reshape(n_c, n_inner).mean(axis=1)

            # Build detailed DataFrame for this chunk
            tile = lambda x: np.tile(x, (N_c, 1))
            chunk_results = {
                "dt":                    tile(inner_dt),
                "liab_cashflow":         tile(inner_liab_cfs),
                "liab_pv_gaap":          tile(inner_liab_pv_gaap),
                "liab_pv_ifrs":          tile(inner_liab_pv_ifrs),
                "meltdown_liabilities":  tile(inner_meltdown),
                "next_2_liabs":          tile(inner_next2),
                "bund_fwds":             tile(inner_fwds),
                "bund_yield":            tile(inner_yields),
                "fee":                   fee_arr_c,
                "cash":                  cash_arr_c,
                "assets":                assets_arr_c,
                "net_asset_return":      net_asset_ret_c,
                "hgb_gap":               hgb_gap_arr_c,
                "hgb_payment":           hgb_payment_arr_c,
                "bund_comparator":       bund_comparator_arr_c,
                "performance_payment":   perf_payment_arr_c,
                "additional_payment":    additional_pay_arr_c,
            }
            
            # MultiIndex: (outer_sim, inner_sim, date)
            outer_sims = np.repeat(np.arange(c_start, c_end), n_inner * T_inner)
            inner_sims = np.tile(np.repeat(np.arange(n_inner), T_inner), n_c)
            dates_tiled = np.tile(sched.dates[1:], N_c)
            
            index = pd.MultiIndex.from_arrays(
                [outer_sims, inner_sims, dates_tiled],
                names=["outer_sim", "inner_sim", "date"]
            )
            
            chunk_df = pd.DataFrame(
                {k: v.reshape(-1) for k, v in chunk_results.items()},
                index=index
            ).reset_index()
            
            all_results.append(chunk_df)

        # Combine all chunks
        icaap_df = pd.concat(all_results, ignore_index=True)
        
        icaap_df["funding_level_gaap"] = icaap_df["assets"] / icaap_df["liab_pv_gaap"]
        icaap_df["funding_level_ifrs"] = icaap_df["assets"] / icaap_df["liab_pv_ifrs"]
        icaap_df["net_bt_return"]      = icaap_df["bund_yield"] + cfg.cmbp_margin

        # ── Result ──
        df_1y = float((1 + sched.yields[0]) ** (-sched.dt[0]))
        pv0   = float(base_result.pv_total_obligations().mean())

        icaap_summary = ICAAPResult(
            pv_hgb_t1=pv_hgb_cond,
            pv_perf_t1=pv_perf_cond,
            pv0=pv0,
            df_1y=df_1y,
            n_outer=n_outer,
            n_inner=n_inner,
            icaap_df=icaap_df,
        )

        return icaap_summary