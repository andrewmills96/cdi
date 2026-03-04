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
    if isinstance(dates, pd.DatetimeIndex):
        delta_days = (dates - pd.Timestamp(val_date)).days
    else:
        delta_days = (pd.to_datetime(dates) - pd.Timestamp(val_date)).days

    return np.asarray(delta_days, dtype=float) / DAYS_PER_YEAR

def fit_to_shape(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Trim and pad an array to shape with zeros."""
    if len(shape) == 3 and a.ndim == 2:
        a = np.expand_dims(a, axis=-1)
    trimmed = a[tuple(slice(0, min(a.shape[i], shape[i])) for i in range(len(shape)))]
    pad_widths = [(0, max(0, shape[i] - trimmed.shape[i])) for i in range(len(shape))]
    while len(pad_widths) < trimmed.ndim:
        pad_widths.append((0, 0))
    return np.pad(trimmed, pad_widths, mode='constant')

def total_returns(pvs: np.ndarray, cashflows: np.ndarray, fwds: np.ndarray, a0: float) -> np.ndarray:
    """Calculates the annual total return for a portfolio of bonds"""
    returns = np.empty((pvs.shape[0], pvs.shape[1]), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        returns[:, 0] = (pvs[:, 0] + cashflows[:, 0]) / float(a0) - 1
        returns[:, 1:] = (pvs[:, 1:] + cashflows[:, 1:]) / pvs[:, :-1] - 1

    returns = np.where(np.isfinite(returns), returns, fwds)
    return returns

def map_spreads(ratings: np.ndarray, spread_map: Dict[str, float]) -> np.ndarray:
    """Vectorized mapping of rating label arrays to spread floats."""
    flat = ratings.ravel()
    codes, uniques = pd.factorize(flat, sort=False)
    mapped = np.fromiter((spread_map.get(u, 0.0) for u in uniques), dtype=float, count=len(uniques))
    return mapped[codes].reshape(ratings.shape)

def compute_maturity_flags(cashflows: np.ndarray) -> np.ndarray:
    """Returns a boolean mask True for years up to each bond's final cashflow."""
    n_years, n_bonds = cashflows.shape
    flipped = np.flip(cashflows, axis=0)
    has_cashflow = cashflows.any(axis=0)
    last_cf_idx = np.where(has_cashflow, n_years - 1 - np.argmax(flipped != 0, axis=0), -1)
    year_indices = np.arange(n_years)[:, np.newaxis]
    return year_indices <= last_cf_idx[np.newaxis, :]


# Rates
class Rates:
    """Yield curve container."""
    def __init__(self, yields: np.ndarray, dates: List):
        if len(yields) != len(dates):
            raise ValueError("Yields and dates length mismatch.")
        self.yields = pd.Series(np.asarray(yields, dtype=float), index=pd.DatetimeIndex(dates))

    def interpolate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """Linearly interpolate yields to specified dates."""
        combined = self.yields.index.union(dates).sort_values()
        return self.yields.reindex(combined).interpolate(method="time").reindex(dates)

    def calc_fwds(self, val_date: datetime, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate implied forward rates between dates."""
        y = self.interpolate(dates)
        t = calc_dt(pd.DatetimeIndex(y.index), val_date)
        acc = np.power(1.0 + np.asarray(y.values, dtype=float), t)
        dt = np.diff(t)
        fwds = np.empty_like(y.values)
        fwds[0] = y.iloc[0]
        fwds[1:] = (acc[1:] / acc[:-1]) ** (1.0 / dt) - 1.0
        return pd.Series(fwds, index=y.index, name="fwds")


# Risk Model
class TransitionMatrix:
    """Credit rating transition matrix container."""
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
            raise ValueError(f"Probabilities for {self.labels[bad_rows]} do not sum to 1")

    def indices_to_labels(self, indices: np.ndarray) -> np.ndarray:
        return self.labels[indices]

    def transitions(self, pX: np.ndarray, ratings_map: pd.Series) -> np.ndarray:
        """Vectorised ratings migration."""
        n_sim, n_years, n_issuers = pX.shape
        initial = np.array([self.label_to_idx[r] for r in ratings_map])
        
        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial[np.newaxis, :]

        for t in range(n_years):
            cur = res[:, t, :]
            cum = self.cum_tmatrix[cur]
            pX_t = pX[:, t, :, np.newaxis]
            res[:, t + 1, :] = np.clip(np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1).astype(np.int32)

        return self.indices_to_labels(res)

    def transitions_from_state(self, pX: np.ndarray, initial_indices: np.ndarray) -> np.ndarray:
        """Vectorised migration starting from pre-defined initial states."""
        n_sim, n_years, n_issuers = pX.shape
        res = np.empty((n_sim, n_years + 1, n_issuers), dtype=np.int32)
        res[:, 0, :] = initial_indices

        for t in range(n_years):
            cur = res[:, t, :]
            cum = self.cum_tmatrix[cur]
            pX_t = pX[:, t, :, np.newaxis]
            res[:, t + 1, :] = np.clip(np.sum(pX_t > cum, axis=-1), 0, self.tmatrix.shape[0] - 1).astype(np.int32)

        return self.indices_to_labels(res)

    def fundamental_matrix(self):
        """Expected visits to each transient state before default."""
        Q = self.tmatrix[:-1, :-1]
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        labels = self.labels[:-1]
        return pd.DataFrame(N, index=labels, columns=labels)

    def time_to_default(self):
        """Expected time to default for each rating."""
        N = self.fundamental_matrix().values
        t = N @ np.ones(len(N))
        return pd.Series(t, index=self.labels[:-1], name='Time to Default')

    def __str__(self):
        return str(pd.DataFrame(self.tmatrix, columns=self.labels, index=self.labels).to_markdown(floatfmt=".1%"))


class SimulationResult:
    """Container for CreditRiskModel outputs."""
    def __init__(self, E, S, I, X, pX, transitions, n_sim, n_years):
        self.E, self.S, self.I, self.X, self.pX = E, S, I, X, pX
        self.transitions = transitions
        self.n_sim, self.n_years = n_sim, n_years


class CreditRiskModel:
    """Monte Carlo factor-based Credit Risk Model."""
    def __init__(self, transition_matrix: TransitionMatrix, rho_e: float, rho_s: np.ndarray, issuer_ids: list, sector_map: pd.Series, ratings_map: pd.Series):
        self.transition_matrix = transition_matrix
        self.rho_e = rho_e
        self.rho_s = rho_s
        self.issuer_ids = issuer_ids
        self.sector_map = sector_map
        self.ratings_map = ratings_map
        self.validate_inputs()

    def validate_inputs(self) -> None:
        if not isinstance(self.sector_map, pd.Series) or not isinstance(self.ratings_map, pd.Series):
            raise TypeError("Maps must be pandas Series.")
        if np.any(self.rho_e > self.rho_s):
            raise ValueError("rho_s must be greater than rho_e")

    def run(self, n_sim: int, n_years: int):
        n_issuers, n_sectors = len(self.ratings_map), self.sector_map.nunique()

        E = np.random.normal(size=(n_sim, n_years))
        S = np.random.normal(size=(n_sim, n_years, n_sectors))
        I = np.random.normal(size=(n_sim, n_years, n_issuers))

        rho_s_i = self.rho_s[self.sector_map]
        X = np.sqrt(self.rho_e) * E[:, :, np.newaxis] + \
            np.sqrt(rho_s_i - self.rho_e)[np.newaxis, np.newaxis, :] * S[:, :, self.sector_map] +\
            np.sqrt(1 - rho_s_i)[np.newaxis, np.newaxis, :] * I

        pX = sp.stats.norm.cdf(X)
        transitions = self.transition_matrix.transitions(pX, self.ratings_map)

        transitions = pd.DataFrame(
            transitions.reshape(n_sim * (n_years+1), n_issuers),
            index=pd.MultiIndex.from_product([range(n_sim), range(n_years + 1)], names=["sim", "year"]),
            columns=self.issuer_ids
        )

        return SimulationResult(E, S, I, X, pX, transitions, n_sim, n_years)


# Assets
class Issuers:
    """Bond issuer details."""
    def __init__(self, ids: List[str], ratings: pd.Series, sectors: pd.Series, names: Optional[pd.Series] = None):
        self.ids, self.ratings, self.sectors, self.names = ids, ratings, sectors, names


class BondSimulationResult:
    """Output of Bonds.run_sim()."""
    def __init__(self, transitions, received_cashflows, recovery_payments, total_cashflows, pvs, dates, bond_ids):
        self.transitions = transitions
        self.received_cashflows = received_cashflows
        self.recovery_payments = recovery_payments
        self.total_cashflows = total_cashflows
        self.pvs = pvs
        self.dates = dates
        self.bond_ids = bond_ids

    def to_dataframe(self, by_bond: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(
            {'rating': self.transitions.reshape(-1), 'cashflow': self.total_cashflows.reshape(-1), 'pv': self.pvs.reshape(-1)},
            index=pd.MultiIndex.from_product([range(self.total_cashflows.shape[0]), self.dates, self.bond_ids], names=["scenario", "date", "bond_id"]),
        ).reset_index()
        return df if by_bond else df.groupby(['scenario', 'date']).sum().drop(columns='bond_id').reset_index()


class Bonds:
    """Bond universe: cashflows, issuer links, and simulation."""
    def __init__(self, ids: list, issuer_ids: pd.Series, recoveries: pd.Series, cashflows: pd.DataFrame, issuers: Issuers, descriptions: pd.Series | None = None):
        self.ids, self.issuer_ids, self.recoveries = ids, issuer_ids, recoveries
        self.cashflows = cashflows[ids].sort_index()
        self.issuers = issuers
        self.descriptions = descriptions
        self.validate_inputs()

    def validate_inputs(self) -> None:
        if len(set(self.ids)) != len(self.ids): raise ValueError("ids must be unique.")
        if ((self.recoveries < 0) | (self.recoveries > 1)).any(): raise ValueError("recoveries between 0 and 1.")

    def year_end_cashflows(self, val_date: Optional[datetime] = None) -> pd.DataFrame:
        cf = self.cashflows.resample("ME").sum()
        if val_date: cf = cf.loc[cf.index > val_date]
        return cf.resample("YE").sum()

    def pv(self, val_date: pd.Timestamp, rates: Rates, spread_map: Dict[str, float]) -> pd.Series:
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

        unique_spreads, spread_inverse = np.unique(map_spreads(bond_transitions, spread_map), return_inverse=True)
        dtime_mat = dt[:, np.newaxis] - dt[:n_years][np.newaxis, :]
        future = dtime_mat > 0

        yields = np.asarray(yields, dtype=float)
        yield_mat = (1 + yields)[:, np.newaxis] * np.ones((1, n_years))

        pv_table = np.zeros((len(unique_spreads), n_years, n_bonds))
        for si, s in enumerate(unique_spreads):
            dfs_mat = np.zeros_like(dtime_mat)
            dfs_mat[future] = (yield_mat[future] + s) ** (-dtime_mat[future])
            pv_table[si] = dfs_mat.T @ cashflows

        spread_idx = spread_inverse.reshape(n_sim, n_years, n_bonds)
        t_idx = np.arange(n_years)[np.newaxis, :, np.newaxis]
        b_idx = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        return pv_table[spread_idx, t_idx, b_idx]

    def run_sim(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult):
        cashflows_df = self.year_end_cashflows(val_date)
        cashflows = cashflows_df.values
        dates = cashflows_df.index
        n_sim, n_years_sim = sim_results.n_sim, sim_results.n_years
        n_years = min(n_years_sim, cashflows.shape[0])

        bond_transitions = (
            sim_results.transitions.loc[sim_results.transitions.index.get_level_values("year") != 0, self.issuer_ids]
            .set_axis(self.ids, axis=1).to_numpy().reshape(n_sim, n_years_sim, len(self.ids))
        )[:, :n_years, :]

        not_matured = compute_maturity_flags(cashflows)
        is_defaulted = (bond_transitions == DEFAULT_LABEL)
        not_defaulted = ~is_defaulted
        first_default = (np.cumsum(is_defaulted, axis=1) == 1) & is_defaulted

        received_cashflows = cashflows[np.newaxis, :n_years, :] * not_defaulted
        recovery_payments = first_default * np.asarray(self.recoveries.values, dtype=float)[np.newaxis, np.newaxis, :] * not_matured[np.newaxis, :n_years, :]
        total_cashflows = received_cashflows + recovery_payments

        yields = np.asarray(rates.interpolate(pd.DatetimeIndex(dates)).values, dtype=float)
        dt = calc_dt(pd.DatetimeIndex(dates), val_date)
        pvs = self._run_sim_pv(cashflows, bond_transitions, spread_map, yields, dt)

        ever_defaulted = np.cumsum(is_defaulted, axis=1) > 0
        pvs = pvs * ~ever_defaulted

        return BondSimulationResult(bond_transitions, received_cashflows, recovery_payments, total_cashflows, pvs, dates[:n_years], self.ids)


# Liabilities
class Liabilities:
    """Liability cashflows."""
    def __init__(self, cashflows: pd.Series, dates: List):
        if len(cashflows) != len(dates): raise ValueError("Mismatch.")
        self.cashflows = pd.Series(cashflows, index=pd.to_datetime(dates))

    def pv(self, val_date: datetime, rates: Union[Rates, float], shift=0.0, timeline=False, n_years: Optional[float] = None) -> Union[float, np.ndarray]:
        cfs = self.cashflows.loc[self.cashflows.index > val_date]
        dt = calc_dt(pd.DatetimeIndex(cfs.index), val_date)

        if isinstance(rates, Rates):
            net_yield = np.asarray(rates.interpolate(pd.DatetimeIndex(cfs.index)).values, dtype=float) + shift
        else:
            net_yield = np.full(len(cfs), float(rates) + shift, dtype=float)

        cf_values = np.asarray(cfs.values, dtype=float)
        if timeline:
            pv = np.array([np.sum(cf_values[i + 1:, np.newaxis] * ((1 + net_yield[i+1:, np.newaxis])**-(dt[i + 1:] - dt[i]))) for i in range(len(cfs.index))])
            return pv[:n_years] if n_years is not None else pv
        return (cf_values * (1 + net_yield) ** -dt).sum()

    def __str__(self):
        return pd.DataFrame(self.cashflows).to_markdown(floatfmt='.0f')


# CDI
def allocate_bond_sim(bond_sim: BondSimulationResult, allocation: pd.Series, shape: tuple, target: str = 'cashflows'):
    """Calculate allocated cashflows or PVs."""
    ids = allocation.index.to_list()
    allocations = allocation.values
    bond_idx = {bid: i for i, bid in enumerate(bond_sim.bond_ids)}
    selected_idx = [bond_idx[id] for id in ids]

    if target == 'pvs': nominal = bond_sim.pvs[:, :, selected_idx]
    elif target == 'cashflows': nominal = bond_sim.total_cashflows[:, :, selected_idx]
    else: raise TypeError('Invalid Option')

    return fit_to_shape((nominal * allocations).sum(axis=2), shape)


class CDISimulationResult:
    """Output container for CDI sim."""
    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, config: "FoxConfig", bond_sim: BondSimulationResult, cdi_sim: pd.DataFrame):
        self.config, self.bond_sim, self.cdi_sim = config, bond_sim, cdi_sim

    def _discount_factors(self) -> pd.Series:
        return (1 + self.cdi_sim["bund_yield"]) ** (-self.cdi_sim["dt"])

    def _pv_by_scenario(self, column: str) -> pd.Series:
        return (self.cdi_sim[column] * self._discount_factors()).groupby(self.cdi_sim["scenario"]).sum()

    def pv_hgb_payments(self) -> pd.Series: return self._pv_by_scenario("hgb_payment").rename("pv_hgb_payment")
    def pv_performance_payments(self) -> pd.Series: return self._pv_by_scenario("performance_payment").rename("pv_performance_payment")
    def pv_total_obligations(self) -> pd.Series: return (self.pv_hgb_payments() + self.pv_performance_payments()).rename("pv_total_obligations")
    def pv_fees(self) -> pd.Series: return self._pv_by_scenario("fee").rename("pv_fee")
    def pv_net(self) -> pd.Series: return (self.pv_fees() - self.pv_total_obligations()).rename("pv_net")

    def obligation_metrics(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        qs = quantiles or self._DEFAULT_QUANTILES
        scenarios = pd.DataFrame({
            "pv_hgb_payment": self.pv_hgb_payments(), "pv_performance_payment": self.pv_performance_payments(),
            "pv_total_obligations": self.pv_total_obligations(), "pv_fee": self.pv_fees(), "pv_net": self.pv_net(),
        })

        summary = scenarios.quantile(qs).T
        summary.insert(0, "mean", scenarios.mean())

        prob = pd.DataFrame({"mean": [(scenarios["pv_hgb_payment"] > 0).mean(), (scenarios["pv_performance_payment"] > 0).mean(), (scenarios["pv_net"] < 0).mean()]},
                            index=["prob_hgb_trigger", "prob_perf_trigger", "prob_net_negative"])
        for q in qs: prob[q] = np.nan
        return pd.concat([summary, prob])


class ICAAPResult:
    """Container for ICAAP nested Monte Carlo results."""
    _DEFAULT_QUANTILES = [0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995]

    def __init__(self, pv_hgb_t1: np.ndarray, pv_perf_t1: np.ndarray, pv0: float, df_1y: float, n_outer: int, n_inner: int, icaap_df: Optional[pd.DataFrame] = None):
        self.pv_hgb_t1, self.pv_perf_t1, self.pv0, self.df_1y = pv_hgb_t1, pv_perf_t1, pv0, df_1y
        self.n_outer, self.n_inner = n_outer, n_inner
        self.icaap_df = icaap_df

    @property
    def pv_total_t1(self) -> np.ndarray: return self.pv_hgb_t1 + self.pv_perf_t1
    @property
    def pv_total_discounted(self) -> np.ndarray: return self.pv_total_t1 * self.df_1y
    @property
    def delta_pv(self) -> np.ndarray: return self.pv_total_discounted - self.pv0

    def var(self, confidence: float = 0.995) -> float: return float(np.quantile(self.delta_pv, confidence))
    def tvar(self, confidence: float = 0.995) -> float:
        d = self.delta_pv
        threshold = np.quantile(d, confidence)
        tail = d[d >= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(threshold)

    def summary(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        qs = quantiles or self._DEFAULT_QUANTILES
        scenarios = pd.DataFrame({
            "pv_hgb_t1": self.pv_hgb_t1, "pv_perf_t1": self.pv_perf_t1,
            "pv_total_t1": self.pv_total_t1, "pv_total_discounted": self.pv_total_discounted, "delta_pv": self.delta_pv,
        })
        tbl = scenarios.quantile(qs).T
        tbl.insert(0, "mean", scenarios.mean())
        cap = pd.DataFrame({"mean": [self.var(), self.tvar(), self.pv0]}, index=["capital_var_995", "capital_tvar_995", "pv0_baseline"])
        for q in qs: cap[q] = np.nan
        return pd.concat([tbl, cap])


class FoxConfig:
    """Mandate parameters."""
    def __init__(self, heubeck_liabilities, r_gaap, r_ifrs, cmbp_margin, asset_buffer, mortality_buffer, fee, performance_cap, additional_payment_year: int = 10, performance_payment_year: int = 25):
        self.heubeck_liabilities, self.r_gaap, self.r_ifrs = heubeck_liabilities, r_gaap, r_ifrs
        self.cmbp_margin, self.asset_buffer, self.mortality_buffer = cmbp_margin, asset_buffer, mortality_buffer
        self.fee, self.performance_cap = fee, performance_cap
        self.additional_payment_year, self.performance_payment_year = additional_payment_year, performance_payment_year

    def to_dict(self): return self.__dict__


class CDIMandate_Fox:
    """The Fox CDI mandate waterfall and analytics logic."""
    def __init__(self, liabilities: Liabilities, cash: float, bonds: Bonds, cdi_allocation: pd.Series, cmbp_allocation: pd.Series, config: FoxConfig):
        self.liabilities, self.cash, self.bonds = liabilities, cash, bonds
        self.cdi_allocation, self.cmbp_allocation, self.config = cdi_allocation, cmbp_allocation, config

    def _prepare_deterministic_metrics(self, val_date: datetime, rates: Rates, max_years: int) -> dict:
        """Pre-compute deterministic parameters used across standard and nested simulations."""
        liab_series = self.liabilities.cashflows.loc[val_date:][:max_years]
        liab_cashflows = np.asarray(liab_series.values, dtype=float)
        dates = pd.DatetimeIndex(liab_series.index)
        T = len(dates)
        dt = calc_dt(dates, val_date)

        liab_pv_gaap = np.asarray(self.liabilities.pv(val_date, self.config.r_gaap, timeline=True, n_years=max_years), dtype=float)
        liab_pv_ifrs = np.asarray(self.liabilities.pv(val_date, self.config.r_ifrs, timeline=True, n_years=max_years), dtype=float)

        meltdown_liabs = np.maximum(
            0.0,
            (self.config.heubeck_liabilities - liab_cashflows.cumsum()) * self.config.mortality_buffer * ((1 + self.config.r_gaap) ** dt)
        )

        full_cfs = np.asarray(self.liabilities.cashflows.loc[val_date:].values, dtype=float)
        if len(full_cfs) < T + 2:
            raise ValueError(f"Liability cashflows too short: need at least {T + 2} entries.")
        next2 = full_cfs[1 : T + 1] + full_cfs[2 : T + 2]

        fwds = np.asarray(rates.calc_fwds(val_date, dates).values, dtype=float)
        yields = np.asarray(rates.interpolate(dates).values, dtype=float)

        asset_buffer_arr = np.array([self.config.asset_buffer * ((1 + self.config.r_ifrs)**(t+1)) if t < 10 else 0.0 for t in range(T)])

        return {
            "dates": dates, "T": T, "dt": dt, "liab_cashflows": liab_cashflows,
            "liab_pv_gaap": liab_pv_gaap, "liab_pv_ifrs": liab_pv_ifrs, "meltdown_liabs": meltdown_liabs,
            "next2": next2, "fwds": fwds, "yields": yields, "asset_buffer_arr": asset_buffer_arr
        }

    def _calc_deterministic_cmbp(self, val_date: datetime, rates: Rates, spread_map: dict, max_years: int):
            """Calculate the total returns for the AAAA-rated CMBP portfolio deterministically upfront."""
            raw_cfs = self.bonds.year_end_cashflows(val_date)
            cashflows = raw_cfs.values
            cf_dates = raw_cfs.index
            
            # Calculate dt and yields for the full cashflow schedule, not just max_years
            dt_full = calc_dt(pd.DatetimeIndex(cf_dates), val_date)
            yields_full = np.asarray(rates.interpolate(pd.DatetimeIndex(cf_dates)).values, dtype=float)
            
            # The number of years we actually want to simulate
            T = min(max_years, len(cf_dates))

            current_ratings = np.asarray(self.bonds.issuer_ids.map(self.bonds.issuers.ratings).reindex(self.bonds.ids).values, dtype=object)
            bond_trans = np.tile(current_ratings[np.newaxis, np.newaxis, :], (1, T, 1))

            # PV calculation using the full vectors
            pvs = self.bonds._run_sim_pv(cashflows, bond_trans, spread_map, yields_full, dt_full)

            # Align the allocation array to match the full bond universe (pad missing with 0.0)
            alloc = self.cmbp_allocation.reindex(self.bonds.ids, fill_value=0.0).values
            
            cmbp_pvs = (pvs[0] * alloc).sum(axis=1)
            cmbp_cfs = (cashflows[:T] * alloc).sum(axis=1)
            cmbp_t0 = (self.bonds.pv(pd.Timestamp(val_date), rates, spread_map) * self.cmbp_allocation).sum()

            # For the forward rates, we only need them for the simulation horizon (T)
            sim_dates = cf_dates[:T]
            fwds = np.asarray(rates.calc_fwds(val_date, pd.DatetimeIndex(sim_dates)).values, dtype=float)[:T]
            
            bt = np.zeros(T)
            if T > 0:
                bt[0] = (cmbp_pvs[0] + cmbp_cfs[0]) / cmbp_t0 - 1
                if T > 1:
                    bt[1:] = (cmbp_pvs[1:] + cmbp_cfs[1:]) / cmbp_pvs[:-1] - 1
            
            bt = np.where(np.isfinite(bt), bt, fwds)
            return bt, cmbp_t0, cmbp_pvs, cmbp_cfs

    def run(self, val_date: datetime, rates: Rates, spread_map: dict, sim_results: SimulationResult) -> CDISimulationResult:
        n_sim = sim_results.n_sim
        metrics = self._prepare_deterministic_metrics(val_date, rates, sim_results.n_years)
        
        T, dt, dates = metrics["T"], metrics["dt"], metrics["dates"]
        fwds, yields = metrics["fwds"], metrics["yields"]
        liab_cashflows = metrics["liab_cashflows"]
        liab_pv_gaap, liab_pv_ifrs = metrics["liab_pv_gaap"], metrics["liab_pv_ifrs"]
        meltdown_liabs, next2, asset_buffer_arr = metrics["meltdown_liabs"], metrics["next2"], metrics["asset_buffer_arr"]

        bt, cmbp_t0, cmbp_pvs_1d, cmbp_cfs_1d = self._calc_deterministic_cmbp(val_date, rates, spread_map, sim_results.n_years)

        bond_prices = self.bonds.pv(pd.Timestamp(val_date), rates, spread_map)
        cdi_t0 = (self.cdi_allocation * bond_prices).sum()
        total_assets_0 = cdi_t0 + self.cash
        day0_hgb_gap = (self.config.heubeck_liabilities * self.config.mortality_buffer - (total_assets_0 + self.config.asset_buffer))

        raw_bond_cfs = self.bonds.year_end_cashflows(val_date)
        exp_cdi_cf = np.asarray((raw_bond_cfs * self.cdi_allocation).sum(axis=1).values, dtype=float)[:T]
        exp_cdi_cf = np.pad(exp_cdi_cf, (0, max(0, T - len(exp_cdi_cf))), mode='constant')

        bond_sim = self.bonds.run_sim(val_date, rates, spread_map, sim_results)
        shape = (n_sim, T)
        asset_cfs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'cashflows')
        asset_pvs = allocate_bond_sim(bond_sim, self.cdi_allocation, shape, 'pvs')

        cash_arr, assets_arr, fee_arr = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        hgb_gap_arr, hgb_payment_arr, bund_comparator_arr = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        perf_payment_arr, additional_payment_arr, net_asset_ret = np.zeros(shape), np.zeros(shape), np.zeros(shape)

        cash_t, assets_t, bund_t = np.full(n_sim, self.cash), np.full(n_sim, total_assets_0), np.full(n_sim, total_assets_0)
        hgb_gap_t, cum_hgb_payment = np.full(n_sim, day0_hgb_gap), np.zeros(n_sim)

        for t in range(T):
            prev_assets_t = assets_t

            fee_t = self.config.fee * (cash_t * (1 + fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])
            cash_t = cash_t * (1 + fwds[t]) + asset_cfs[:, t] - liab_cashflows[t] - fee_t
            assets_t = cash_t + asset_pvs[:, t]

            hgb_gap_t = np.clip(meltdown_liabs[t] - (assets_t + asset_buffer_arr[t]), 0.0, hgb_gap_t)
            hgb_pay_t = np.clip(next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb_payment)
            cash_t += hgb_pay_t
            assets_t += hgb_pay_t
            cum_hgb_payment += hgb_pay_t

            additional_payment_t = 0.0
            if t == self.config.additional_payment_year - 1:
                additional_payment_t = np.clip(np.minimum(liab_pv_gaap[t] - assets_t, 1.1 * liab_pv_ifrs[t] - assets_t), 0.0, asset_buffer_arr[t])
                cash_t += additional_payment_t
                assets_t += additional_payment_t

            bund_t = bund_t * (1 + bt[t] + self.config.cmbp_margin) - liab_cashflows[t] + additional_payment_t
            performance_payment = np.clip(bund_t - assets_t, 0.0, self.config.performance_cap) if t == self.config.performance_payment_year - 1 else 0.0
            net_ret_t = (assets_t + liab_cashflows[t] - additional_payment_t) / prev_assets_t - 1.0

            fee_arr[:, t] = fee_t
            cash_arr[:, t] = cash_t
            assets_arr[:, t] = assets_t
            hgb_gap_arr[:, t] = hgb_gap_t
            hgb_payment_arr[:, t] = hgb_pay_t
            perf_payment_arr[:, t] = performance_payment
            additional_payment_arr[:, t] = additional_payment_t
            bund_comparator_arr[:, t] = bund_t
            net_asset_ret[:, t] = net_ret_t

        tile = lambda x: np.tile(x, (n_sim, 1))
        results = {
            "dt": tile(dt), "liab_cashflow": tile(liab_cashflows), "liab_pv_gaap": tile(liab_pv_gaap), "liab_pv_ifrs": tile(liab_pv_ifrs),
            "meltdown_liabilities": tile(meltdown_liabs), "next_2_liabs": tile(next2), "bund_fwds": tile(fwds), "bund_yield": tile(yields),
            "expected_cdi_cashflow": tile(exp_cdi_cf), "asset_cashflow": asset_cfs, "remaining_asset_pv": asset_pvs,
            "cmbp_cashflow": tile(cmbp_cfs_1d), "cmbp_pv": tile(cmbp_pvs_1d), "bt": tile(bt),
            "fee": fee_arr, "cash": cash_arr, "assets": assets_arr, "net_asset_return": net_asset_ret,
            "hgb_gap": hgb_gap_arr, "hgb_payment": hgb_payment_arr, "bund_comparator": bund_comparator_arr,
            "performance_payment": perf_payment_arr, "additional_payment": additional_payment_arr,
        }
        df = pd.DataFrame({k: v.reshape(-1) for k, v in results.items()}, index=pd.MultiIndex.from_product([np.arange(n_sim), dates], names=["scenario", "date"])).reset_index()

        df["funding_level_gaap"] = df["assets"] / df["liab_pv_gaap"]
        df["funding_level_ifrs"] = df["assets"] / df["liab_pv_ifrs"]
        df["net_bt_return"] = df["bt"] + self.config.cmbp_margin

        return CDISimulationResult(config=self.config, bond_sim=bond_sim, cdi_sim=df)

    def icaap_capital(self, val_date: datetime, rates: Rates, spread_map: dict, cr_model: CreditRiskModel, base_result: CDISimulationResult, n_outer: int = 500, n_inner: int = 500, chunk_size: int = 50) -> ICAAPResult:
        n_years = base_result.bond_sim.total_cashflows.shape[1]
        if n_years < 2: raise ValueError("ICAAP requires n_years >= 2.")
        n_years_inner = n_years - 1

        metrics = self._prepare_deterministic_metrics(val_date, rates, n_years)
        dt_full = metrics["dt"]

        # Deterministic CMBP portfolio returns calculated upfront
        bt_full, _, _, _ = self._calc_deterministic_cmbp(val_date, rates, spread_map, n_years)

        # Slice deterministic arrays for inner sim loops
        T_inner            = metrics["T"] - 1
        inner_fwds         = metrics["fwds"][1:]
        inner_yields       = metrics["yields"][1:]
        inner_liab_cfs     = metrics["liab_cashflows"][1:]
        inner_meltdown     = metrics["meltdown_liabs"][1:]
        inner_next2        = metrics["next2"][1:]
        inner_liab_pv_gaap = metrics["liab_pv_gaap"][1:]
        inner_liab_pv_ifrs = metrics["liab_pv_ifrs"][1:]
        inner_asset_buffer = metrics["asset_buffer_arr"][1:]
        inner_bt           = bt_full[1:]

        inner_dt = dt_full[1:] - dt_full[0]
        inner_df = (1 + inner_yields) ** (-inner_dt)

        # Analytical bund comparator logic utilizing deterministic `inner_bt`
        inner_g = 1 + inner_bt + self.config.cmbp_margin
        cum_g   = np.cumprod(inner_g)

        perf_inner_t = self.config.performance_payment_year - 2
        ap_inner_t   = self.config.additional_payment_year - 2
        has_perf     = 0 <= perf_inner_t < T_inner

        if has_perf:
            bund_growth = cum_g[perf_inner_t]
            bund_drain = float(bund_growth * (inner_liab_cfs[: perf_inner_t + 1] / cum_g[: perf_inner_t + 1]).sum())
            ap_fwd_growth = float(cum_g[perf_inner_t] / cum_g[ap_inner_t]) if 0 <= ap_inner_t <= perf_inner_t else 0.0
            perf_df = float(inner_df[perf_inner_t])

        # Step 1: Execute 1-year outer sim
        outer_cr  = cr_model.run(n_outer, 1)
        outer_cdi = self.run(val_date, rates, spread_map, outer_cr)

        df1 = outer_cdi.cdi_sim
        cash_outer    = df1["cash"].values
        hgb_gap_outer = df1["hgb_gap"].values
        cum_hgb_outer = df1["hgb_payment"].values
        bund_outer    = df1["bund_comparator"].values

        year1_mask    = outer_cr.transitions.index.get_level_values("year") == 1
        year1_ratings = outer_cr.transitions.loc[year1_mask].values
        already_defaulted = (outer_cdi.bond_sim.transitions[:, 0, :] == DEFAULT_LABEL)

        tm = cr_model.transition_matrix
        initial_indices_all = np.array([[tm.label_to_idx[r] for r in year1_ratings[s]] for s in range(n_outer)])

        # Step 2: Inner loop via batching
        n_issuers = len(cr_model.issuer_ids)
        rho_s_i = cr_model.rho_s[cr_model.sector_map]
        inner_val_date = metrics["dates"][0]

        sqrt_rho_e   = np.sqrt(cr_model.rho_e)
        sqrt_rho_s_e = np.sqrt(rho_s_i - cr_model.rho_e)
        sqrt_1_rho_s = np.sqrt(1 - rho_s_i)

        pv_hgb_cond  = np.empty(n_outer)
        pv_perf_cond = np.empty(n_outer)

        for c_start in range(0, n_outer, chunk_size):
            c_end  = min(c_start + chunk_size, n_outer)
            n_c, N_c = c_end - c_start, (c_end - c_start) * n_inner

            E = np.random.normal(size=(N_c, n_years_inner))
            S = np.random.normal(size=(N_c, n_years_inner, cr_model.sector_map.nunique()))
            I_noise = np.random.normal(size=(N_c, n_years_inner, n_issuers))

            X = sqrt_rho_e * E[:, :, np.newaxis] + sqrt_rho_s_e[np.newaxis, np.newaxis, :] * S[:, :, cr_model.sector_map] + sqrt_1_rho_s[np.newaxis, np.newaxis, :] * I_noise
            pX = sp.stats.norm.cdf(X)

            init_chunk = np.repeat(initial_indices_all[c_start:c_end], n_inner, axis=0)
            inner_trans_arr = tm.transitions_from_state(pX, init_chunk)

            inner_trans_df = pd.DataFrame(
                inner_trans_arr.reshape(N_c * (n_years_inner + 1), n_issuers),
                index=pd.MultiIndex.from_product([range(N_c), range(n_years_inner + 1)], names=["sim", "year"]),
                columns=cr_model.issuer_ids,
            )
            inner_sim = SimulationResult(E, S, I_noise, X, pX, inner_trans_df, N_c, n_years_inner)

            inner_bond_sim = self.bonds.run_sim(inner_val_date, rates, spread_map, inner_sim)
            already_def_c = np.repeat(already_defaulted[c_start:c_end], n_inner, axis=0)
            inner_bond_sim.recovery_payments[:, 0, :] *= ~already_def_c
            inner_bond_sim.total_cashflows = inner_bond_sim.received_cashflows + inner_bond_sim.recovery_payments

            shape_c = (N_c, T_inner)
            asset_cfs = allocate_bond_sim(inner_bond_sim, self.cdi_allocation, shape_c, "cashflows")
            asset_pvs = allocate_bond_sim(inner_bond_sim, self.cdi_allocation, shape_c, "pvs")

            del inner_bond_sim, inner_trans_arr, inner_trans_df, inner_sim, E, S, I_noise, X, pX

            cash_t    = np.repeat(np.array(cash_outer[c_start:c_end], dtype=float), n_inner)
            hgb_gap_t = np.repeat(np.array(hgb_gap_outer[c_start:c_end], dtype=float), n_inner)
            cum_hgb   = np.repeat(np.array(cum_hgb_outer[c_start:c_end], dtype=float), n_inner)

            hgb_arr, additional_injected, assets_at_perf = np.zeros(shape_c), np.zeros(N_c), np.zeros(N_c)

            for t in range(T_inner):
                orig_t = t + 1
                fee_t = self.config.fee * (cash_t * (1 + inner_fwds[t]) + asset_pvs[:, t] + asset_cfs[:, t])
                cash_t = (cash_t * (1 + inner_fwds[t]) + asset_cfs[:, t] - inner_liab_cfs[t] - fee_t)
                assets_t = cash_t + asset_pvs[:, t]

                hgb_gap_t = np.clip(inner_meltdown[t] - (assets_t + inner_asset_buffer[t]), 0.0, hgb_gap_t)
                hgb_pay_t = np.clip(inner_next2[t] - assets_t, 0.0, hgb_gap_t - cum_hgb)
                cash_t += hgb_pay_t
                assets_t += hgb_pay_t
                cum_hgb += hgb_pay_t

                if orig_t == self.config.additional_payment_year - 1:
                    add_t = np.clip(np.minimum(inner_liab_pv_gaap[t] - assets_t, 1.1 * inner_liab_pv_ifrs[t] - assets_t), 0.0, inner_asset_buffer[t])
                    cash_t += add_t
                    assets_t += add_t
                    additional_injected += add_t

                if t == perf_inner_t: assets_at_perf = assets_t.copy()
                hgb_arr[:, t] = hgb_pay_t

            if has_perf:
                bund_chunk = np.repeat(np.array(bund_outer[c_start:c_end], dtype=float), n_inner)
                bund_at_perf = bund_chunk * bund_growth - bund_drain + additional_injected * ap_fwd_growth
                pv_perf_path = np.clip(bund_at_perf - assets_at_perf, 0.0, self.config.performance_cap) * perf_df
            else:
                pv_perf_path = np.zeros(N_c)

            pv_hgb_path = (hgb_arr * inner_df[np.newaxis, :]).sum(axis=1)

            pv_hgb_cond[c_start:c_end]  = pv_hgb_path.reshape(n_c, n_inner).mean(axis=1)
            pv_perf_cond[c_start:c_end] = pv_perf_path.reshape(n_c, n_inner).mean(axis=1)

        return ICAAPResult(
            pv_hgb_t1=pv_hgb_cond, pv_perf_t1=pv_perf_cond,
            pv0=float(base_result.pv_total_obligations().mean()),
            df_1y=float((1 + metrics["yields"][0]) ** (-dt_full[0])),
            n_outer=n_outer, n_inner=n_inner,
            icaap_df=None
        )