"""
CDI Credit Risk Model — Challenger (v2)

Design principles
-----------------
* Explicit ``np.random.Generator`` everywhere — no global RNG state.
* Integer-indexed ratings in all hot paths; string labels only at I/O
  boundaries via a shared ``RatingSchema``.
* Dataclasses for configs and result containers.
* A single, standalone waterfall function shared by base-case and ICAAP.
* ICAAP pre-generates all random draws before chunking so results are
  invariant to ``chunk_size``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════
#  Constants & helpers
# ═══════════════════════════════════════════════════════════════════════════
DAYS_PER_YEAR = 365.0


def year_frac(dates: pd.DatetimeIndex, val_date: pd.Timestamp) -> np.ndarray:
    """ACT/365 year fractions from *val_date* to each date."""
    return np.asarray((dates - val_date).days, dtype=np.float64) / DAYS_PER_YEAR


def _pad_or_trim(a: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim 1-D *a* to *target_len*."""
    if len(a) >= target_len:
        return a[:target_len]
    return np.pad(a, (0, target_len - len(a)))


# ═══════════════════════════════════════════════════════════════════════════
#  Rating schema — single source of truth for label ↔ integer mapping
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RatingSchema:
    """Immutable label ↔ int mapping.  The *last* label is always Default."""

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_l2i", {l: i for i, l in enumerate(self.labels)})

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def default_idx(self) -> int:
        return self.n - 1

    def to_idx(self, labels: np.ndarray) -> np.ndarray:
        """Vectorised label → int conversion."""
        flat = np.asarray(labels).ravel()
        codes, uniques = pd.factorize(flat, sort=False)
        mapped = np.array([self._l2i[u] for u in uniques], dtype=np.int32)
        return mapped[codes].reshape(labels.shape)

    def to_label(self, indices: np.ndarray) -> np.ndarray:
        arr = np.asarray(self.labels)
        return arr[indices]


DEFAULT_SCHEMA = RatingSchema(labels=(
    'AAAA', 'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
    'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
    'CCC+', 'CCC', 'CCC-', 'CC+', 'Def',
))


# ═══════════════════════════════════════════════════════════════════════════
#  Yield curve
# ═══════════════════════════════════════════════════════════════════════════
class YieldCurve:
    """Interpolating yield-curve wrapper.  All dates must be timezone-naive."""

    def __init__(self, yields: np.ndarray, dates: pd.DatetimeIndex):
        if len(yields) != len(dates):
            raise ValueError("yields / dates length mismatch")
        self._s = pd.Series(np.asarray(yields, dtype=np.float64), index=dates)

    def at(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Linearly interpolated yields at arbitrary dates."""
        combined = self._s.index.union(dates).sort_values()
        return np.asarray(
            self._s.reindex(combined).interpolate("time").reindex(dates), dtype=np.float64,
        )

    def forwards(self, val_date: pd.Timestamp, dates: pd.DatetimeIndex) -> np.ndarray:
        """Implied 1-period forward rates."""
        y = self.at(dates)
        t = year_frac(dates, val_date)
        acc = (1 + y) ** t
        fwds = np.empty_like(y)
        fwds[0] = y[0]
        dt = np.diff(t)
        fwds[1:] = (acc[1:] / acc[:-1]) ** (1.0 / dt) - 1.0
        return fwds


# ═══════════════════════════════════════════════════════════════════════════
#  Transition matrix
# ═══════════════════════════════════════════════════════════════════════════
class TransitionMatrix:
    """Row-stochastic credit migration matrix operating on integer indices."""

    def __init__(self, matrix: np.ndarray, schema: RatingSchema):
        n = schema.n
        if matrix.shape != (n, n):
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match schema size {n}")
        bad = np.where(~np.isclose(matrix.sum(axis=1), 1.0))[0]
        if bad.size:
            raise ValueError(f"Rows {bad} don't sum to 1")

        self.matrix = matrix
        self.cum = np.cumsum(matrix, axis=1)
        self.schema = schema

    # ── migration engine (pure int arrays) ────────────────────────────────
    def migrate(
        self,
        pX: np.ndarray,
        initial: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorised rating migration.

        Parameters
        ----------
        pX : (n_sim, n_years, n_entities) uniform draws in [0, 1]
        initial : (n_entities,) or (n_sim, n_entities) int32 starting indices

        Returns
        -------
        (n_sim, n_years + 1, n_entities) int32 rating-index paths.
        """
        n_sim, n_years, n_entities = pX.shape
        max_idx = self.matrix.shape[0] - 1

        path = np.empty((n_sim, n_years + 1, n_entities), dtype=np.int32)
        if initial.ndim == 1:
            path[:, 0, :] = initial[np.newaxis, :]
        else:
            path[:, 0, :] = initial

        for t in range(n_years):
            cum_row = self.cum[path[:, t, :]]                      # (n_sim, n_entities, n_states)
            path[:, t + 1, :] = np.clip(
                (pX[:, t, :, np.newaxis] > cum_row).sum(axis=-1),
                0, max_idx,
            ).astype(np.int32)

        return path

    # ── analytics ─────────────────────────────────────────────────────────
    def fundamental(self) -> pd.DataFrame:
        Q = self.matrix[:-1, :-1]
        N = np.linalg.inv(np.eye(len(Q)) - Q)
        lbl = list(self.schema.labels[:-1])
        return pd.DataFrame(N, index=lbl, columns=lbl)

    def expected_time_to_default(self) -> pd.Series:
        N = self.fundamental().values
        return pd.Series(N @ np.ones(len(N)), index=self.schema.labels[:-1])


# ═══════════════════════════════════════════════════════════════════════════
#  Factor-based credit model
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CreditSimResult:
    """Container returned by ``CreditModel.simulate``."""
    paths: np.ndarray           # (n_sim, n_years+1, n_issuers) int32
    pX: np.ndarray              # (n_sim, n_years, n_issuers) float64
    n_sim: int
    n_years: int
    schema: RatingSchema
    rho_e: float
    rho_s: np.ndarray

    # ── convenience ───────────────────────────────────────────────────────
    def label_paths(self) -> np.ndarray:
        return self.schema.to_label(self.paths)

    def year1_indices(self) -> np.ndarray:
        """(n_sim, n_issuers) ratings at t = 1."""
        return self.paths[:, 1, :]


class CreditModel:
    """BRS-style single-factor model with economy/sector/idio decomposition."""

    def __init__(
        self,
        tmatrix: TransitionMatrix,
        rho_e: float,
        rho_s: np.ndarray,
        n_issuers: int,
        sector_ids: np.ndarray,
        initial_ratings: np.ndarray,
    ):
        self.tmatrix = tmatrix
        self.rho_e = rho_e
        self.rho_s = np.asarray(rho_s, dtype=np.float64)
        self.n_issuers = n_issuers
        self.sector_ids = np.asarray(sector_ids, dtype=np.int32)
        self.initial_ratings = np.asarray(initial_ratings, dtype=np.int32)  # int indices
        self.schema = tmatrix.schema

        rho_s_i = self.rho_s[self.sector_ids]
        if np.any(rho_e > rho_s_i):
            raise ValueError("rho_s must be >= rho_e for every issuer")

        # Pre-compute factor loadings (per issuer)
        self._w_e = np.sqrt(rho_e)
        self._w_s = np.sqrt(rho_s_i - rho_e)
        self._w_i = np.sqrt(1 - rho_s_i)

    # ── random-draw generation ────────────────────────────────────────────
    def generate_pX(
        self, rng: np.random.Generator, n_sim: int, n_years: int,
    ) -> np.ndarray:
        """Correlated uniform draws (n_sim, n_years, n_issuers)."""
        n_sectors = int(self.sector_ids.max()) + 1

        E = rng.standard_normal((n_sim, n_years))
        S = rng.standard_normal((n_sim, n_years, n_sectors))
        I = rng.standard_normal((n_sim, n_years, self.n_issuers))

        X = (
            self._w_e * E[:, :, np.newaxis]
            + self._w_s[np.newaxis, np.newaxis, :] * S[:, :, self.sector_ids]
            + self._w_i[np.newaxis, np.newaxis, :] * I
        )
        return sp_stats.norm.cdf(X)

    # ── full simulation ───────────────────────────────────────────────────
    def simulate(
        self,
        rng: np.random.Generator,
        n_sim: int,
        n_years: int,
        *,
        initial: np.ndarray | None = None,
        pX: np.ndarray | None = None,
    ) -> CreditSimResult:
        """
        Run the credit migration simulation.

        Parameters
        ----------
        rng : np.random.Generator
        n_sim, n_years : simulation dimensions
        initial : optional per-scenario starting ratings (n_sim, n_issuers)
                  int32.  Defaults to ``self.initial_ratings`` broadcast.
        pX : optional pre-computed uniform draws.  If ``None``, fresh
             draws are generated from *rng*.
        """
        if pX is None:
            pX = self.generate_pX(rng, n_sim, n_years)

        init = self.initial_ratings if initial is None else initial
        paths = self.tmatrix.migrate(pX, init)

        return CreditSimResult(
            paths=paths, pX=pX,
            n_sim=n_sim, n_years=n_years,
            schema=self.schema,
            rho_e=self.rho_e, rho_s=self.rho_s,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Bonds
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BondUniverse:
    """
    Static bond data.  Everything is aligned by a common integer bond index.

    Parameters
    ----------
    cashflows : (n_dates, n_bonds) scheduled cashflows
    dates     : DatetimeIndex of cashflow dates
    issuer_ids : (n_bonds,) int mapping bond → issuer
    recoveries : (n_bonds,) LGD recovery rates in [0, 1]
    spread_table : (n_ratings,) spread per rating index
    """
    cashflows: np.ndarray
    dates: pd.DatetimeIndex
    issuer_ids: np.ndarray
    recoveries: np.ndarray
    spread_table: np.ndarray    # indexed by rating int

    @property
    def n_bonds(self) -> int:
        return self.cashflows.shape[1]

    @property
    def n_dates(self) -> int:
        return self.cashflows.shape[0]

    # ── deterministic PV (t = 0) ──────────────────────────────────────────
    def pv0(self, val_date: pd.Timestamp, curve: YieldCurve,
            rating_idx: np.ndarray) -> np.ndarray:
        """PV of each bond at *val_date* given integer rating indices per bond."""
        dt = year_frac(self.dates, val_date)
        y = curve.at(self.dates)
        sp = self.spread_table[rating_idx]                         # (n_bonds,)
        dfs = (1 + y[:, np.newaxis] + sp[np.newaxis, :]) ** (-dt[:, np.newaxis])
        return (self.cashflows * dfs).sum(axis=0)

    # ── maturity flags ────────────────────────────────────────────────────
    def _maturity_mask(self) -> np.ndarray:
        """(n_dates, n_bonds) True up to & including last non-zero CF date."""
        n_d, n_b = self.cashflows.shape
        has_cf = self.cashflows.any(axis=0)
        rev = np.flip(self.cashflows, axis=0)
        last = np.where(has_cf, n_d - 1 - np.argmax(rev != 0, axis=0), -1)
        return np.arange(n_d)[:, np.newaxis] <= last[np.newaxis, :]

    # ── stochastic simulation ─────────────────────────────────────────────
    def simulate(
        self,
        val_date: pd.Timestamp,
        curve: YieldCurve,
        credit_paths: np.ndarray,      # (n_sim, n_years, n_issuers) int32
        default_idx: int,
    ) -> "BondSimResult":
        """
        Project cashflows and mark-to-model PVs under simulated rating paths.

        Parameters
        ----------
        credit_paths : int-indexed rating paths *excluding* year 0,
                       shape (n_sim, n_years, n_issuers).
        """
        cf = self.cashflows                                        # (n_dates, n_bonds)
        n_dates, n_bonds = cf.shape
        n_sim, n_years_sim, _ = credit_paths.shape
        T = min(n_dates, n_years_sim)

        # Bond-level paths: map issuer paths → bonds
        bond_paths = credit_paths[:, :T, self.issuer_ids]          # (n_sim, T, n_bonds)

        is_def = bond_paths == default_idx
        ever_def = np.cumsum(is_def, axis=1) > 0
        first_def = (np.cumsum(is_def, axis=1) == 1) & is_def
        not_matured = self._maturity_mask()[:T, :]

        # Received cashflows (zero after default)
        received = cf[np.newaxis, :T, :] * ~ever_def

        # Recovery on first default year (if bond not yet matured)
        recovery = (
            first_def
            * self.recoveries[np.newaxis, np.newaxis, :]
            * not_matured[np.newaxis, :, :]
        )
        total_cf = received + recovery

        # PV of remaining cashflows at each (sim, year) node
        pvs = self._sim_pvs(cf, bond_paths, curve, val_date, T)
        pvs *= ~ever_def                                           # zero after default

        return BondSimResult(
            total_cf=total_cf, pvs=pvs, bond_paths=bond_paths,
            dates=self.dates[:T], n_sim=n_sim, T=T,
        )

    def _sim_pvs(
        self,
        cf: np.ndarray,
        bond_paths: np.ndarray,
        curve: YieldCurve,
        val_date: pd.Timestamp,
        T: int,
    ) -> np.ndarray:
        """PV lookup-table approach: (n_sim, T, n_bonds)."""
        n_dates, n_bonds = cf.shape
        n_sim = bond_paths.shape[0]

        dt = year_frac(self.dates, val_date)
        y = curve.at(self.dates)

        # Build PV table indexed by [spread_idx, year, bond]
        # Spread index = rating int of the bond at that node
        spreads_flat = self.spread_table[bond_paths.ravel()]
        unique_sp, sp_inv = np.unique(spreads_flat, return_inverse=True)
        n_u = len(unique_sp)

        dtime = dt[:, np.newaxis] - dt[:T][np.newaxis, :]         # (n_dates, T)
        future = dtime > 0
        y_base = (1 + y)[:, np.newaxis] * np.ones((1, T))

        pv_tab = np.zeros((n_u, T, n_bonds))
        for si, s in enumerate(unique_sp):
            dfs = np.zeros_like(dtime)
            dfs[future] = (y_base[future] + s) ** (-dtime[future])
            pv_tab[si] = dfs.T @ cf

        sp_idx = sp_inv.reshape(n_sim, T, n_bonds)
        t_ax = np.arange(T)[np.newaxis, :, np.newaxis]
        b_ax = np.arange(n_bonds)[np.newaxis, np.newaxis, :]
        return pv_tab[sp_idx, t_ax, b_ax]


@dataclass
class BondSimResult:
    total_cf: np.ndarray        # (n_sim, T, n_bonds)
    pvs: np.ndarray             # (n_sim, T, n_bonds)
    bond_paths: np.ndarray      # (n_sim, T, n_bonds) int32
    dates: pd.DatetimeIndex
    n_sim: int
    T: int

    def portfolio(self, weights: np.ndarray, target: str = "cf") -> np.ndarray:
        """Weighted sum across bonds → (n_sim, T)."""
        src = self.total_cf if target == "cf" else self.pvs
        n_sim, T_src, _ = src.shape
        T = min(T_src, len(weights))
        # weights may be shorter than n_bonds if portfolio is a subset
        out = (src[:, :T, :len(weights)] * weights[np.newaxis, np.newaxis, :]).sum(axis=2)
        # Pad if needed
        if out.shape[1] < T_src:
            out = np.pad(out, ((0, 0), (0, T_src - out.shape[1])))
        return out


# ═══════════════════════════════════════════════════════════════════════════
#  Liabilities
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Liabilities:
    cashflows: np.ndarray       # (n_dates,)
    dates: pd.DatetimeIndex

    def pv_scalar(self, val_date: pd.Timestamp, rate: float) -> float:
        cf = self.cashflows[self.dates > val_date]
        d = self.dates[self.dates > val_date]
        dt = year_frac(d, val_date)
        return float((cf * (1 + rate) ** (-dt)).sum())

    def pv_timeline(self, val_date: pd.Timestamp, rate: float, T: int) -> np.ndarray:
        """PV at each year-end looking *forward* from that point."""
        cf = self.cashflows[self.dates > val_date]
        d = self.dates[self.dates > val_date]
        dt = year_frac(d, val_date)
        n = len(cf)
        out = np.empty(min(n, T))
        for i in range(len(out)):
            fwd_dt = dt[i + 1:] - dt[i]
            out[i] = (cf[i + 1:] * (1 + rate) ** (-fwd_dt)).sum()
        return out

    def annual(self, val_date: pd.Timestamp, T: int) -> np.ndarray:
        """First T annual cashflows after val_date."""
        cf = self.cashflows[self.dates > val_date]
        return _pad_or_trim(cf, T)


# ═══════════════════════════════════════════════════════════════════════════
#  Mandate configuration
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class MandateConfig:
    heubeck_liabilities: float
    r_gaap: float
    r_ifrs: float
    mortality_buffer: float
    cmbp_margin: float
    fee_rate: float
    asset_buffer_0: float
    performance_cap: float
    additional_payment_year: int = 10        # 1-indexed year
    performance_payment_year: int = 25       # 1-indexed year


# ═══════════════════════════════════════════════════════════════════════════
#  Deterministic schedule (pre-computed once)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Schedule:
    """All time-series that are deterministic w.r.t. credit scenarios."""
    dates: pd.DatetimeIndex
    T: int
    dt: np.ndarray
    yields: np.ndarray
    fwds: np.ndarray
    df: np.ndarray
    liab_cf: np.ndarray
    liab_pv_gaap: np.ndarray
    liab_pv_ifrs: np.ndarray
    meltdown: np.ndarray
    next2: np.ndarray
    asset_buffer: np.ndarray
    exp_cdi_cf: np.ndarray
    bt: np.ndarray             # CMBP total return


def build_schedule(
    val_date: pd.Timestamp,
    curve: YieldCurve,
    liabs: Liabilities,
    cfg: MandateConfig,
    bonds: BondUniverse,
    cdi_weights: np.ndarray,
    cmbp_weights: np.ndarray,
    T: int,
) -> Schedule:
    """Compute all deterministic schedule arrays."""

    # ── liability vectors ─────────────────────────────────────────────────
    liab_cf = liabs.annual(val_date, T)
    # Use the bond dates for year-end alignment
    dates = bonds.dates[:T]
    dt = year_frac(dates, val_date)
    liab_pv_gaap = liabs.pv_timeline(val_date, cfg.r_gaap, T)
    liab_pv_ifrs = liabs.pv_timeline(val_date, cfg.r_ifrs, T)

    cum_liab = liab_cf.cumsum()
    meltdown = np.maximum(
        0.0,
        (cfg.heubeck_liabilities - cum_liab) * cfg.mortality_buffer * (1 + cfg.r_gaap) ** dt,
    )

    full_cf = _pad_or_trim(liabs.cashflows[liabs.dates > val_date], T + 2)
    next2 = full_cf[1:T + 1] + full_cf[2:T + 2]

    # ── rates ─────────────────────────────────────────────────────────────
    y = curve.at(dates)
    fwds = curve.forwards(val_date, dates)
    df = (1 + y) ** (-dt)

    # ── asset buffer ──────────────────────────────────────────────────────
    ab = np.array([
        cfg.asset_buffer_0 * (1 + cfg.r_ifrs) ** (t + 1) if t < 10 else 0.0
        for t in range(T)
    ])

    # ── expected CDI cashflows ────────────────────────────────────────────
    raw = bonds.cashflows[:T, :]
    exp_cdi_cf = (raw * cdi_weights[np.newaxis, :]).sum(axis=1)

    # ── CMBP total returns (deterministic — AAAA absorbing, spread ≈ 0) ──
    cmbp_cf = (raw * cmbp_weights[np.newaxis, :]).sum(axis=1)
    full_dates = bonds.dates
    full_dt = year_frac(full_dates, val_date)
    full_y = curve.at(full_dates)
    n_full = len(full_dates)

    # CMBP PV at t=0 (uses base yields only — AAAA spread is 0)
    cmbp_t0 = float(
        (cmbp_cf * (1 + full_y[:len(cmbp_cf)]) ** (-full_dt[:len(cmbp_cf)])).sum()
    ) if len(cmbp_cf) > 0 else 0.0

    # CMBP PV at each year-end
    full_cmbp_cf = _pad_or_trim((bonds.cashflows * cmbp_weights[np.newaxis, :]).sum(axis=1), n_full)
    cmbp_pvs = np.zeros(T)
    for t in range(T):
        mask = np.arange(n_full) > t
        dtime = full_dt[mask] - full_dt[t]
        cmbp_pvs[t] = (full_cmbp_cf[mask] * (1 + full_y[mask]) ** (-dtime)).sum()

    bt = np.empty(T)
    cf_T = full_cmbp_cf[:T]
    bt[0] = (cmbp_pvs[0] + cf_T[0]) / cmbp_t0 - 1 if cmbp_t0 > 0 else fwds[0]
    for t in range(1, T):
        bt[t] = (cmbp_pvs[t] + cf_T[t]) / cmbp_pvs[t - 1] - 1 if cmbp_pvs[t - 1] > 0 else fwds[t]

    return Schedule(
        dates=dates, T=T, dt=dt, yields=y, fwds=fwds, df=df,
        liab_cf=liab_cf, liab_pv_gaap=liab_pv_gaap, liab_pv_ifrs=liab_pv_ifrs,
        meltdown=meltdown, next2=next2, asset_buffer=ab,
        exp_cdi_cf=exp_cdi_cf, bt=bt,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Waterfall engine
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WaterfallResult:
    fee: np.ndarray
    cash: np.ndarray
    assets: np.ndarray
    hgb_gap: np.ndarray
    hgb_payment: np.ndarray
    performance_payment: np.ndarray
    additional_payment: np.ndarray
    bund_comparator: np.ndarray
    net_asset_return: np.ndarray


def run_waterfall(
    n_sim: int,
    T: int,
    cfg: MandateConfig,
    sched_fwds: np.ndarray,
    sched_liab_cf: np.ndarray,
    sched_liab_pv_gaap: np.ndarray,
    sched_liab_pv_ifrs: np.ndarray,
    sched_meltdown: np.ndarray,
    sched_next2: np.ndarray,
    sched_ab: np.ndarray,
    sched_bt: np.ndarray,
    asset_cf: np.ndarray,      # (n_sim, T)
    asset_pv: np.ndarray,      # (n_sim, T)
    cash_0: np.ndarray,
    assets_0: np.ndarray,
    hgb_gap_0: np.ndarray,
    cum_hgb_0: np.ndarray,
    bund_0: np.ndarray,
    add_pay_idx: int,          # 0-indexed
    perf_pay_idx: int,         # 0-indexed
) -> WaterfallResult:
    """Pure-numpy CDI waterfall.  All scenario vectors are (n_sim,)."""

    cash = cash_0.copy()
    assets = assets_0.copy()
    hgb_gap = hgb_gap_0.copy()
    cum_hgb = cum_hgb_0.copy()
    bund = bund_0.copy()

    shape = (n_sim, T)
    fee_arr = np.zeros(shape)
    cash_arr = np.zeros(shape)
    assets_arr = np.zeros(shape)
    hgb_gap_arr = np.zeros(shape)
    hgb_pay_arr = np.zeros(shape)
    perf_arr = np.zeros(shape)
    add_arr = np.zeros(shape)
    bund_arr = np.zeros(shape)
    ret_arr = np.zeros(shape)

    for t in range(T):
        prev = assets.copy()

        fee = cfg.fee_rate * (cash * (1 + sched_fwds[t]) + asset_cf[:, t] + asset_pv[:, t])
        cash = cash * (1 + sched_fwds[t]) + asset_cf[:, t] - sched_liab_cf[t] - fee
        assets = cash + asset_pv[:, t]

        hgb_gap = np.clip(sched_meltdown[t] - (assets + sched_ab[t]), 0.0, hgb_gap)
        hgb_pay = np.clip(sched_next2[t] - assets, 0.0, hgb_gap - cum_hgb)
        cash += hgb_pay
        assets += hgb_pay
        cum_hgb += hgb_pay

        add = np.zeros(n_sim)
        if t == add_pay_idx:
            add = np.clip(
                np.minimum(
                    sched_liab_pv_gaap[t] - assets,
                    1.1 * sched_liab_pv_ifrs[t] - assets,
                ),
                0.0, sched_ab[t],
            )
            cash += add
            assets += add

        bund = bund * (1 + sched_bt[t] + cfg.cmbp_margin) - sched_liab_cf[t] + add

        perf = np.zeros(n_sim)
        if t == perf_pay_idx:
            perf = np.clip(bund - assets, 0.0, cfg.performance_cap)

        ret = (assets + sched_liab_cf[t] - add) / prev - 1.0

        fee_arr[:, t] = fee
        cash_arr[:, t] = cash
        assets_arr[:, t] = assets
        hgb_gap_arr[:, t] = hgb_gap
        hgb_pay_arr[:, t] = hgb_pay
        perf_arr[:, t] = perf
        add_arr[:, t] = add
        bund_arr[:, t] = bund
        ret_arr[:, t] = ret

    return WaterfallResult(
        fee=fee_arr, cash=cash_arr, assets=assets_arr,
        hgb_gap=hgb_gap_arr, hgb_payment=hgb_pay_arr,
        performance_payment=perf_arr, additional_payment=add_arr,
        bund_comparator=bund_arr, net_asset_return=ret_arr,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CDI simulation result
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CDIResult:
    wf: WaterfallResult
    sched: Schedule
    bond_sim: BondSimResult
    asset_cf: np.ndarray
    asset_pv: np.ndarray
    n_sim: int

    _Q = [0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995]

    def _df(self) -> np.ndarray:
        return np.tile(self.sched.df, (self.n_sim, 1))

    def pv_series(self, name: str) -> np.ndarray:
        arr = getattr(self.wf, name)
        return (arr * self._df()).sum(axis=1)

    def summary(self) -> pd.DataFrame:
        pv_hgb = self.pv_series("hgb_payment")
        pv_perf = self.pv_series("performance_payment")
        pv_fee = self.pv_series("fee")
        pv_net = pv_fee - pv_hgb - pv_perf

        df = pd.DataFrame({
            "pv_hgb": pv_hgb, "pv_perf": pv_perf,
            "pv_fee": pv_fee, "pv_net": pv_net,
        })
        tbl = df.quantile(self._Q).T
        tbl.insert(0, "mean", df.mean())
        return tbl

    def to_dataframe(self) -> pd.DataFrame:
        """Full scenario × time DataFrame."""
        T = self.sched.T
        tile = lambda x: np.tile(x, (self.n_sim, 1))
        d = {
            "dt": tile(self.sched.dt), "liab_cf": tile(self.sched.liab_cf),
            "fwd": tile(self.sched.fwds), "yield": tile(self.sched.yields),
            "asset_cf": self.asset_cf, "asset_pv": self.asset_pv,
            "fee": self.wf.fee, "cash": self.wf.cash, "assets": self.wf.assets,
            "hgb_gap": self.wf.hgb_gap, "hgb_payment": self.wf.hgb_payment,
            "perf_payment": self.wf.performance_payment,
            "add_payment": self.wf.additional_payment,
            "bund_comp": self.wf.bund_comparator,
            "net_ret": self.wf.net_asset_return,
        }
        idx = pd.MultiIndex.from_product(
            [range(self.n_sim), self.sched.dates[:T]], names=["sim", "date"],
        )
        df = pd.DataFrame({k: v.ravel() for k, v in d.items()}, index=idx)
        df["fl_gaap"] = df["assets"] / np.tile(self.sched.liab_pv_gaap, self.n_sim)
        return df.reset_index()


# ═══════════════════════════════════════════════════════════════════════════
#  ICAAP result
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ICAAPResult:
    pv_hgb: np.ndarray          # (n_outer,)
    pv_perf: np.ndarray
    pv_fee: np.ndarray

    _Q = [0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995]

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "pv_hgb_t1": self.pv_hgb,
            "pv_perf_t1": self.pv_perf,
            "pv_fee_t1": self.pv_fee,
        })
        tbl = df.quantile(self._Q).T
        tbl.insert(0, "mean", df.mean())
        return tbl


# ═══════════════════════════════════════════════════════════════════════════
#  CDI mandate engine
# ═══════════════════════════════════════════════════════════════════════════
class CDIMandate:
    """
    Orchestrates the full CDI simulation pipeline.

    All randomness flows through explicit ``np.random.Generator`` instances
    so that results are reproducible and invariant to chunking.
    """

    def __init__(
        self,
        bonds: BondUniverse,
        liabs: Liabilities,
        cfg: MandateConfig,
        cdi_weights: np.ndarray,
        cmbp_weights: np.ndarray,
        cash_0: float,
    ):
        self.bonds = bonds
        self.liabs = liabs
        self.cfg = cfg
        self.cdi_w = np.asarray(cdi_weights, dtype=np.float64)
        self.cmbp_w = np.asarray(cmbp_weights, dtype=np.float64)
        self.cash_0 = cash_0

    # ── base case ─────────────────────────────────────────────────────────
    def run(
        self,
        val_date: pd.Timestamp,
        curve: YieldCurve,
        cr: CreditSimResult,
    ) -> CDIResult:
        sched = build_schedule(
            val_date, curve, self.liabs, self.cfg,
            self.bonds, self.cdi_w, self.cmbp_w,
            T=cr.n_years,
        )
        T = sched.T
        n_sim = cr.n_sim

        # Bond sim (paths exclude year 0)
        bond_sim = self.bonds.simulate(
            val_date, curve, cr.paths[:, 1:, :], cr.schema.default_idx,
        )
        a_cf = bond_sim.portfolio(self.cdi_w, "cf")[:, :T]
        a_pv = bond_sim.portfolio(self.cdi_w, "pv")[:, :T]

        # t = 0 asset value
        bond_rating_idx = cr.paths[0, 0, self.bonds.issuer_ids]    # same for all sims
        pv0 = self.bonds.pv0(val_date, curve, bond_rating_idx)
        cdi_v0 = float((self.cdi_w * pv0).sum())
        total_a0 = cdi_v0 + self.cash_0
        gap0 = self.cfg.heubeck_liabilities * self.cfg.mortality_buffer - (total_a0 + self.cfg.asset_buffer_0)

        wf = run_waterfall(
            n_sim=n_sim, T=T, cfg=self.cfg,
            sched_fwds=sched.fwds, sched_liab_cf=sched.liab_cf,
            sched_liab_pv_gaap=sched.liab_pv_gaap,
            sched_liab_pv_ifrs=sched.liab_pv_ifrs,
            sched_meltdown=sched.meltdown, sched_next2=sched.next2,
            sched_ab=sched.asset_buffer, sched_bt=sched.bt,
            asset_cf=a_cf, asset_pv=a_pv,
            cash_0=np.full(n_sim, self.cash_0),
            assets_0=np.full(n_sim, total_a0),
            hgb_gap_0=np.full(n_sim, gap0),
            cum_hgb_0=np.zeros(n_sim),
            bund_0=np.full(n_sim, total_a0),
            add_pay_idx=self.cfg.additional_payment_year - 1,
            perf_pay_idx=self.cfg.performance_payment_year - 1,
        )

        return CDIResult(
            wf=wf, sched=sched, bond_sim=bond_sim,
            asset_cf=a_cf, asset_pv=a_pv, n_sim=n_sim,
        )

    # ── ICAAP ─────────────────────────────────────────────────────────────
    def run_icaap(
        self,
        val_date: pd.Timestamp,
        curve: YieldCurve,
        cr_model: CreditModel,
        rng: np.random.Generator,
        n_outer: int = 100,
        n_inner: int = 100,
        n_years: int = 25,
        chunk_size: int = 50,
    ) -> ICAAPResult:
        """
        Nested Monte-Carlo.

        All random draws are generated *before* the chunk loop so results
        are invariant to ``chunk_size``.
        """
        cfg = self.cfg
        n_inner_years = n_years - 1
        N_all = n_outer * n_inner

        # ── outer 1-year simulation ───────────────────────────────────────
        outer_rng = np.random.Generator(np.random.PCG64(rng.integers(2**63)))
        outer_cr = cr_model.simulate(outer_rng, n_outer, 1)
        outer_cdi = self.run(val_date, curve, outer_cr)

        # ── deterministic inner schedule (year 2 onwards) ─────────────────
        full_sched = build_schedule(
            val_date, curve, self.liabs, self.cfg,
            self.bonds, self.cdi_w, self.cmbp_w,
            T=n_years,
        )
        T_in = full_sched.T - 1
        inner_dates = full_sched.dates[1:]
        inner_val_date = full_sched.dates[0]
        s = full_sched                                             # alias for brevity
        in_fwds = s.fwds[1:];      in_yields = s.yields[1:]
        in_liab = s.liab_cf[1:];   in_melt = s.meltdown[1:]
        in_next2 = s.next2[1:];    in_ab = s.asset_buffer[1:]
        in_gaap = s.liab_pv_gaap[1:]; in_ifrs = s.liab_pv_ifrs[1:]
        in_dt = s.dt[1:] - s.dt[0]
        in_df = (1 + in_yields) ** (-in_dt)
        in_bt = s.bt[1:]

        # ── pre-generate ALL inner random draws ───────────────────────────
        inner_rng = np.random.Generator(np.random.PCG64(rng.integers(2**63)))
        pX_all = cr_model.generate_pX(inner_rng, N_all, n_inner_years)

        # ── outer-year starting conditions ────────────────────────────────
        year1_idx = outer_cr.year1_indices()                       # (n_outer, n_issuers)
        outer_bond_def = (
            outer_cdi.bond_sim.bond_paths[:, 0, :] == cr_model.schema.default_idx
        )

        # Waterfall starting state from outer sim
        o_wf = outer_cdi.wf
        o_cash = o_wf.cash[:, 0]                                   # outer sim is 1-year
        o_assets = o_wf.assets[:, 0]
        o_gap = o_wf.hgb_gap[:, 0]
        o_hgb = o_wf.hgb_payment[:, 0]
        o_bund = o_wf.bund_comparator[:, 0]

        # ── chunked inner bond simulation ─────────────────────────────────
        a_cf_3d = np.zeros((n_outer, n_inner, T_in))
        a_pv_3d = np.zeros((n_outer, n_inner, T_in))

        outer_offsets = np.arange(n_outer)[:, np.newaxis] * n_inner

        for c0 in range(0, n_inner, chunk_size):
            c1 = min(c0 + chunk_size, n_inner)
            nc = c1 - c0
            Nc = nc * n_outer

            # Gather correct pX rows for this chunk
            inner_offsets = np.arange(c0, c1)[np.newaxis, :]
            rows = (outer_offsets + inner_offsets).ravel()
            pX_c = pX_all[rows]

            init_c = np.repeat(year1_idx, nc, axis=0)
            cr_c = cr_model.simulate(inner_rng, Nc, n_inner_years, initial=init_c, pX=pX_c)

            bsim_c = self.bonds.simulate(
                inner_val_date, curve, cr_c.paths[:, 1:, :], cr_model.schema.default_idx,
            )

            # Double-recovery fix
            def_c = np.repeat(outer_bond_def, nc, axis=0)
            bsim_c.total_cf[:, 0, :] -= bsim_c.total_cf[:, 0, :] * def_c
            # (just zero out the recovery component for already-defaulted bonds)

            cf_c = bsim_c.portfolio(self.cdi_w, "cf")[:, :T_in]
            pv_c = bsim_c.portfolio(self.cdi_w, "pv")[:, :T_in]

            a_cf_3d[:, c0:c1, :] = cf_c.reshape(n_outer, nc, T_in)
            a_pv_3d[:, c0:c1, :] = pv_c.reshape(n_outer, nc, T_in)

            del bsim_c, cr_c, pX_c

        del pX_all

        a_cf = a_cf_3d.reshape(N_all, T_in)
        a_pv = a_pv_3d.reshape(N_all, T_in)
        del a_cf_3d, a_pv_3d

        # ── inner waterfall ───────────────────────────────────────────────
        perf_idx = cfg.performance_payment_year - 2

        wf_in = run_waterfall(
            n_sim=N_all, T=T_in, cfg=cfg,
            sched_fwds=in_fwds, sched_liab_cf=in_liab,
            sched_liab_pv_gaap=in_gaap, sched_liab_pv_ifrs=in_ifrs,
            sched_meltdown=in_melt, sched_next2=in_next2,
            sched_ab=in_ab, sched_bt=in_bt,
            asset_cf=a_cf, asset_pv=a_pv,
            cash_0=np.repeat(o_cash, n_inner),
            assets_0=np.repeat(o_assets, n_inner),
            hgb_gap_0=np.repeat(o_gap, n_inner),
            cum_hgb_0=np.repeat(o_hgb, n_inner),
            bund_0=np.repeat(o_bund, n_inner),
            add_pay_idx=cfg.additional_payment_year - 2,
            perf_pay_idx=perf_idx,
        )

        # ── aggregate ─────────────────────────────────────────────────────
        if 0 <= perf_idx < T_in:
            pv_perf = wf_in.performance_payment[:, perf_idx] * in_df[perf_idx]
        else:
            pv_perf = np.zeros(N_all)
        pv_hgb = (wf_in.hgb_payment * in_df[np.newaxis, :]).sum(axis=1)
        pv_fee = (wf_in.fee * in_df[np.newaxis, :]).sum(axis=1)

        return ICAAPResult(
            pv_hgb=pv_hgb.reshape(n_outer, n_inner).mean(axis=1),
            pv_perf=pv_perf.reshape(n_outer, n_inner).mean(axis=1),
            pv_fee=pv_fee.reshape(n_outer, n_inner).mean(axis=1),
        )
