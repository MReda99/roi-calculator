from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.stats import beta, triang, norm

# ------------------------------
# Helpers
# ------------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))

def beta_from_mean_ci(mean: float, low: float, high: float, z: float = 1.96) -> Tuple[float, float]:
    """
    Approximate Beta(alpha,beta) parameters from mean and 95% CI endpoints.
    If CI is narrow, distribution is tight; if wide, it's looser.
    """
    # approximate variance from CI
    var = ((high - low) / (2 * z)) ** 2
    mean = np.clip(mean, 1e-6, 1 - 1e-6)
    # method-of-moments for Beta
    tmp = mean * (1 - mean) / var - 1
    a = mean * tmp
    b = (1 - mean) * tmp
    if a <= 0 or b <= 0 or np.isnan(a) or np.isnan(b):
        # fallback: mildly-informative
        a, b = max(mean * 100, 1.0), max((1 - mean) * 100, 1.0)
    return a, b

def pct(x): 
    return f"{x*100:,.1f}%"

# ------------------------------
# Core data structures
# ------------------------------

@dataclass
class FunnelInputs:
    # Audience and throughput
    weekly_reachable_contacts: int
    weeks: int
    # Compliance mode: '506b' or '506c'
    mode: str  # '506b' or '506c'

    # Base rates (means) as decimals, with conservative and upside bands for Beta
    open_rate_mean: float; open_rate_low: float; open_rate_high: float
    reply_rate_mean: float; reply_rate_low: float; reply_rate_high: float
    booking_rate_mean: float; booking_rate_low: float; booking_rate_high: float
    show_rate_mean: float; show_rate_low: float; show_rate_high: float
    close_rate_mean: float; close_rate_low: float; close_rate_high: float

    # Deal economics
    avg_check_usd: float
    carry_rate: float          # sponsor/promo economics if relevant (0–1)
    fee_rate: float            # acquisition/management fee % of capital year 1

    # Costs
    fixed_fee_usd: float       # your fixed project fee (Phase 0 + Phase 1)
    variable_cost_per_contact: float  # list verification, sends, enrichment, etc.
    infra_pass_through_usd: float     # Digital Twin infra

    # Deliverability & list quality uplift parameters (multipliers)
    deliverability_uplift: float      # e.g., 1.15 after SPF/DKIM/DMARC + warm-up
    list_quality_uplift: float        # e.g., 1.05 after hygiene

    # Universe multipliers by compliance mode
    universe_multiplier_506b: float   # typically lower (relationships only)
    universe_multiplier_506c: float   # broader top-of-funnel

@dataclass
class SimConfig:
    simulations: int = 100000
    seed: int = 42

# ------------------------------
# Simulation engine
# ------------------------------

def simulate(f: FunnelInputs, cfg: SimConfig) -> Dict:
    rng = np.random.default_rng(cfg.seed)

    # Compliance effect on weekly reach
    universe_mult = f.universe_multiplier_506b if f.mode.lower() == "506b" else f.universe_multiplier_506c
    weekly_contacts = int(f.weekly_reachable_contacts * universe_mult)

    # Costs
    total_contacts = weekly_contacts * f.weeks
    variable_costs = total_contacts * f.variable_cost_per_contact
    fixed_costs = f.fixed_fee_usd + f.infra_pass_through_usd
    base_costs = fixed_costs + variable_costs

    # Beta distributions for each rate (bounded 0–1)
    a_o, b_o = beta_from_mean_ci(f.open_rate_mean, f.open_rate_low, f.open_rate_high)
    a_r, b_r = beta_from_mean_ci(f.reply_rate_mean, f.reply_rate_low, f.reply_rate_high)
    a_b, b_b = beta_from_mean_ci(f.booking_rate_mean, f.booking_rate_low, f.booking_rate_high)
    a_s, b_s = beta_from_mean_ci(f.show_rate_mean, f.show_rate_low, f.show_rate_high)
    a_c, b_c = beta_from_mean_ci(f.close_rate_mean, f.close_rate_low, f.close_rate_high)

    # Draws
    open_rate   = clamp01(beta(a_o, b_o).rvs(cfg.simulations, random_state=rng)) * f.deliverability_uplift * f.list_quality_uplift
    reply_rate  = clamp01(beta(a_r, b_r).rvs(cfg.simulations, random_state=rng))
    booking_rate= clamp01(beta(a_b, b_b).rvs(cfg.simulations, random_state=rng))
    show_rate   = clamp01(beta(a_s, b_s).rvs(cfg.simulations, random_state=rng))
    close_rate  = clamp01(beta(a_c, b_c).rvs(cfg.simulations, random_state=rng))

    # Top-of-funnel volume each simulation
    contacts = np.full(cfg.simulations, total_contacts, dtype=float)
    opens    = contacts * open_rate
    replies  = opens    * reply_rate
    bookings = replies  * booking_rate
    shows    = bookings * show_rate
    closes   = shows    * close_rate

    committed_capital = closes * f.avg_check_usd
    year1_fees = committed_capital * f.fee_rate
    year1_carry= committed_capital * f.carry_rate

    revenue_gross = year1_fees + year1_carry
    profit = revenue_gross - base_costs

    # KPI summaries
    out = {
        "contacts_total": total_contacts,
        "base_costs": base_costs,
        "variable_costs": variable_costs,
        "fixed_costs": fixed_costs,
        "samples": pd.DataFrame({
            "opens": opens, "replies": replies, "bookings": bookings,
            "shows": shows, "closes": closes,
            "capital": committed_capital,
            "rev": revenue_gross, "profit": profit
        })
    }
    return out

# ------------------------------
# Scenarios and sensitivity
# ------------------------------

def scenario_table(f: FunnelInputs, base_sim: Dict) -> pd.DataFrame:
    """
    Builds worst / conservative / expected / upside scenarios by pinning rates.
    """
    # Helper to run a deterministic single-sim simulation
    def run_with(means: Dict[str, float]) -> Dict[str, float]:
        weekly_contacts = int(f.weekly_reachable_contacts * (f.universe_multiplier_506b if f.mode=='506b' else f.universe_multiplier_506c))
        total_contacts = weekly_contacts * f.weeks
        o = total_contacts * means["open"]
        r = o * means["reply"]
        b = r * means["book"]
        s = b * means["show"]
        c = s * means["close"]
        capital = c * f.avg_check_usd
        rev = capital * f.fee_rate + capital * f.carry_rate
        profit = rev - (f.fixed_fee_usd + f.infra_pass_through_usd + total_contacts * f.variable_cost_per_contact)
        return {
            "Contacts": total_contacts, "Opens": o, "Replies": r, "Bookings": b,
            "Shows": s, "Closes": c, "Capital ($)": capital, "Revenue ($)": rev, "Profit ($)": profit
        }

    worst = run_with({
        "open": max(0.01, f.open_rate_low) * f.deliverability_uplift * f.list_quality_uplift,
        "reply": max(0.005, f.reply_rate_low),
        "book": max(0.01, f.booking_rate_low),
        "show": max(0.30, f.show_rate_low),
        "close": max(0.05, f.close_rate_low),
    })
    conservative = run_with({
        "open": f.open_rate_mean * f.deliverability_uplift * f.list_quality_uplift,
        "reply": f.reply_rate_mean,
        "book": f.booking_rate_mean,
        "show": f.show_rate_mean,
        "close": f.close_rate_mean,
    })
    expected = base_sim["samples"][["capital","rev","profit"]].median().to_dict()
    # Need full funnel medians for expected:
    sf = base_sim["samples"]
    expected_full = {
        "Contacts": base_sim["contacts_total"],
        "Opens": sf["opens"].median(),
        "Replies": sf["replies"].median(),
        "Bookings": sf["bookings"].median(),
        "Shows": sf["shows"].median(),
        "Closes": sf["closes"].median(),
        "Capital ($)": expected["capital"],
        "Revenue ($)": expected["rev"],
        "Profit ($)": expected["profit"]
    }
    upside = run_with({
        "open": min(0.95, f.open_rate_high) * f.deliverability_uplift * f.list_quality_uplift,
        "reply": min(0.50, f.reply_rate_high),
        "book": min(0.70, f.booking_rate_high),
        "show": min(0.95, f.show_rate_high),
        "close": min(0.60, f.close_rate_high),
    })

    table = pd.DataFrame([worst, conservative, expected_full, upside],
                         index=["Worst case", "Conservative", "Expected (median)", "Upside"])
    return table

def one_at_a_time_sensitivity(f: FunnelInputs, base_sim: Dict, step: float = 0.05) -> pd.DataFrame:
    """
    Perturb each rate +/- step to estimate local sensitivity on profit.
    """
    def eval_profit(**kw):
        f2 = FunnelInputs(**{**f.__dict__, **kw})
        sim = simulate(f2, SimConfig(simulations=20000, seed=7))
        return sim["samples"]["profit"].median()

    baseline = base_sim["samples"]["profit"].median()
    factors = [
        ("Open rate",   "open_rate_mean"),
        ("Reply rate",  "reply_rate_mean"),
        ("Booking rate","booking_rate_mean"),
        ("Show rate",   "show_rate_mean"),
        ("Close rate",  "close_rate_mean"),
        ("Avg check",   "avg_check_usd"),
        ("Fee rate",    "fee_rate"),
        ("Carry rate",  "carry_rate"),
        ("Variable cost/contact","variable_cost_per_contact")
    ]
    rows = []
    for label, field in factors:
        v = getattr(f, field)
        if "rate" in field or "fee" in field or "carry" in field:
            v_up = min(0.999, v*(1+step))
            v_dn = max(0.0001, v*(1-step))
        elif "avg_check" in field:
            v_up = v*(1+step)
            v_dn = v*(1-step)
        else:
            v_up = v*(1+step)
            v_dn = v*(1-step)
        p_up = eval_profit(**{field: v_up})
        p_dn = eval_profit(**{field: v_dn})
        rows.append({"Factor": label, "Minus": p_dn, "Plus": p_up, "Range": p_up - p_dn, "Baseline": baseline})
    df = pd.DataFrame(rows).sort_values("Range", ascending=False).reset_index(drop=True)
    return df