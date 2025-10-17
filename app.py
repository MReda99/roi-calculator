# app.py
# Streamlit Raise ROI & Risk Simulator (Phase 0–3) — production-ready
# Run: streamlit run app.py

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dataclasses import dataclass

# ---------------------------
# Config & Defaults
# ---------------------------

DEFAULT_CONFIG = {
    "pricing": {
        "echelon_fee_phase01": 4800.0,   # your Phase 0–1 service fee
        "infra_cap_phase3": 1000.0,      # pass-through infra cap (upper bound)
        "per_message_cost": 0.008        # optional marginal cost per outbound
    },
    "financials": {
        "avg_check_size": 250000.0,      # average committed capital per "yes"
        "close_rate_cap": 1.0,           # safety max for close rate
        "max_contacts_week": 100000
    },
    "bands": {
        # open rate of delivered messages
        "open_mean": 0.32, "open_low": 0.18, "open_high": 0.48,
        # reply is % of opens
        "reply_mean": 0.07, "reply_low": 0.03, "reply_high": 0.12,
        # booking is % of replies
        "book_mean": 0.20, "book_low": 0.08, "book_high": 0.35,
        # show is % of bookings
        "show_mean": 0.70, "show_low": 0.50, "show_high": 0.85,
        # close is % of shows
        "close_mean": 0.14, "close_low": 0.05, "close_high": 0.25,
        # deliverability uplift (Phase 0 hygiene; Phase 2 tweaks may add a small lift)
        "deliver_uplift_mean": 0.90, "deliver_uplift_low": 0.80, "deliver_uplift_high": 0.96
    },
    "phase_multipliers": {
        # these are additive “gravy” multipliers applied AFTER Phase 1 if toggled
        # they scale reply and booking primarily (content resonance, better hooks), and show-rate modestly
        "phase2": {"reply_mult": 1.20, "book_mult": 1.25, "show_mult": 1.05, "close_mult": 1.00},
        "phase3": {"reply_mult": 1.10, "book_mult": 1.10, "show_mult": 1.05, "close_mult": 1.00}
    },
    "sim": {
        "iterations": 100000,           # fast but statistically meaningful
        "random_seed": 42
    }
}

def load_config():
    # For Streamlit Cloud deployment, just use defaults
    # Config file loading removed to avoid yaml dependency issues
    return DEFAULT_CONFIG

CFG = load_config()
np.random.seed(CFG["sim"]["random_seed"])

# ---------------------------
# Helpers
# ---------------------------

def clamp(p, lo=0.0, hi=1.0):
    return np.clip(p, lo, hi)

def triangular(n, low, mode, high):
    # Ensure valid triangular distribution parameters: low <= mode <= high
    low = max(0.0, low)
    high = min(1.0, high)
    mode = max(low, min(mode, high))  # Clamp mode between low and high
    return np.random.triangular(left=low, mode=mode, right=high, size=n)

def dollars(x): return f"${x:,.0f}"
def dollars_latex(x): return f"\\${x:,.0f}"
def pct(x): return f"{x*100:.1f}%"

@dataclass
class Inputs:
    rule: str                # "506(b)" or "506(c)"
    weeks: int
    contacts_week: int
    preset: str              # "Conservative" | "Base" | "Upside"
    add_phase2: bool
    add_phase3: bool
    per_msg_cost: float
    fee_phase01: float
    infra_cap_phase3: float
    avg_check: float

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Raise ROI & Risk Simulator", layout="wide")

st.title("Raise ROI & Risk Simulator")
st.caption("Ultra-conservative planning with Monte Carlo, compliance modes, and measurable deliverability uplift.")

# Phase selector (positioned top, easy to see)
colp1, colp2, colp3, colp4 = st.columns([1.2,1,1,1])
with colp1:
    phase_label = st.radio("Plan scope", ["Phase 0–1 only", "Add Phase 2", "Add Phase 3 (with Phase 2)"],
                           help="Phase 0–1 = foundations + first qualified calls (no influencer content required). "
                                "Phase 2–3 are additive 'gravy' lifts.")
add_p2 = (phase_label != "Phase 0–1 only")
add_p3 = (phase_label == "Add Phase 3 (with Phase 2)")

# Sidebar with cleaner grouped controls
with st.sidebar:
    st.header("Assumptions")

    compliance_mode = st.selectbox("Offering type", ["Rule 506(b)", "Rule 506(c)"])

    planning_weeks = st.number_input("Planning window (weeks)", 4, 26, 10)
    weekly_contacts = st.number_input("Reachable people per week", 200, 10000, 2500, step=100)

    st.markdown("### Business inputs")
    avg_check_size = st.number_input("Average investment per close ($)", 25000, 1000000, 100000, step=5000)
    total_costs = st.number_input("All-in costs this window ($)", 1000, 50000, 10800, step=500)
    # value_view = st.selectbox("View metrics as", ["Michael (capital committed)", "Vendor ROI view"])
    # value_factor = 1.00 if value_view.startswith("Michael") else st.number_input("Value factor for ROI view", 0.0, 1.0, 0.10, step=0.01)

    st.session_state["avg_check_size"] = avg_check_size
    st.session_state["total_costs"] = total_costs
    # st.session_state["value_factor"] = value_factor

    st.markdown("### Conversion (conservative defaults)")
    # A single slider per stage (mean). Lows/highs go in Advanced.
    open_mean    = st.slider("People who open (%)", 0.05, 0.80, 0.32)
    reply_mean   = st.slider("Openers who reply (%)", 0.01, 0.40, 0.07)
    booking_mean = st.slider("Repliers who book (%)", 0.02, 0.60, 0.20)
    show_mean    = st.slider("Bookings that show (%)", 0.20, 0.95, 0.70)
    close_mean   = st.slider("Shows that invest (%)", 0.01, 0.50, 0.12)

    with st.expander("Advanced variability"):
        open_low, open_high = st.slider("Open rate band", 0.05, 0.80, (0.18, 0.48))
        reply_low, reply_high = st.slider("Reply rate band", 0.01, 0.40, (0.03, 0.12))
        booking_low, booking_high = st.slider("Booking rate band", 0.02, 0.60, (0.08, 0.35))
        show_low, show_high = st.slider("Show rate band", 0.20, 0.95, (0.50, 0.85))
        close_low, close_high = st.slider("Close rate band", 0.01, 0.50, (0.05, 0.25))

# Create bands dictionary from the new inputs
bands = {
    "open_low": open_low, "open_mean": open_mean, "open_high": open_high,
    "reply_low": reply_low, "reply_mean": reply_mean, "reply_high": reply_high,
    "book_low": booking_low, "book_mean": booking_mean, "book_high": booking_high,
    "show_low": show_low, "show_mean": show_mean, "show_high": show_high,
    "close_low": close_low, "close_mean": close_mean, "close_high": close_high,
    "deliver_uplift_low": CFG["bands"]["deliver_uplift_low"], 
    "deliver_uplift_mean": CFG["bands"]["deliver_uplift_mean"], 
    "deliver_uplift_high": CFG["bands"]["deliver_uplift_high"]
}

inp = Inputs(
    rule = "506(b)" if "b" in compliance_mode else "506(c)",
    weeks = planning_weeks,
    contacts_week = weekly_contacts,
    preset = "Base",  # Simplified since we're using direct sliders
    add_phase2 = add_p2,
    add_phase3 = add_p3,
    per_msg_cost = CFG["pricing"]["per_message_cost"],  # Use config default
    fee_phase01 = CFG["pricing"]["echelon_fee_phase01"],  # Use config default
    infra_cap_phase3 = CFG["pricing"]["infra_cap_phase3"],  # Use config default
    avg_check = avg_check_size
)

# ---------------------------
# Monte Carlo engine
# ---------------------------

def run_sim(inp: Inputs, bands: dict, iterations: int):
    n = iterations
    contacts_total = inp.contacts_week * inp.weeks

    # deliverability multiplier (Phase 0 hygiene + operational fixes)
    deliver = triangular(n, bands["deliver_uplift_low"], bands["deliver_uplift_mean"], bands["deliver_uplift_high"])

    open_r = clamp(triangular(n, bands["open_low"], bands["open_mean"], bands["open_high"]))
    reply_r = clamp(triangular(n, bands["reply_low"], bands["reply_mean"], bands["reply_high"]))
    book_r = clamp(triangular(n, bands["book_low"], bands["book_mean"], bands["book_high"]))
    show_r = clamp(triangular(n, bands["show_low"], bands["show_mean"], bands["show_high"]))
    close_r = clamp(triangular(n, bands["close_low"], bands["close_mean"], bands["close_high"]), 0.0, CFG["financials"]["close_rate_cap"])

    # Apply “gravy” multipliers when Phase 2 / Phase 3 included
    rp_mult = 1.0
    bk_mult = 1.0
    sh_mult = 1.0
    cl_mult = 1.0
    if inp.add_phase2:
        m2 = CFG["phase_multipliers"]["phase2"]
        rp_mult *= m2["reply_mult"]; bk_mult *= m2["book_mult"]; sh_mult *= m2["show_mult"]; cl_mult *= m2["close_mult"]
    if inp.add_phase3:
        m3 = CFG["phase_multipliers"]["phase3"]
        rp_mult *= m3["reply_mult"]; bk_mult *= m3["book_mult"]; sh_mult *= m3["show_mult"]; cl_mult *= m3["close_mult"]

    reply_r = clamp(reply_r * rp_mult)
    book_r  = clamp(book_r  * bk_mult)
    show_r  = clamp(show_r  * sh_mult)
    close_r = clamp(close_r * cl_mult)

    delivered = np.round(contacts_total * deliver).astype(int)
    opens    = np.round(delivered * open_r).astype(int)
    replies  = np.round(opens * reply_r).astype(int)
    bookings = np.round(replies * book_r).astype(int)
    shows    = np.round(bookings * show_r).astype(int)
    closes   = np.round(shows * close_r).astype(int)

    capital  = closes * inp.avg_check

    # Costs: fixed Phase 0–1 fee (always), Phase 3 infra only if Phase 3 is selected, plus per-message marginal
    fixed_fees = inp.fee_phase01 + (inp.infra_cap_phase3 if inp.add_phase3 else 0.0)
    msg_costs = delivered * inp.per_msg_cost
    total_cost = fixed_fees + msg_costs
    profit = capital - total_cost

    df = pd.DataFrame({
        "delivered": delivered, "opens": opens, "replies": replies, "bookings": bookings,
        "shows": shows, "closes": closes, "capital": capital, "profit": profit
    })
    return df, contacts_total, fixed_fees

sim_df, contacts_total, fixed_fees = run_sim(inp, bands, CFG["sim"]["iterations"])

def quantiles(s):
    return {
        "p05": np.percentile(s, 5),
        "p50": np.percentile(s, 50),
        "p95": np.percentile(s, 95)
    }

q_cap = quantiles(sim_df["capital"])
q_profit = quantiles(sim_df["profit"])
breakeven_prob = (sim_df["profit"] >= 0).mean()

# ---------------------------
# KPI header
# ---------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Median capital committed", dollars(q_cap["p50"]))
c2.metric("Median profit after costs", dollars(q_profit["p50"]))
c3.metric("Chance of breakeven or better", pct(breakeven_prob))
c4.metric("5th percentile profit (conservative)", dollars(q_profit["p05"]))

st.caption(
    f"Assumes **{inp.rule}**, **{inp.weeks} weeks**, **{inp.contacts_week:,}/week** reachable people. "
    f"Fixed fees included: {dollars(fixed_fees)}; marginal costs scale with delivered volume."
)

# ---------------------------
# Unit Economics (Median Path)
# ---------------------------

st.markdown("---")

# Calculate median scenario values
median_contacts = int(contacts_total)
median_delivered = int(sim_df["delivered"].median())
median_opens = int(sim_df["opens"].median())
median_replies = int(sim_df["replies"].median())
median_bookings = int(sim_df["bookings"].median())
median_shows = int(sim_df["shows"].median())
median_closes = int(sim_df["closes"].median())

# Calculate unit economics
capital_committed = median_closes * inp.avg_check
total_costs = fixed_fees + (median_delivered * inp.per_msg_cost)
net_benefit = capital_committed - total_costs
cp_call = total_costs / max(median_bookings, 1)
cp_dollar = total_costs / max(capital_committed, 1) if capital_committed > 0 else 0

# Create unit economics dataframe
ue = pd.DataFrame([
    ["Contacts", f"{median_contacts:,}"],
    ["Delivered", f"{median_delivered:,}"],
    ["Opens", f"{median_opens:,}"],
    ["Replies", f"{median_replies:,}"],
    ["Bookings", f"{median_bookings:,}"],
    ["Shows", f"{median_shows:,}"],
    ["Closes", f"{median_closes:,}"],
    ["Average check size", f"${inp.avg_check:,.0f}"],
    ["Capital committed (median)", f"${capital_committed:,.0f}"],
    ["Total costs modeled", f"${total_costs:,.0f}"],
    ["Cost per booked call", f"${cp_call:,.0f}"],
    ["Cost per dollar raised", f"${cp_dollar:.3f}"],
    ["Net benefit (median)", f"${net_benefit:,.0f}"],
], columns=["Metric", "Value"])

st.subheader("Unit economics (median path)")
st.dataframe(ue, use_container_width=True)

# ---------------------------
# Tabs
# ---------------------------

tab_funnel, tab_dist, tab_scen, tab_sens = st.tabs(["Funnel", "Distribution", "Scenarios", "Sensitivity"])

with tab_funnel:
    # Median path for the funnel + table (p05, p50, p95)
    med = sim_df.quantile(0.50).round(0).astype(int)
    lo  = sim_df.quantile(0.05).round(0).astype(int)
    hi  = sim_df.quantile(0.95).round(0).astype(int)

    stages = ["Contacts", "Delivered", "Opens", "Replies", "Bookings", "Shows", "Closes"]
    values = [contacts_total, med["delivered"], med["opens"], med["replies"], med["bookings"], med["shows"], med["closes"]]

    fig = go.Figure(go.Funnel(
        y = stages,
        x = values,
        textinfo="value+percent previous"
    ))
    fig.update_layout(height=440, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Tabular breakdown
    table = pd.DataFrame({
        "Stage": ["Contacts", "Delivered", "Opens", "Replies", "Bookings", "Shows", "Closes", "Capital ($)", "Profit ($)"],
        "p05":   [contacts_total, lo["delivered"], lo["opens"], lo["replies"], lo["bookings"], lo["shows"], lo["closes"], round(q_cap["p05"]), round(q_profit["p05"])],
        "p50":   [contacts_total, med["delivered"], med["opens"], med["replies"], med["bookings"], med["shows"], med["closes"], round(q_cap["p50"]), round(q_profit["p50"])],
        "p95":   [contacts_total, hi["delivered"], hi["opens"], hi["replies"], hi["bookings"], hi["shows"], hi["closes"], round(q_cap["p95"]), round(q_profit["p95"])]
    })
    st.subheader("Conservative / Median / Upside (5th / 50th / 95th percentiles)")
    st.dataframe(table, use_container_width=True)

    # Export simulation data
    csv = sim_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download raw simulation (CSV)", csv, "simulation.csv", "text/csv")

with tab_dist:
    st.subheader("Profit distribution")
    figd = px.histogram(sim_df, x="profit", nbins=60, opacity=0.85)
    figd.add_vline(x=q_profit["p05"], line_dash="dash", line_color="red")
    figd.add_vline(x=q_profit["p50"], line_dash="dash", line_color="black")
    figd.add_vline(x=q_profit["p95"], line_dash="dash", line_color="green")
    figd.update_layout(height=460, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(figd, use_container_width=True)
    st.caption(f"Shaded percentiles marked: 5th {dollars(q_profit['p05'])}, median {dollars(q_profit['p50'])}, 95th {dollars(q_profit['p95'])}. "
               f"Breakeven odds: {pct(breakeven_prob)}.")

with tab_scen:
    st.subheader("Side-by-side scenarios")
    # Build three fixed scenarios (conservative / base / upside) for Phase 0–1 only
    def scen(preset_name):
        _inp = Inputs(**{**inp.__dict__, "add_phase2": False, "add_phase3": False})
        if preset_name == "Conservative":
            bandsX = CFG["bands"]
        elif preset_name == "Upside":
            bandsX = {**CFG["bands"],
                      "reply_mean": min(CFG["bands"]["reply_high"]*0.95, 0.20),
                      "book_mean": min(CFG["bands"]["book_high"]*0.90, 0.40),
                      "show_mean": max(CFG["bands"]["show_mean"], 0.75)}
        else:
            bandsX = {**CFG["bands"],}  # Base == default bands
        dfX, _, _ = run_sim(_inp, bandsX, 50000)
        qp = quantiles(dfX["profit"]); qc = quantiles(dfX["capital"])
        return dollars(qc["p50"]), pct((dfX["profit"]>=0).mean()), dollars(qp["p05"]), dollars(qp["p50"])

    colA, colB, colC = st.columns(3)
    for col, name in zip([colA, colB, colC], ["Conservative", "Base", "Upside"]):
        med_cap, p_be, p05, p50 = scen(name)
        with col:
            st.markdown(f"### {name}")
            st.metric("Median capital", med_cap)
            st.metric("Breakeven odds", p_be)
            st.metric("5th percentile profit", p05)
            st.metric("Median profit", p50)
            st.caption("Phase 0–1 only (no influencer content required).")

    if inp.add_phase2 or inp.add_phase3:
        st.markdown("---")
        st.subheader("Incremental lift from Phase 2–3 ('gravy')")
        base_inp = Inputs(**{**inp.__dict__, "add_phase2": False, "add_phase3": False})
        base_df, _, _ = run_sim(base_inp, bands, 50000)
        gravy_df, _, _ = run_sim(inp, bands, 50000)
        inc = pd.DataFrame({
            "Metric": ["Median capital", "Median profit", "Breakeven odds"],
            "Base (P0–1)": [dollars(np.percentile(base_df["capital"],50)),
                            dollars(np.percentile(base_df["profit"],50)),
                            pct((base_df["profit"]>=0).mean())],
            "With Phase 2/3": [dollars(np.percentile(gravy_df["capital"],50)),
                               dollars(np.percentile(gravy_df["profit"],50)),
                               pct((gravy_df["profit"]>=0).mean())]
        })
        st.dataframe(inc, use_container_width=True)

with tab_sens:
    st.subheader("What moves the needle most?")
    # One-at-a-time sensitivity: +/- 10% on means
    drivers = {
        "Reply rate (of opens)": ("reply_mean", 0.10),
        "Booking rate (of replies)": ("book_mean", 0.10),
        "Show rate (of bookings)": ("show_mean", 0.10),
        "Close rate (of shows)": ("close_mean", 0.10),
        "Deliverability uplift": ("deliver_uplift_mean", 0.05),
        "Open rate (delivered)": ("open_mean", 0.10)
    }
    base_med = np.percentile(sim_df["profit"], 50)
    rows = []
    for label, (key, step) in drivers.items():
        b_up = dict(bands); b_dn = dict(bands)
        b_up[key] = clamp(bands[key]*(1+step), 0.0, 1.0)
        b_dn[key] = clamp(bands[key]*(1-step), 0.0, 1.0)
        up_df,_,_ = run_sim(inp, b_up, 20000)
        dn_df,_,_ = run_sim(inp, b_dn, 20000)
        rows.append([label, np.percentile(dn_df["profit"],50)-base_med, np.percentile(up_df["profit"],50)-base_med])
    sens = pd.DataFrame(rows, columns=["Driver","Down (Δ median profit)","Up (Δ median profit)"])
    figt = go.Figure()
    figt.add_trace(go.Bar(y=sens["Driver"], x=sens["Down (Δ median profit)"], orientation='h', name="Down"))
    figt.add_trace(go.Bar(y=sens["Driver"], x=sens["Up (Δ median profit)"], orientation='h', name="Up"))
    figt.update_layout(barmode="overlay", height=460, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(figt, use_container_width=True)
    st.dataframe(sens, use_container_width=True)

# ---------------------------
# Plain-English summary (sales psych)
# ---------------------------

st.markdown("---")
st.subheader("What this means, in plain English")
st.markdown(f"""
- With **{inp.rule}** and **{inp.contacts_week:,} people/week** for **{inp.weeks} weeks**, the **median** path yields **{dollars_latex(q_cap['p50'])}** committed and **{dollars_latex(q_profit['p50'])}** after fees.
- Even under a very conservative 5th-percentile case, profit is **{dollars_latex(q_profit['p05'])}**.
- The chance of breaking even or better is **{pct(breakeven_prob)}**.
- Phase 2–3 are **optional** and treated strictly as additive lift ('gravy'). Phase 0-1 alone is designed to stand on its own.
""")
