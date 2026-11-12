"""
Just Bet It dashboard: model predictions vs. sportsbook QB passing-yards lines.

Run from the project root (after python -m scripts.train):
    streamlit run app.py

Everything here reuses the pipeline's own code (feature building, models,
line selection, betting math) from scripts/ and src/, so the dashboard shows
exactly what the command-line scripts compute.
"""

import html
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Just Bet It", page_icon="🏈", layout="wide")

from scripts.backtest import american_payout, load_closing_prop_lines, simulate_betting_strategy  # noqa: E402
from scripts.weekly_report import CONFIDENT_RULE  # noqa: E402
from scripts.predict_upcoming import format_kickoff, load_upcoming_lines, upcoming_qbs  # noqa: E402
from scripts.train import MODELS_PATH, load_models, predict_all  # noqa: E402
from scripts.xgboost_reg import build_features, build_upcoming_features, has_history, load_qb_games  # noqa: E402
from scipy.stats import binom  # noqa: E402
from src.betting import (  # noqa: E402
    BetConfig, ResidualModel, best_bet, cap_exposure, evaluate_side, from_bundle, simulate_bankroll,
)
from src.db.session import SessionLocal, engine  # noqa: E402

OVER, UNDER, INK, MUTED, GOLD = "#1F7A4D", "#B03A2E", "#14213D", "#5B6573", "#C9A227"
# Chart legends above the plot, wrapping instead of running off the edge
LEGEND_TOP = alt.Legend(orient="top", title=None, columns=3, labelLimit=160)

# Walk-forward predictions for past games with lines (scripts/vs_closing_lines.py):
# each week predicted by models trained only on the games before it. The clean
# set (XGBoost settings tuned only on games before 2024) is preferred.
WALKFORWARD = next((f for f in (ROOT / "data" / "raw" / "walkforward_preds_pre2024.csv",
                                ROOT / "data" / "raw" / "walkforward_preds.csv") if f.exists()),
                   ROOT / "data" / "raw" / "walkforward_preds_pre2024.csv")
REPORTS = ROOT / "reports"
MODEL_NAMES = {
    "xgboost": "XGBoost", "ensemble": "Ensemble", "random_forest": "Random forest",
    "linear_regression": "Linear regression", "neural_network": "Neural net",
}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .brand {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.9rem;
            line-height: 1; color: {INK}; border-left: 6px solid {GOLD}; padding-left: .55rem; }}
  .brand-sub {{ color: {MUTED}; font-size: .9rem; margin: .35rem 0 0 .9rem; }}
  .summary {{ font-size: 1.08rem; color: {INK}; max-width: 70ch; margin: .2rem 0 1.2rem; }}
  .note {{ color: {MUTED}; font-size: .85rem; max-width: 80ch; }}

  .pick {{ display: grid; grid-template-columns: 180px 160px minmax(240px, 1fr) 140px 160px;
           gap: 1.1rem; align-items: center; padding: .9rem 0; border-bottom: 1px solid #D7DDD5; }}
  .pick:first-of-type {{ border-top: 1px solid #D7DDD5; }}
  .when {{ color: {MUTED}; font-size: .85rem; }}
  .matchup {{ font-weight: 600; }}
  .qb {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 600; line-height: 1.1; }}
  .team {{ color: {MUTED}; font-size: .85rem; }}
  .lean {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.25rem; font-weight: 700; }}
  .price {{ color: {MUTED}; font-size: .85rem; }}
  .ev {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 600; text-align: right; }}
  .ev small.stake {{ color: {INK}; font-weight: 600; }}
  .ev small {{ display: block; font-family: 'Barlow', sans-serif; font-size: .78rem; color: {MUTED};
               font-weight: 400; }}

  /* the yard strip: line vs model on a yardage scale */
  .strip {{ position: relative; height: 54px; }}
  .field {{ position: absolute; left: 0; right: 0; top: 14px; height: 16px; background: #DDE6DA;
            border-radius: 3px; }}
  .tick {{ position: absolute; top: 14px; height: 16px; width: 1px; background: #B9C6B5; }}
  .ticklab {{ position: absolute; top: 34px; transform: translateX(-50%); font-size: .7rem; color: #7D8794; }}
  .edge {{ position: absolute; top: 14px; height: 16px; opacity: .28; }}
  .lineMark {{ position: absolute; top: 8px; height: 28px; width: 3px; background: {INK};
               transform: translateX(-50%); border-radius: 1px; }}
  .modelMark {{ position: absolute; top: 15px; width: 14px; height: 14px; border-radius: 50%;
                transform: translateX(-50%); border: 2px solid #fff; }}
  .l5Mark {{ position: absolute; top: 17px; width: 10px; height: 10px; border-radius: 50%;
             transform: translateX(-50%); border: 2px solid {MUTED}; background: transparent; }}
  .marklab {{ position: absolute; top: -6px; transform: translateX(-50%); font-size: .74rem;
              white-space: nowrap; font-weight: 600; }}
  .legend {{ display: flex; gap: 1.4rem; flex-wrap: wrap; color: {MUTED}; font-size: .82rem; margin: .3rem 0 .6rem; }}
  .legend span {{ display: inline-flex; align-items: center; gap: .4rem; }}
  .sw {{ display: inline-block; }}

  @media (max-width: 900px) {{
    .pick {{ grid-template-columns: 1fr 1fr; }}
    .strip {{ grid-column: 1 / -1; order: 3; }}
    .ev {{ text-align: left; }}
  }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data access (cached; refreshed every few minutes or on "Refresh data")
# --------------------------------------------------------------------------
def model_mtime():
    return MODELS_PATH.stat().st_mtime if MODELS_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def _bundle(mtime):
    return load_models()


def bundle():
    mtime = model_mtime()
    return _bundle(mtime) if mtime else None


@st.cache_data(ttl=600, show_spinner="Loading QB game history...")
def qb_games():
    return load_qb_games()


@st.cache_data(ttl=600, show_spinner="Building features...")
def all_features():
    return build_features(qb_games())


@st.cache_data(ttl=600)
def teams():
    return pd.read_sql(text("SELECT team_id, team_name FROM teams"), engine).set_index("team_id")["team_name"]


@st.cache_data(ttl=600, show_spinner="Scoring past games...")
def history_predictions(mtime):
    """Every model's pregame prediction for every QB game with enough history."""
    b = bundle()
    feats = all_features()
    feats = feats[has_history(feats)].copy()
    if feats.empty:
        return feats
    for name, pred in predict_all(b["models"], feats[b["features"]]).items():
        feats[name] = pred
    return feats


@st.cache_data(ttl=300, show_spinner="Pricing this week's lines...")
def upcoming_picks(hours, mtime, min_ev, kelly):
    """Same logic as scripts/predict_upcoming.py, kept as data instead of printed text."""
    b = bundle()
    prob_model = from_bundle(b, "xgboost")
    config = BetConfig(min_ev=min_ev, kelly_multiplier=kelly)
    lines = load_upcoming_lines(hours)
    if lines.empty:
        return lines, pd.DataFrame()
    qbs = upcoming_qbs(lines)
    feats = build_upcoming_features(
        qb_games(),
        qbs[["game_id", "player_id", "player_name", "team_id", "opponent_team_id",
             "game_date", "season", "week", "home_away"]],
    )
    cols = ["game_id", "player_id", "avg_pass_yds_last5", *MODEL_NAMES]
    if feats.empty:
        feats = pd.DataFrame(columns=cols)
    else:
        for name, pred in predict_all(b["models"], feats[b["features"]]).items():
            feats[name] = pred
    qbs = qbs.merge(feats[cols], on=["game_id", "player_id"], how="left")

    rows = []
    for q in qbs.itertuples():
        books = lines[(lines["game_id"] == q.game_id) & (lines["player_id"] == q.player_id)]
        line = books["line"].median()
        has_model = pd.notna(q.xgboost)
        bet = best_bet(books, q.xgboost, prob_model, config) if has_model else None
        rows.append({
            "game_id": q.game_id, "player_id": q.player_id,
            "kickoff": pd.Timestamp(q.kickoff), "when": format_kickoff(q.kickoff),
            "matchup": f"{q.away} at {q.home}",
            "team": q.home if q.is_home else q.away,
            "qb": q.player_name, "books": len(books), "line": line,
            "model": q.xgboost, "last5": q.avg_pass_yds_last5, **{k: getattr(q, k) for k in MODEL_NAMES},
            "edge": q.xgboost - line if has_model else np.nan,
            "lean": bet["side"] if bet else None,
            "best_book": bet["book"] if bet else None,
            "best_line": bet["line"] if bet else np.nan,
            "best_price": bet["price"] if bet else np.nan,
            "p_win": bet["p_win"] if bet else np.nan,
            "market_p_win": bet["market_p_win"] if bet else np.nan,
            "ev": bet["ev"] if bet else np.nan,
            "stake": bet["stake"] if bet else 0.0,
            "qualifies": bool(bet["qualifies"]) if bet else False,
        })
    table = pd.DataFrame(rows)
    table["confident"] = confident(table)   # the pre-registered rule, at the consensus line
    # Bets in the window are placed together, so their combined stake is capped like one day's slate
    table["stake"] = cap_exposure(table["stake"], config.max_daily_exposure)
    return lines, table


def walkforward_mtime():
    return WALKFORWARD.stat().st_mtime if WALKFORWARD.exists() else None


@st.cache_data(ttl=600, show_spinner="Scoring past games...")
def fair_predictions(mtime, wf_mtime):
    """Predictions for games the models never trained on: the walk-forward ones
    where cached, plus the current models' for games after their training cutoff.
    `source` says which."""
    hist = history_predictions(mtime)
    keys = ["game_id", "player_id"]
    after = hist[hist["game_date"] > pd.Timestamp(bundle()["train_end"])].assign(source="current models")
    if not wf_mtime:
        return after.reset_index(drop=True)
    wf = pd.read_csv(WALKFORWARD)[keys + list(MODEL_NAMES)]
    walked = hist.drop(columns=list(MODEL_NAMES)).merge(wf, on=keys).assign(source="walk-forward")
    after = after[~after.set_index(keys).index.isin(walked.set_index(keys).index)]
    return pd.concat([walked, after], ignore_index=True)


@st.cache_data(ttl=600, show_spinner="Loading closing lines...")
def closing_lines():
    """One closing line per QB-game, with prices (scripts/backtest.py)."""
    session = SessionLocal()
    try:
        return load_closing_prop_lines(session)
    finally:
        session.close()


def backtest_bets(mtime, wf_mtime):
    """Graded games (pregame line + result) the models never trained on, and how
    many graded games there are in all."""
    lines = closing_lines()
    if lines.empty:
        return pd.DataFrame(), 0
    total = len(history_predictions(mtime).merge(lines, on=["game_id", "player_id"]))
    bets = fair_predictions(mtime, wf_mtime).merge(lines, on=["game_id", "player_id"], how="inner")
    return bets.sort_values(["game_date", "game_id"]).reset_index(drop=True), total


def earlier_misses(bets, key):
    """Win chances for each date built only from this model's misses on games
    before that date, so no bet is sized with knowledge of its own result.
    (Until 50 earlier misses exist, a bell curve of the misses so far is used.)"""
    d = bets[["game_date", "passing_yards", key]].dropna().sort_values("game_date")
    dates = d["game_date"].to_numpy()
    misses = (d["passing_yards"] - d[key]).to_numpy(dtype=float)
    cache = {}

    def for_date(date):
        if date not in cache:
            cache[date] = ResidualModel(misses[dates < pd.Timestamp(date).to_datetime64()])
        return cache[date]
    return for_date


def confident(df):
    """The pre-registered confident-picks rule (scripts/weekly_report.py)."""
    sides = np.sign(df[CONFIDENT_RULE["models"]].to_numpy() - df[["line"]].to_numpy())
    return (((df["xgboost"] - df["line"]).abs() >= CONFIDENT_RULE["min_edge"])
            & (np.abs(sides.sum(axis=1)) == len(CONFIDENT_RULE["models"]))).to_numpy()


def american(price):
    return "" if pd.isna(price) else f"{price:+.0f}"


# --------------------------------------------------------------------------
# Small rendering helpers
# --------------------------------------------------------------------------
def yard_strip(line, model, last5, lean):
    vals = [v for v in (line, model, last5) if pd.notna(v)]
    lo = max(0, int((min(vals) - 35) // 25 * 25))
    hi = int((max(vals) + 35) // 25 * 25 + 25)

    def pos(v):
        return (v - lo) / (hi - lo) * 100

    parts = ['<div class="strip"><div class="field"></div>']
    for y in range(lo, hi + 1, 25):
        parts.append(f'<div class="tick" style="left:{pos(y):.2f}%"></div>')
        if y % 50 == 0 and lo < y < hi:
            parts.append(f'<div class="ticklab" style="left:{pos(y):.2f}%">{y}</div>')
    if pd.notna(model):
        color = OVER if lean == "OVER" else UNDER
        left, right = sorted((pos(line), pos(model)))
        parts.append(f'<div class="edge" style="left:{left:.2f}%;width:{right - left:.2f}%;background:{color}"></div>')
    if pd.notna(last5):
        parts.append(f'<div class="l5Mark" style="left:{pos(last5):.2f}%" title="Last 5 games: {last5:.0f} yds"></div>')
    parts.append(f'<div class="lineMark" style="left:{pos(line):.2f}%" title="Consensus line {line:g}"></div>')
    if pd.notna(model):
        parts.append(f'<div class="modelMark" style="left:{pos(model):.2f}%;background:{color}" '
                     f'title="Model {model:.0f} yds"></div>')
        # Put the two labels on opposite sides so they don't collide
        line_first = line <= model
        parts.append(f'<div class="marklab" style="left:{pos(line):.2f}%;color:{INK};'
                     f'transform:translateX({"-100%" if line_first else "0"})">{line:g}</div>')
        parts.append(f'<div class="marklab" style="left:{pos(model):.2f}%;color:{color};'
                     f'transform:translateX({"0" if line_first else "-100%"})">{model:.0f}</div>')
    else:
        parts.append(f'<div class="marklab" style="left:{pos(line):.2f}%">{line:g}</div>')
    parts.append("</div>")
    return "".join(parts)


def pick_row(p, bankroll, min_ev):
    e = html.escape
    if isinstance(p["lean"], str):  # missing values arrive as NaN, which is truthy
        color = OVER if p["lean"] == "OVER" else UNDER
        lean = (f'<div class="lean" style="color:{color}">{p["lean"].title()} {p["best_line"]:g}</div>'
                f'<div class="price">{e(p["best_book"])} {american(p["best_price"])}</div>')
        ev_color = OVER if p["ev"] > 0 else MUTED
        ev_text = f'{p["ev"]:+.2f}' if abs(p["ev"]) >= 0.005 else "0.00"
        if p["qualifies"]:
            sizing = f'<small class="stake">Bet {p["stake"] * bankroll:.2f} units</small>'
        else:
            sizing = f'<small>Pass: EV under {min_ev:+.2f}</small>'
        ev = (f'<div class="ev" style="color:{ev_color}">{ev_text}'
              f'<small>{p["p_win"]:.0%} model, {p["market_p_win"]:.0%} market</small>{sizing}</div>')
    else:
        lean = '<div class="price">Not enough history to predict</div>'
        ev = '<div class="ev"></div>'
    return (
        f'<div class="pick">'
        f'<div><div class="when">{e(p["when"])}</div><div class="matchup">{e(p["matchup"])}</div></div>'
        f'<div><div class="qb">{e(p["qb"])}</div><div class="team">{e(p["team"])}, {p["books"]} books</div></div>'
        f'{yard_strip(p["line"], p["model"], p["last5"], p["lean"])}'
        f'<div>{lean}</div>{ev}</div>'
    )


LEGEND = f"""
<div class="legend">
  <span><i class="sw" style="width:3px;height:16px;background:{INK}"></i>Sportsbook line (median across books)</span>
  <span><i class="sw" style="width:12px;height:12px;border-radius:50%;background:{OVER}"></i>XGBoost prediction</span>
  <span><i class="sw" style="width:9px;height:9px;border-radius:50%;border:2px solid {MUTED}"></i>Last 5 games average</span>
  <span>EV is expected profit per 1-unit bet at the best price; market is the books' no-vig chance</span>
</div>
"""

DISCLAIMER = ("Not betting advice. These models haven't been shown to beat real lines yet (see Backtest). "
              "Model win chances come from how far off its past predictions were; when the model is far "
              "more confident than the market, the model is usually the one that's wrong. Stakes are "
              "fractional Kelly, capped at 5% of bankroll per bet.")


def require_models():
    if bundle() is None:
        st.warning("No trained models yet. Train them on the Models & data page, "
                   "or run `python -m scripts.train` in the terminal.")
        st.stop()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_picks():
    require_models()
    with st.sidebar:
        st.subheader("Filters")
        hours = st.select_slider("Kickoff within", options=[24, 48, 72, 96, 120, 168], value=72,
                                 format_func=lambda h: f"{h} hours" if h < 168 else "7 days")
        min_edge = st.slider("Minimum edge (yards)", 0, 80, 0, step=5,
                             help="Only show QBs where the model and the line differ by at least this much")
        sort = st.radio("Sort by", ["Expected value", "Edge", "Kickoff"], horizontal=False)
        st.subheader("Bet sizing")
        bankroll = st.number_input("Bankroll (units)", min_value=1.0, value=100.0, step=10.0)
        min_ev = st.slider("Minimum EV to bet", 0.0, 0.20, BetConfig.min_ev, step=0.01, format="%.2f",
                           help="Expected profit per 1 unit staked. Bets below this are passed.")
        kelly = st.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5, 1.0], value=BetConfig.kelly_multiplier,
                                 format_func=lambda k: {0.1: "Tenth", 0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[k],
                                 help="Share of the full Kelly stake to bet. Quarter is the usual choice "
                                      "because full Kelly punishes overconfident probabilities hard.")

    lines, picks = upcoming_picks(hours, model_mtime(), min_ev, kelly)
    st.header("This week's picks")
    if picks.empty:
        st.info(f"No stored prop lines for games kicking off in the next {hours} hours. "
                "Lines are saved by the weekly Friday job, so check back after it runs "
                "or widen the kickoff window.")
        return

    priced = picks[picks["lean"].notna()]
    big = priced[priced["edge"].abs() >= 15]
    bets = picks[picks["qualifies"]]
    st.markdown(
        f'<p class="summary">{len(picks)} quarterbacks have lines for games in the next {hours} hours. '
        f'The model disagrees with the line by 15+ yards on {len(big)} of them. {len(bets)} clear the '
        f'{min_ev:+.2f} EV filter, for a total stake of {bets["stake"].sum() * bankroll:.2f} units '
        f'({bets["stake"].sum():.0%} of bankroll; all bets together are capped at '
        f'{BetConfig.max_daily_exposure:.0%}).</p>', unsafe_allow_html=True)

    shown = picks[picks["edge"].abs().fillna(0) >= min_edge] if min_edge else picks
    key = {"Expected value": ("ev", False), "Edge": ("edge", False), "Kickoff": ("kickoff", True)}[sort]
    if sort == "Edge":
        shown = shown.reindex(shown["edge"].abs().sort_values(ascending=False, na_position="last").index)
    else:
        shown = shown.sort_values(key[0], ascending=key[1], na_position="last")

    if shown.empty:
        st.info(f"No QBs with an edge of {min_edge}+ yards. Lower the minimum edge to see more.")
    else:
        st.markdown(LEGEND, unsafe_allow_html=True)
        st.markdown("".join(pick_row(p, bankroll, min_ev) for _, p in shown.iterrows()), unsafe_allow_html=True)

    st.subheader("Compare sportsbooks")
    options = shown[["game_id", "player_id", "qb", "matchup"]].drop_duplicates()
    if not options.empty:
        choice = st.selectbox("Quarterback", options.itertuples(index=False),
                              format_func=lambda o: f"{o.qb} ({o.matchup})")
        p = shown[(shown["game_id"] == choice.game_id) & (shown["player_id"] == choice.player_id)].iloc[0]
        books = lines[(lines["game_id"] == choice.game_id) & (lines["player_id"] == choice.player_id)]
        table = books[["sportsbook", "line", "over_odds", "under_odds"]].sort_values("line").copy()
        if pd.notna(p["model"]):
            prob_model = from_bundle(bundle(), "xgboost")
            for side in ("OVER", "UNDER"):
                price_col = "over_odds" if side == "OVER" else "under_odds"
                table[f"{side.title()} EV"] = [
                    evaluate_side(p["model"], r.line, getattr(r, price_col), side, prob_model)["ev"]
                    for r in table.itertuples()]
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "sportsbook": "Book",
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "over_odds": st.column_config.NumberColumn("Over price", format="%+d"),
                "under_odds": st.column_config.NumberColumn("Under price", format="%+d"),
                "Over EV": st.column_config.NumberColumn(format="%+.2f"),
                "Under EV": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

    with st.expander("All numbers as a table"):
        st.dataframe(
            shown[["when", "matchup", "qb", "books", "line", "model", "ensemble", "last5", "edge",
                   "lean", "best_book", "best_line", "best_price", "p_win", "market_p_win", "ev", "stake"]].assign(
                       stake=lambda d: d["stake"] * bankroll),
            hide_index=True, width="stretch",
            column_config={
                "when": "Kickoff (CT)", "matchup": "Game", "qb": "QB", "books": "Books",
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "model": st.column_config.NumberColumn("XGBoost", format="%.0f"),
                "ensemble": st.column_config.NumberColumn("Ensemble", format="%.0f"),
                "last5": st.column_config.NumberColumn("Last 5 avg", format="%.0f"),
                "edge": st.column_config.NumberColumn("Edge", format="%+.0f"),
                "lean": "Lean", "best_book": "Best book",
                "best_line": st.column_config.NumberColumn("Best line", format="%.1f"),
                "best_price": st.column_config.NumberColumn("Price", format="%+d"),
                "p_win": st.column_config.ProgressColumn("Model win %", format="percent", min_value=0, max_value=1),
                "market_p_win": st.column_config.NumberColumn("Market %", format="percent"),
                "ev": st.column_config.NumberColumn("EV / unit", format="%+.2f"),
                "stake": st.column_config.NumberColumn("Stake (units)", format="%.2f"),
            },
        )
    st.markdown(f'<p class="note">{DISCLAIMER}</p>', unsafe_allow_html=True)


def graded_confident(bets):
    """Confident picks among graded games: XGBoost's side, result and units at the closing price."""
    sure = bets[confident(bets)].copy()
    over = sure["xgboost"] > sure["line"]
    y, line = sure["passing_yards"], sure["line"]
    won = np.where(over, y > line, y < line)
    sure["side"] = np.where(over, "OVER", "UNDER")
    sure["edge"] = sure["xgboost"] - line
    sure["result"] = np.where(y == line, "push", np.where(won, "win", "loss"))
    price = np.where(over, sure["over_odds"], sure["under_odds"])
    sure["units"] = np.select([sure["result"] == "win", sure["result"] == "loss"],
                              [american_payout(price), -1.0], 0.0)
    return sure


def record_row(label, g):
    w, l = int((g["result"] == "win").sum()), int((g["result"] == "loss").sum())
    return {"": label, "Picks": len(g), "W-L": f"{w}-{l}", "Win rate": 100 * w / (w + l) if w + l else np.nan,
            "Units": g["units"].sum(), "ROI": 100 * g["units"].sum() / len(g) if len(g) else np.nan}


def page_confident():
    require_models()
    rule = CONFIDENT_RULE
    with st.sidebar:
        st.subheader("Filters")
        hours = st.select_slider("Kickoff within", options=[24, 48, 72, 96, 120, 168], value=72,
                                 format_func=lambda h: f"{h} hours" if h < 168 else "7 days", key="conf_hours")
        bankroll = st.number_input("Bankroll (units)", min_value=1.0, value=100.0, step=10.0, key="conf_bank")

    st.header("Most confident picks")
    st.markdown(
        f'<p class="summary">The games where the models agree most: XGBoost is at least {rule["min_edge"]:g} '
        f'yards from the line <b>and</b> all four models pick the same side. This rule was registered on '
        f'{pd.Timestamp(rule["registered"]):%b %d, %Y}, before any game it\'s judged on. In testing, '
        'lines moved toward these picks after Friday (closing-line value, a real edge), but profit '
        'isn\'t proven yet.</p>', unsafe_allow_html=True)

    lines, picks = upcoming_picks(hours, model_mtime(), BetConfig.min_ev, BetConfig.kelly_multiplier)
    st.subheader("This week")
    sure = picks[picks["confident"]] if not picks.empty else picks
    if picks.empty:
        st.info(f"No stored prop lines for games kicking off in the next {hours} hours. Lines are saved by "
                "the Friday 9am job, so check back after it runs or widen the kickoff window.")
    elif sure.empty:
        st.info(f"None of the {len(picks)} quarterbacks with lines in the next {hours} hours meet the rule.")
    else:
        sure = sure.reindex(sure["edge"].abs().sort_values(ascending=False).index)
        st.caption(f"{len(sure)} of {len(picks)} quarterbacks with lines in the next {hours} hours, "
                   "biggest gap between the model and the line first.")
        st.markdown(LEGEND, unsafe_allow_html=True)
        st.markdown("".join(pick_row(p, bankroll, BetConfig.min_ev) for _, p in sure.iterrows()),
                    unsafe_allow_html=True)
        st.dataframe(
            sure[["qb", "matchup", "line", *rule["models"], "edge"]],
            hide_index=True, width="stretch",
            column_config={
                "qb": "QB", "matchup": "Game", "line": st.column_config.NumberColumn("Line", format="%.1f"),
                **{m: st.column_config.NumberColumn(MODEL_NAMES[m], format="%.0f") for m in rule["models"]},
                "edge": st.column_config.NumberColumn("XGBoost edge", format="%+.0f"),
            })

    st.subheader("Track record")
    bets, _ = backtest_bets(model_mtime(), walkforward_mtime())
    if bets.empty:
        st.info("No graded games the models never trained on yet. Build the walk-forward predictions with "
                "`python -m scripts.vs_closing_lines` to see how the rule did on past seasons.")
    else:
        hist = graded_confident(bets)
        seasons = [record_row(str(season), g) for season, g in hist.groupby("season")]
        seasons.append(record_row("All", hist))
        st.dataframe(pd.DataFrame(seasons), hide_index=True, width="stretch", column_config={
            "Win rate": st.column_config.NumberColumn(format="%.1f%%", help="About 52.4% breaks even at -110"),
            "Units": st.column_config.NumberColumn(format="%+.2f"),
            "ROI": st.column_config.NumberColumn(format="%.1f%%"),
        })
        st.caption(f"Every confident pick among {len(bets):,} past games, out of sample (models retrained "
                   "week by week on earlier games only), flat 1-unit bets at each game's closing price. "
                   "Live picks are made at Friday's line, which the backtest showed is easier to beat.")

        curve = hist.sort_values(["game_date", "game_id"]).assign(
            Pick=lambda d: np.arange(1, len(d) + 1), Units=lambda d: d["units"].cumsum())
        line_chart = alt.Chart(curve).mark_line(color=OVER, strokeWidth=2.5).encode(
            x=alt.X("Pick:Q", title="Confident pick number"), y=alt.Y("Units:Q", title="Running profit (units)"),
            tooltip=[alt.Tooltip("game_date:T", title="Date"), alt.Tooltip("player_name:N", title="QB"),
                     alt.Tooltip("Units:Q", format="+.1f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
        st.altair_chart((zero + line_chart).properties(height=260), width="stretch")

        names = teams()
        order = st.radio("Show", ["Most recent", "Biggest edges"], horizontal=True)
        shown = (hist.sort_values(["game_date", "game_id"], ascending=False) if order == "Most recent"
                 else hist.reindex(hist["edge"].abs().sort_values(ascending=False).index)).head(25)
        st.dataframe(
            shown.assign(game=shown["team_id"].map(names).fillna("?") + " vs "
                         + shown["opponent_team_id"].map(names).fillna("?"))[
                ["game_date", "player_name", "game", "line", "xgboost", "edge", "side",
                 "passing_yards", "result", "units"]],
            hide_index=True, width="stretch",
            column_config={
                "game_date": st.column_config.DateColumn("Date", format="MMM D, YYYY"),
                "player_name": "QB", "game": "Game",
                "line": st.column_config.NumberColumn("Line", format="%.1f"),
                "xgboost": st.column_config.NumberColumn("XGBoost", format="%.0f"),
                "edge": st.column_config.NumberColumn("Edge", format="%+.0f"),
                "side": "Pick", "passing_yards": st.column_config.NumberColumn("Actual", format="%.0f"),
                "result": "Result", "units": st.column_config.NumberColumn("Units", format="%+.2f"),
            })

    graded = REPORTS / "graded_picks.csv"
    live = pd.read_csv(graded) if graded.exists() else pd.DataFrame()
    if "confident_pick" in live:
        live = live[live["confident_pick"].astype(str) == "True"]
    else:
        live = live.iloc[0:0]
    st.subheader("Live record")
    if live.empty:
        st.caption(f"No live confident picks graded yet. The weekly report starts counting games after "
                   f"{pd.Timestamp(rule['registered']):%b %d, %Y}; see reports/weekly_results.md.")
    else:
        live = live.rename(columns={"xgboost_result": "result", "xgboost_units": "units"})
        st.dataframe(pd.DataFrame([record_row("Since the rule was registered", live)]), hide_index=True,
                     width="stretch", column_config={
                         "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
                         "Units": st.column_config.NumberColumn(format="%+.2f"),
                         "ROI": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption("Graded each Friday by the weekly report at the Friday line (reports/weekly_results.md).")
    st.markdown(f'<p class="note">{DISCLAIMER}</p>', unsafe_allow_html=True)


def page_qb():
    require_models()
    b = bundle()
    games = qb_games()
    names = teams()
    starters = games[games["attempts"] >= 10].sort_values("game_date")
    roster = (starters.groupby("player_id")
              .agg(name=("player_name", "last"), team_id=("team_id", "last"),
                   last=("game_date", "max"), n=("game_id", "count"))
              .sort_values(["last", "n"], ascending=False).reset_index())
    roster["label"] = roster["name"] + ", " + roster["team_id"].map(names).fillna("?")

    st.header("Quarterback lookup")
    pick = st.selectbox("Quarterback", roster.itertuples(index=False), format_func=lambda r: r.label,
                        placeholder="Type a name or school")
    if pick is None:
        return
    qb_id, team_id = pick.player_id, pick.team_id

    fair = fair_predictions(model_mtime(), walkforward_mtime())
    lines = closing_lines()
    mine = games[(games["player_id"] == qb_id)].sort_values("game_date").tail(12).copy()
    mine = mine.merge(fair[fair["player_id"] == qb_id][["game_id", "xgboost"]], on="game_id", how="left")
    if not lines.empty:
        mine = mine.merge(lines[lines["player_id"] == qb_id][["game_id", "consensus"]], on="game_id", how="left")
    else:
        mine["consensus"] = np.nan
    mine["opponent"] = mine["opponent_team_id"].map(names)
    mine["label"] = (mine["game_date"].dt.strftime("%b %d '%y").str.replace(" 0", " ")
                     + " vs " + mine["opponent"].fillna("?"))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Recent games")
        if mine.empty:
            st.info("No games on record.")
        else:
            x = alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=160, labelOverlap=False))
            bars = alt.Chart(mine).mark_bar(color="#9FB3C8").encode(
                x=x, y=alt.Y("passing_yards:Q", title="Passing yards"),
                tooltip=[alt.Tooltip("label:N", title="Game"), alt.Tooltip("passing_yards:Q", title="Actual"),
                         alt.Tooltip("consensus:Q", title="Sportsbook line", format=".1f"),
                         alt.Tooltip("xgboost:Q", title="Model (pregame)", format=".0f")])
            ticks = alt.Chart(mine.dropna(subset=["consensus"])).mark_tick(
                color=GOLD, thickness=3).encode(x=x, y="consensus:Q")
            dots = alt.Chart(mine.dropna(subset=["xgboost"])).mark_point(
                filled=True, size=90, color=INK, shape="diamond").encode(x=x, y="xgboost:Q")
            st.altair_chart((bars + ticks + dots).properties(height=400), width="stretch")
            st.caption("Bars are actual passing yards, gold ticks the sportsbook line at close, and diamonds "
                       "what XGBoost predicted before the game from models that had never seen it "
                       "(retrained week by week on earlier games only). Games without a diamond weren't "
                       "scored that way.")
            scored = mine.dropna(subset=["xgboost"])
            if len(scored):
                miss = (scored["xgboost"] - scored["passing_yards"]).abs().mean()
                lined = scored.dropna(subset=["consensus"])
                vs_line = (f" (the line missed by {(lined['consensus'] - lined['passing_yards']).abs().mean():.0f} "
                           f"on the same games)" if len(lined) == len(scored) else "")
                st.caption(f"Model's average miss on these {len(scored)} games: {miss:.0f} yards{vs_line}.")

    with right:
        st.subheader("Predict a game")
        season = int(mine["season"].max()) if not mine.empty else None
        sched = pd.read_sql(text("""
            SELECT game_id, game_date, season, week, home_team_id, away_team_id
            FROM games WHERE :t IN (home_team_id, away_team_id) AND season >= :s
            ORDER BY game_date
        """), engine, params={"t": int(team_id), "s": season or 0})
        if sched.empty:
            st.info("No scheduled games found for this team.")
            return
        sched["game_date"] = pd.to_datetime(sched["game_date"])
        sched["is_home"] = sched["home_team_id"] == team_id
        sched["opp"] = np.where(sched["is_home"], sched["away_team_id"], sched["home_team_id"])
        today = pd.Timestamp.now().normalize()
        sched["label"] = (sched["game_date"].dt.strftime("%b %d").str.replace(" 0", " ")
                          + np.where(sched["is_home"], " vs ", " at ") + sched["opp"].map(names).fillna("?")
                          + np.where(sched["game_date"] >= today, "  (upcoming)", ""))
        upcoming = sched.index[sched["game_date"] >= today]
        default = int(upcoming[0]) if len(upcoming) else len(sched) - 1
        g = st.selectbox("Game", sched.itertuples(index=False), index=default, format_func=lambda r: r.label)

        row = pd.DataFrame([{
            "game_id": g.game_id, "player_id": qb_id, "player_name": pick.name, "team_id": team_id,
            "opponent_team_id": g.opp, "game_date": g.game_date, "season": g.season, "week": g.week,
            "home_away": "home" if g.is_home else "away",
        }])
        feats = build_upcoming_features(games, row)
        if feats.empty:
            st.info(f"{pick.name} needs at least 3 earlier games with 10+ attempts before the models can predict.")
            return
        preds = predict_all(b["models"], feats[b["features"]])
        f = feats.iloc[0]
        line = pd.read_sql(text("""
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY line)::float AS line FROM (
                SELECT DISTINCT ON (sportsbook) line FROM qb_prop_lines
                WHERE game_id = :g AND player_id = :p ORDER BY sportsbook, fetched_at DESC) t
        """), engine, params={"g": int(g.game_id), "p": int(qb_id)})["line"].iloc[0]

        chart_df = pd.DataFrame({"model": [MODEL_NAMES[k] for k in preds],
                                 "yards": [float(v[0]) for v in preds.values()]})
        y = alt.Y("model:N", sort=list(chart_df["model"]), title=None)
        bars = alt.Chart(chart_df).mark_bar(size=18).encode(
            y=y, x=alt.X("yards:Q", title="Predicted passing yards"),
            color=alt.condition(alt.datum.model == "XGBoost", alt.value(INK), alt.value("#9FB3C8")),
            tooltip=["model", alt.Tooltip("yards:Q", format=".0f")])
        labels = bars.mark_text(align="left", dx=4, fontSize=13).encode(text=alt.Text("yards:Q", format=".0f"),
                                                                       color=alt.value(INK))
        layers = [bars, labels]
        refs = []
        if pd.notna(line):
            refs.append(("Line", line, GOLD))
        if pd.notna(f["passing_yards"]):
            refs.append(("Actual", f["passing_yards"], OVER))
        for name, val, color in refs:
            layers.append(alt.Chart(pd.DataFrame({"v": [val], "t": [f"{name} {val:g}"]}))
                          .mark_rule(color=color, strokeWidth=2.5, strokeDash=[5, 3]).encode(x="v:Q",
                          tooltip=[alt.Tooltip("t:N", title=name)]))
        st.altair_chart(alt.layer(*layers).properties(height=230), width="stretch")

        facts = [f"Last 5 games: {f['avg_pass_yds_last5']:.0f} yds"]
        if pd.notna(f["opp_pass_yds_allowed_avg"]):
            facts.append(f"opponent allows {f['opp_pass_yds_allowed_avg']:.0f} pass yds a game")
        if pd.notna(line):
            facts.append(f"sportsbook line {line:g} (gold)")
        if pd.notna(f["passing_yards"]):
            facts.append(f"actual result {f['passing_yards']:.0f} (green)")
        st.caption("; ".join(facts) + ".")


def page_backtest():
    require_models()
    b = bundle()
    st.header("Backtest")
    st.markdown('<p class="summary">How the models would have done betting real pregame lines '
                'on games they never trained on.</p>', unsafe_allow_html=True)
    wf_mtime = walkforward_mtime()
    bets, total = backtest_bets(model_mtime(), wf_mtime)
    if bets.empty:
        if total:
            st.info(f"{total} QB games have a pregame line and a result, but the current models trained on "
                    f"all of them (through {pd.Timestamp(b['train_end']).date()}). To score them fairly, build "
                    "the walk-forward predictions once with `python -m scripts.vs_closing_lines` "
                    "(retrains the models week by week; takes a few minutes), then refresh.")
        else:
            st.info("No QB games have both a stored pregame line and a final stat line yet. Lines are saved "
                    "by the weekly Friday job; results appear once that week's CFBD stats are ingested.")
        return

    c1, c2 = st.columns([2, 3])
    seasons = sorted(bets["season"].unique())
    season = c1.selectbox("Season", ["All"] + seasons, format_func=str)
    edge = c2.slider("Only bet when the model and line differ by at least (yards)", 0, 60, 0, step=5,
                     help="For exploring only. Picking the threshold that looks best here is overfitting: "
                          "it will look worse on future games. The fair test is the pre-registered "
                          "confident-picks rule (15 yards, all four models agree), fixed before these games "
                          "were scored.")
    if season != "All":
        bets = bets[bets["season"] == season].reset_index(drop=True)
    y = bets["passing_yards"].to_numpy(float)
    line = bets["line"].to_numpy(float)
    walked = int((bets["source"] == "walk-forward").sum())
    st.caption(f"{len(bets):,} games from {bets['game_date'].min():%b %d, %Y} to {bets['game_date'].max():%b %d, %Y}, "
               f"at each game's closing line (the book nearest the consensus of {bets['books'].mean():.1f} books "
               f"on average). {walked:,} are scored walk-forward: every week predicted by models retrained only "
               "on the games before it."
               + (f" The other {len(bets) - walked:,} are newer than that and scored by the current models."
                  if walked < len(bets) else "")
               + (" Small sample: win rate and ROI will swing a lot until more weeks are in." if len(bets) < 200 else ""))

    def result_row(name, pred, games):
        r = simulate_betting_strategy(pred, games, edge)
        return {"Model": name, "Avg miss (yds)": np.mean(np.abs(games["passing_yards"].to_numpy(float) - pred)),
                "Bets": r["total_bets"], "Win rate": 100 * r["win_rate"],
                "Profit (units)": r["total_profit"], "ROI": 100 * r["roi"]}

    rows = [result_row(name, bets[key].to_numpy(float), bets) for key, name in MODEL_NAMES.items()]
    sure = confident(bets)
    if sure.any():
        rows.append(result_row("Confident picks", bets.loc[sure, "xgboost"].to_numpy(float), bets[sure]))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Avg miss (yds)": st.column_config.NumberColumn(format="%.1f",
                                                        help="Mean absolute error vs. actual passing yards"),
        "Win rate": st.column_config.NumberColumn(format="%.1f%%", help="About 52.4% breaks even at -110"),
        "Profit (units)": st.column_config.NumberColumn(format="%+.2f"),
        "ROI": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption(f"The sportsbook line itself misses by {np.mean(np.abs(y - line)):.1f} yards on average. "
               "That's the bar to beat: a model that doesn't miss by less than the line is unlikely "
               "to find real value. Confident picks are the pre-registered rule: XGBoost at least "
               f"{CONFIDENT_RULE['min_edge']:g} yards from the line and all four models on the same side. "
               "About 52.4% wins break even at -110.")

    # Running profit, per model, in kickoff order (same rules as simulate_betting_strategy)
    order = bets.sort_values(["game_date", "game_id"]).index
    curves = []
    for key, name in [*MODEL_NAMES.items(), ("xgboost", "Confident picks")]:
        e = bets[key].to_numpy(float) - line
        over, under = e > edge, e < -edge
        if name == "Confident picks":
            over, under = over & sure, under & sure
        win = (over & (y > line)) | (under & (y < line))
        lose = (over & (y < line)) | (under & (y > line))
        profit = np.where(win & over, american_payout(bets["over_odds"]), 0.0) \
            + np.where(win & under, american_payout(bets["under_odds"]), 0.0) - lose
        s = pd.Series(profit, index=bets.index).loc[order]
        placed = (over | under)[order]
        curves.append(pd.DataFrame({"Bet number": np.arange(1, placed.sum() + 1),
                                    "Profit": s[placed].cumsum().to_numpy(), "Model": name}))
    curves = pd.concat(curves)
    st.subheader("Running profit")
    chart = alt.Chart(curves).mark_line(strokeWidth=2).encode(
        x="Bet number:Q", y=alt.Y("Profit:Q", title="Units"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=[*MODEL_NAMES.values(), "Confident picks"],
                        range=[INK, GOLD, "#6C8EAD", "#A3A9B3", "#8C6D9E", OVER]), legend=LEGEND_TOP),
        strokeWidth=alt.condition((alt.datum.Model == "XGBoost") | (alt.datum.Model == "Confident picks"),
                                  alt.value(3), alt.value(1.5)))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
    st.altair_chart((zero + chart).properties(height=320), width="stretch")
    st.caption("Flat 1-unit bets at each game's closing price, in kickoff order; pushes return the stake. "
               "Confident picks bet far fewer games, so their line is shorter.")

    st.subheader("Kelly bankroll")
    st.markdown('<p class="note">Bets only when expected value clears the minimum, sized as a fraction '
                'of full Kelly from each model\'s win chance, replayed in date order. Each day\'s win chances '
                'come only from that model\'s misses on earlier games, so no bet is sized using its own '
                'result. Games on the same day are sized from that morning\'s bankroll, and one day\'s bets '
                'together never exceed 30% of it. The settings below are for exploring: picking the ones '
                'that look best here is overfitting.</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    min_ev = c1.slider("Minimum EV", 0.0, 0.20, BetConfig.min_ev, step=0.01, format="%.2f", key="bt_ev")
    kelly = c2.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5, 1.0], value=BetConfig.kelly_multiplier,
                             format_func=lambda k: {0.1: "Tenth", 0.25: "Quarter", 0.5: "Half", 1.0: "Full"}[k],
                             key="bt_kelly")
    start = c3.number_input("Starting bankroll", min_value=1.0, value=100.0, step=10.0, key="bt_bank")
    config = BetConfig(min_ev=min_ev, min_edge_yards=edge, kelly_multiplier=kelly)
    summary, paths = [], []
    for key, name in MODEL_NAMES.items():
        log, k = simulate_bankroll(bets, key, earlier_misses(bets, key), config, start=start)
        decided = k["wins"] + k["losses"]
        summary.append({"Model": name, "Bets": k["bets"], "Win rate": 100 * k["win_rate"],
                        "p": float(binom.sf(k["wins"] - 1, decided, 0.524)) if decided else np.nan,
                        "Staked": k["staked"], "Profit": k["profit"], "ROI on stakes": 100 * k["roi"],
                        "Final bankroll": k["final"], "Worst drawdown": 100 * k["max_drawdown"]})
        if len(log):
            paths.append(pd.DataFrame({"Bet number": np.arange(0, len(log) + 1),
                                       "Bankroll": np.r_[start, log["bankroll"].to_numpy()], "Model": name}))
    summary = pd.DataFrame(summary)
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
        "p": st.column_config.NumberColumn("p-value", format="%.3f",
                                           help="Chance of a win rate this high if the true rate were break-even "
                                                "(52.4%). Below 0.05 would be evidence of a real edge."),
        "Staked": st.column_config.NumberColumn(format="%.2f"),
        "Profit": st.column_config.NumberColumn(format="%+.2f"),
        "ROI on stakes": st.column_config.NumberColumn(format="%.1f%%"),
        "Final bankroll": st.column_config.NumberColumn(format="%.2f"),
        "Worst drawdown": st.column_config.NumberColumn(format="%.1f%%",
                                                        help="Largest fall from a peak, measured at the end of each day"),
    })
    if paths:
        paths = pd.concat(paths)
        chart = alt.Chart(paths).mark_line(strokeWidth=2).encode(
            x="Bet number:Q", y=alt.Y("Bankroll:Q", title="Bankroll (units)", scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=alt.Scale(domain=list(MODEL_NAMES.values()),
                            range=[INK, GOLD, "#6C8EAD", "#A3A9B3", "#8C6D9E"]), legend=LEGEND_TOP),
            strokeWidth=alt.condition(alt.datum.Model == "XGBoost", alt.value(3), alt.value(1.5)))
        base = alt.Chart(pd.DataFrame({"y": [start]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
        st.altair_chart((base + chart).properties(height=320), width="stretch")
    else:
        st.info("No bets cleared the minimum EV. Lower it to see a bankroll path.")
    real = summary[summary["p"] < 0.05]
    verdict = ("No model's Kelly result is statistically different from break-even: the bankroll swings are "
               "mostly luck, magnified by compounding." if real.empty else
               f"Better than break-even at p < 0.05: {', '.join(real['Model'])}. Several models and settings "
               "were compared, so treat even this with caution.")
    st.markdown(f'<p class="note"><b>{verdict}</b> ROI on stakes is the honest number to look at; a big final '
                'bankroll can come from a lucky streak early on. The pre-registered confident-picks rule '
                '(see that tab) is the fair test of an edge.</p>', unsafe_allow_html=True)


def page_models():
    st.header("Models & data")
    b = bundle()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Trained models")
        if b is None:
            st.info("No trained models yet.")
        else:
            trained = pd.Timestamp(b.get("trained_at")) if b.get("trained_at") else None
            st.markdown(
                f"Trained {trained:%b %d, %Y at %I:%M %p} on games through "
                f"**{pd.Timestamp(b['train_end']):%b %d, %Y}**, using {len(b['features'])} pregame features."
                if trained is not None else f"Trained on games through {pd.Timestamp(b['train_end']).date()}.")
            xgb = b["models"].get("xgboost")
            if xgb is not None and hasattr(xgb, "feature_importances_"):
                imp = pd.DataFrame({"feature": b["features"], "importance": xgb.feature_importances_})
                st.altair_chart(alt.Chart(imp).mark_bar(color=INK, size=13).encode(
                    y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
                    x=alt.X("importance:Q", title="XGBoost importance"),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(
                    height=24 * len(imp)),
                    width="stretch")
        st.markdown('<p class="note">Retraining takes about 10 seconds and replaces models/qb_models.pkl. '
                    'It only reads the database.</p>', unsafe_allow_html=True)
        if st.button("Retrain models", type="primary"):
            with st.spinner("Training all four models and the ensemble..."):
                run = subprocess.run([sys.executable, "-m", "scripts.train"], cwd=ROOT,
                                     capture_output=True, text=True)
            if run.returncode == 0:
                st.cache_data.clear()
                st.success("Models retrained.")
                st.code(run.stdout, language=None)
            else:
                st.error("Training failed. Output:")
                st.code(run.stdout + run.stderr, language=None)

    with right:
        st.subheader("Database")
        counts = []
        with engine.connect() as c:
            for t in ["games", "teams", "qb_game_stats", "qb_prop_lines", "live_picks",
                      "qb_prop_closing_lines", "game_lines", "odds", "ingestion_log"]:
                counts.append({"Table": t, "Rows": c.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()})
            snap = c.execute(text("SELECT MAX(fetched_at) FROM qb_prop_lines")).scalar()
            latest_game = c.execute(text(
                "SELECT MAX(game_date) FROM qb_game_stats")).scalar()
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch",
                     column_config={"Rows": st.column_config.NumberColumn(format="%d")})
        st.markdown(f"Latest prop-line snapshot: **{snap:%b %d, %Y %H:%M} UTC**" if snap else
                    "No prop-line snapshots yet.")
        st.markdown(f"Latest QB stats: **{latest_game:%b %d, %Y}**" if latest_game else "No QB stats yet.")
        log = pd.read_sql(text("""SELECT run_at, source, record_count, success, error_message
                                  FROM ingestion_log ORDER BY run_at DESC LIMIT 8"""), engine)
        if not log.empty:
            st.caption("Recent ingestion runs")
            st.dataframe(log, hide_index=True, width="stretch", column_config={
                "run_at": st.column_config.DatetimeColumn("Ran at", format="MMM D, h:mm a"),
                "source": "Source", "record_count": "Records", "success": "OK", "error_message": "Error"})


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand">Just Bet It</div>'
                '<div class="brand-sub">QB passing yards vs. the books</div>', unsafe_allow_html=True)
    st.write("")

nav = st.navigation([
    st.Page(page_picks, title="This week", icon=":material/sports_football:", url_path="picks", default=True),
    st.Page(page_confident, title="Confident picks", icon=":material/verified:", url_path="confident"),
    st.Page(page_qb, title="QB lookup", icon=":material/person_search:", url_path="qb"),
    st.Page(page_backtest, title="Backtest", icon=":material/query_stats:", url_path="backtest"),
    st.Page(page_models, title="Models & data", icon=":material/database:", url_path="models"),
], position="top")

with st.sidebar:
    if st.button("Refresh data", help="Reload from the database (otherwise refreshed every few minutes)"):
        st.cache_data.clear()
        st.rerun()

try:
    nav.run()
except OperationalError as err:
    st.error("Can't reach the database. Check that DATABASE_URL in your .env file is the Neon connection "
             "string (ending in ?sslmode=require) and that you're online.")
    with st.expander("Error details"):
        st.code(str(err.orig) if getattr(err, "orig", None) else str(err), language=None)
