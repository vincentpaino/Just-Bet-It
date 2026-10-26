"""
Top-level ingestion entry point.
Mirrors the ingest_data() pseudocode from Doc II Section 3.3 / Doc III
Section 4.3 — pulls odds, pulls historical stats, logs the run.

Usage:
    python -m src.ingestion.run_ingestion --year 2026          # CFBD games, game lines, QB stats
    python -m src.ingestion.run_ingestion --props               # ...plus QB prop lines
    python -m src.ingestion.run_ingestion --odds --skip-stats --skip-qb   # game odds only

The Odds API steps (--odds, --props) are opt-in because the free tier has
only 500 credits a month; see scripts/snapshot_odds.sh for the weekly job.

Games are matched by their CFBD game ID. Older rows that predate the ID
column are found by matchup + date, then stamped with the ID and given the
corrected game date (this is what migrates existing data on a re-run).

Run order matters: team/game stats must run before QB stats, because QB rows
attach to games by CFBD game ID.
"""

import argparse
import re
from datetime import datetime, timedelta, timezone
from functools import partial

from dotenv import load_dotenv
from sqlalchemy import and_, or_

load_dotenv()

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import (
    GameLine,
    IngestionLog,
    Team,
    Odds,
    QBGameStat,
    QBPropLine,
    Game,
)

from src.db.session import SessionLocal, init_db

from .dates import as_utc_datetime, to_game_date, utc_date

from .odds_api import (
    fetch_event_props,
    fetch_events,
    fetch_odds,
    swap_home_away,
    validate_and_format as validate_odds,
    validate_and_format_props,
)

from .team_names import (
    build_odds_name_map,
    normalize,
    resolve_odds_team_name,
)

from .cfbd_ingest import (
    fetch_game_lines,
    fetch_teams,
    fetch_team_season_stats,
    fetch_qb_season_stats,
    validate_and_format as validate_stats,
    validate_and_format_qb_stats,
    CFBDUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Naive UTC timestamp (what the DateTime columns store)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get_or_create_team(session, team_name: str) -> Team:
    """Find a team by name or create it if it doesn't exist."""

    team = (
        session.query(Team)
        .filter_by(team_name=team_name)
        .first()
    )

    if team is None:
        team = Team(team_name=team_name)
        session.add(team)
        session.flush()

    return team


def _same_matchup(team_a_id: int, team_b_id: int):
    """SQL condition: these two teams play each other (either home/away)."""
    return or_(
        and_(
            Game.home_team_id == team_a_id,
            Game.away_team_id == team_b_id,
        ),
        and_(
            Game.home_team_id == team_b_id,
            Game.away_team_id == team_a_id,
        ),
    )


def _find_game_by_matchup(
    session,
    team_a_id: int,
    team_b_id: int,
    dates,
    unstamped_only: bool = False,
):
    """
    Find a game between two teams on any of the given dates.
    With unstamped_only=True, only games that don't have a CFBD ID yet
    are considered (used to migrate old rows).
    """

    dates = [d for d in set(dates) if d is not None]

    if not dates:
        return None

    query = (
        session.query(Game)
        .filter(Game.game_date.in_(dates))
        .filter(_same_matchup(team_a_id, team_b_id))
    )

    if unstamped_only:
        query = query.filter(Game.cfbd_game_id.is_(None))

    return query.first()


def _apply_result(game, cfbd_home_team_id, home_score, away_score) -> bool:
    """
    Write scores onto a game row, in the orientation that row uses.
    (A game created from odds data can have home/away reversed relative to
    CFBD, so scores are swapped to match.) Returns True if swapped.
    Only non-None scores are written, so a missing score never erases one.
    """

    swapped = game.home_team_id != cfbd_home_team_id

    if swapped:
        home_score, away_score = away_score, home_score

    if home_score is not None:
        game.home_score = home_score

    if away_score is not None:
        game.away_score = away_score

    if home_score is not None and away_score is not None:
        game.home_win = home_score > away_score

    return swapped


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def _load_odds_name_map() -> dict[str, str]:
    """
    Odds API name -> CFBD school lookup. If CFBD can't be reached, fall back
    to the manual overrides so the odds snapshot (which can't be re-fetched
    later) is still saved; unresolved names can be merged afterwards with
    scripts/merge_odds_teams.py.
    """

    try:
        return build_odds_name_map(fetch_teams())
    except Exception as exc:
        print(f"[odds] WARNING: could not load CFBD teams ({exc}); "
              f"Odds API names may not match CFBD teams this run")
        return build_odds_name_map([])  # manual overrides only


class _OddsTeamResolver:
    """Maps Odds API team names to Team rows, remembering names it couldn't map."""

    def __init__(self, session):
        self.session = session
        self.name_map = _load_odds_name_map()
        self.unresolved = set()

    def __call__(self, odds_name: str) -> Team:
        school = resolve_odds_team_name(odds_name, self.name_map)
        if school is None:
            self.unresolved.add(odds_name)
        return _get_or_create_team(self.session, school or odds_name)

    def warn_unresolved(self, label: str):
        if self.unresolved:
            print(f"[{label}] WARNING: no CFBD match for {sorted(self.unresolved)}; "
                  f"stored under the Odds API name. Add them to "
                  f"ODDS_NAME_OVERRIDES in team_names.py.")


def _find_or_create_odds_game(session, home_team: Team, away_team: Team, kickoff):
    """
    The game an Odds API event refers to. If CFBD ingestion hasn't created it
    yet, create a placeholder; the next CFBD run adopts it (matchup + date).
    Returns None if the kickoff time can't be parsed.
    """

    local_date = to_game_date(kickoff)

    if local_date is None:
        return None

    game = _find_game_by_matchup(
        session,
        home_team.team_id,
        away_team.team_id,
        [local_date, utc_date(kickoff)],
    )

    if game is None:
        game = Game(
            home_team_id=home_team.team_id,
            away_team_id=away_team.team_id,
            game_date=local_date,
        )
        session.add(game)
        session.flush()

    return game


def ingest_odds(session, year: int) -> int:
    """
    Fetch and store sportsbook odds.

    Odds API team names are mapped to CFBD school names so odds attach to the
    games CFBD ingestion created. Odds are stored from the game row's home
    side, so they are flipped when the Odds API lists the teams reversed.
    Every run adds a new snapshot (fetched_at), so line movement is kept.
    """

    raw_data = fetch_odds()

    records = validate_odds(raw_data)

    team_for = _OddsTeamResolver(session)

    written = flipped = 0

    for record in records:

        home_team = team_for(record["home_team"])
        away_team = team_for(record["away_team"])

        game = _find_or_create_odds_game(
            session, home_team, away_team, record.get("game_date")
        )

        if game is None:
            continue

        if game.home_team_id != home_team.team_id:
            record = swap_home_away(record)
            flipped += 1

        session.add(
            Odds(
                game_id=game.game_id,
                sportsbook=record.get("sportsbook"),
                home_moneyline=record.get("home_moneyline"),
                away_moneyline=record.get("away_moneyline"),
                spread=record.get("spread"),
                over_under=record.get("over_under"),
                implied_prob_home=record.get("implied_prob_home"),
                implied_prob_away=record.get("implied_prob_away"),
                fetched_at=record.get("fetched_at"),
            )
        )

        written += 1

    session.commit()

    if flipped:
        print(f"[odds] {flipped} rows flipped to match the stored game's home/away")

    team_for.warn_unresolved("odds")

    return written


# ---------------------------------------------------------------------------
# QB passing-yards props
# ---------------------------------------------------------------------------

# Props cost 1 Odds API credit per game whose props are posted (empty
# responses and the events list are free). These limits keep a run from
# wasting credits:
PROPS_WINDOW_HOURS = 24     # only games kicking off within this many hours
PROPS_REFETCH_HOURS = 20    # skip games already snapshotted this recently
PROPS_CREDIT_RESERVE = 50   # never let a run take remaining credits below this
PROPS_MAX_GAMES = 100       # hard cap on paid requests per run

_NAME_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)


def _player_key(name: str) -> str:
    """Compare player names ignoring case, punctuation and Jr./III suffixes."""
    return normalize(_NAME_SUFFIXES.sub("", name or ""))


def _qb_ids_by_name(session, team_ids) -> dict[str, int]:
    """{player name key: CFBD player_id} for QBs who have played for these teams."""

    rows = (
        session.query(QBGameStat.player_name, QBGameStat.player_id)
        .filter(QBGameStat.team_id.in_(team_ids))
        .order_by(QBGameStat.game_date.desc())
    )

    ids = {}
    for name, player_id in rows:
        ids.setdefault(_player_key(name), player_id)  # most recent wins
    return ids


def ingest_qb_props(
    session,
    year: int,
    window_hours: int = PROPS_WINDOW_HOURS,
    credit_reserve: int = PROPS_CREDIT_RESERVE,
    max_games: int = PROPS_MAX_GAMES,
) -> int:
    """
    Snapshot QB passing-yards prop lines for games kicking off in the next
    `window_hours`. Lines are matched to CFBD player IDs by name among the
    two teams' QBs; unmatched players are stored with player_id NULL.

    Credit safety: games snapshotted in the last PROPS_REFETCH_HOURS are
    skipped (so re-running costs nothing), requests stop before remaining
    credits would drop below `credit_reserve`, and at most `max_games` paid
    requests are made. Each game is committed as it's fetched, so credits
    already spent aren't lost if a later request fails.
    """

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=window_hours)
    recent = now.replace(tzinfo=None) - timedelta(hours=PROPS_REFETCH_HOURS)

    all_events, remaining = fetch_events()
    start_remaining = remaining

    events = sorted(
        (e for e in all_events if now < as_utc_datetime(e["commence_time"]) <= cutoff),
        key=lambda e: e["commence_time"],
    )

    print(f"[props] {len(events)} games kick off in the next {window_hours}h; "
          f"{remaining if remaining is not None else 'unknown'} credits remaining")

    team_for = _OddsTeamResolver(session)

    written = unmatched_players = requested = 0
    already_captured = no_props = 0
    skipped_for_budget = []

    for event in events:

        home_team = team_for(event["home_team"])
        away_team = team_for(event["away_team"])

        game = _find_or_create_odds_game(
            session, home_team, away_team, event["commence_time"]
        )

        if game is None:
            continue

        captured = (
            session.query(QBPropLine.id)
            .filter(QBPropLine.game_id == game.game_id)
            .filter(QBPropLine.fetched_at >= recent)
            .first()
        )

        if captured:
            already_captured += 1
            continue

        # Worst case this request costs 1 credit.
        if requested >= max_games or (
            remaining is not None and remaining - 1 < credit_reserve
        ):
            skipped_for_budget.append(f"{event['away_team']} @ {event['home_team']}")
            continue

        data, after = fetch_event_props(event["id"])
        requested += 1
        remaining = after if after is not None else remaining

        records = validate_and_format_props(data)

        if not records:
            no_props += 1
            session.commit()  # keep any placeholder game created above
            continue

        qb_ids = _qb_ids_by_name(session, [home_team.team_id, away_team.team_id])

        for record in records:
            player_id = qb_ids.get(_player_key(record["player_name"]))
            unmatched_players += player_id is None

            session.add(
                QBPropLine(
                    game_id=game.game_id,
                    player_id=player_id,
                    player_name=record["player_name"],
                    sportsbook=record["sportsbook"],
                    line=record["line"],
                    over_odds=record["over_odds"],
                    under_odds=record["under_odds"],
                    fetched_at=record["fetched_at"],
                )
            )
            written += 1

        session.commit()

    session.commit()

    used = (
        start_remaining - remaining
        if start_remaining is not None and remaining is not None else "unknown"
    )
    print(f"[props] requested {requested} games ({no_props} had no props yet, free); "
          f"skipped {already_captured} already captured in the last "
          f"{PROPS_REFETCH_HOURS}h; credits used: {used}, remaining: {remaining}")

    if skipped_for_budget:
        print(f"[props] WARNING: skipped {len(skipped_for_budget)} games to stay above "
              f"the {credit_reserve}-credit reserve / {max_games}-game cap: "
              f"{skipped_for_budget}")

    if unmatched_players:
        print(f"[props] {unmatched_players} lines had no matching CFBD QB "
              f"(stored with player_id NULL)")

    team_for.warn_unresolved("props")

    return written


# ---------------------------------------------------------------------------
# Games (named "team stats" for historical reasons)
# ---------------------------------------------------------------------------

def ingest_team_stats(session, year: int) -> int:
    """
    Fetch CFBD games for a season and create / update them.

    Matching order for each CFBD game:
        1. by cfbd_game_id
        2. by matchup + date, among games that have no ID yet (old rows);
           these get stamped with the ID and the corrected game date
        3. otherwise create a new game
    """

    raw_df = fetch_team_season_stats(year)

    records = validate_stats(raw_df)

    games_by_cfbd_id = {
        game.cfbd_game_id: game
        for game in session.query(Game).filter(Game.cfbd_game_id.is_not(None))
    }

    created = matched = migrated = swapped = 0

    for record in records:

        home_team = _get_or_create_team(session, record["home_team"])
        away_team = _get_or_create_team(session, record["away_team"])

        cfbd_id = record["cfbd_game_id"]

        game = games_by_cfbd_id.get(cfbd_id)

        if game is not None:
            matched += 1

        else:
            game = _find_game_by_matchup(
                session,
                home_team.team_id,
                away_team.team_id,
                [record["game_date"], utc_date(record["start_time"])],
                unstamped_only=True,
            )

            if game is not None:
                migrated += 1
            else:
                game = Game(
                    home_team_id=home_team.team_id,
                    away_team_id=away_team.team_id,
                    game_date=record["game_date"],
                )
                session.add(game)
                created += 1

        game.cfbd_game_id = cfbd_id
        games_by_cfbd_id[cfbd_id] = game

        game.game_date = record["game_date"]
        game.start_time = record["start_time"]
        game.start_time_tbd = record["start_time_tbd"]
        game.season = record["season"]
        game.week = record["week"]
        game.season_type = record["season_type"]
        game.neutral_site = record["neutral_site"]

        if _apply_result(
            game,
            home_team.team_id,
            record["home_score"],
            record["away_score"],
        ):
            swapped += 1

        session.flush()

    session.commit()

    print(
        f"[cfbd_stats] {year}: {created} created, {matched} matched by ID, "
        f"{migrated} migrated from old rows, "
        f"{swapped} with home/away reversed vs CFBD"
    )

    return len(records)


# ---------------------------------------------------------------------------
# Game lines (CFBD spreads / over-unders)
# ---------------------------------------------------------------------------

def ingest_game_lines(session, year: int) -> int:
    """
    Store CFBD game lines (one row per game per sportsbook). Lines attach to
    games by CFBD game ID, so team/game stats must be ingested first. Existing
    rows are updated, so re-running before kickoff keeps the latest line and
    afterwards the closing line.
    """

    records = fetch_game_lines(year)

    game_ids = dict(
        session.query(Game.cfbd_game_id, Game.game_id)
        .filter(Game.cfbd_game_id.is_not(None))
    )

    rows = []
    for record in records:
        game_id = game_ids.get(record.pop("cfbd_game_id"))
        if game_id is not None:            # e.g. FCS-only games we don't track
            rows.append({"game_id": game_id, **record, "updated_at": _utcnow()})

    if rows:
        statement = pg_insert(GameLine.__table__).values(rows)
        statement = statement.on_conflict_do_update(
            constraint="uq_game_line_provider",
            set_={column: statement.excluded[column] for column in (
                "spread", "spread_open", "over_under", "over_under_open",
                "home_moneyline", "away_moneyline", "updated_at",
            )},
        )
        session.execute(statement)
    session.commit()

    print(f"[cfbd_lines] {year}: {len(rows)} lines for "
          f"{len({r['game_id'] for r in rows})} games "
          f"({len(records) - len(rows)} skipped: game not tracked)")
    return len(rows)


# ---------------------------------------------------------------------------
# Quarterbacks
# ---------------------------------------------------------------------------

def ingest_qb_stats(session, year: int) -> int:
    """
    Fetch and store QB game statistics.

    Each QB row attaches to a game by CFBD game ID, so team/game stats must
    be ingested first. (game_id, player_id) is the unique identifier; existing
    rows are skipped so re-running never creates duplicates.
    """

    raw_df = fetch_qb_season_stats(year)

    records = validate_and_format_qb_stats(raw_df)

    print(f"[cfbd_qb] Processing {len(records)} records...")

    # Load lookups once instead of querying per record.
    games_lookup = {
        cfbd_id: (game_id, game_date)
        for cfbd_id, game_id, game_date in session.query(
            Game.cfbd_game_id, Game.game_id, Game.game_date
        ).filter(Game.cfbd_game_id.is_not(None))
    }

    existing = {
        (game_id, player_id)
        for game_id, player_id in session.query(
            QBGameStat.game_id, QBGameStat.player_id
        )
    }

    team_ids = {
        name: team_id
        for name, team_id in session.query(Team.team_name, Team.team_id)
    }

    def team_id_for(name: str) -> int:
        if name not in team_ids:
            team_ids[name] = _get_or_create_team(session, name).team_id
        return team_ids[name]

    written = skipped = missing_game = unmatched_team = 0

    for index, record in enumerate(records, start=1):

        try:
            game_info = games_lookup.get(record["cfbd_game_id"])

            if game_info is None:
                missing_game += 1
                continue

            game_id, game_date = game_info

            key = (game_id, record["player_id"])

            if key in existing:
                skipped += 1
                continue

            team_name = record["team"]

            if team_name == record["home_team"]:
                opponent_name = record["away_team"]
                home_away = "home"

            elif team_name == record["away_team"]:
                opponent_name = record["home_team"]
                home_away = "away"

            else:
                # Don't guess which side the QB played for.
                unmatched_team += 1
                print(
                    f"[cfbd_qb] Skipping: team '{team_name}' is neither "
                    f"'{record['home_team']}' nor '{record['away_team']}' "
                    f"({record.get('player_name')})"
                )
                continue

            session.add(
                QBGameStat(
                    game_id=game_id,
                    player_id=record["player_id"],
                    player_name=record["player_name"],
                    team_id=team_id_for(team_name),
                    opponent_team_id=team_id_for(opponent_name),
                    passing_yards=record.get("passing_yards"),
                    passing_tds=record.get("passing_tds"),
                    passing_ints=record.get("passing_ints"),
                    completions=record.get("completions"),
                    attempts=record.get("attempts"),
                    completion_pct=record.get("completion_pct"),
                    yards_per_attempt=record.get("yards_per_attempt"),
                    rushing_yards=record.get("rushing_yards"),
                    rushing_tds=record.get("rushing_tds"),
                    rushing_attempts=record.get("rushing_attempts"),
                    game_date=game_date,
                    season=record["season"],
                    week=record.get("week"),
                    home_away=home_away,
                )
            )

            # Force PostgreSQL to insert now so errors point at this record.
            session.flush()

            existing.add(key)
            written += 1

            if written % 250 == 0:
                print(
                    f"[cfbd_qb] Inserted {written} new records so far "
                    f"(record {index}/{len(records)})"
                )

        except Exception as exc:
            session.rollback()
            print()
            print("=" * 70)
            print(f"[cfbd_qb] ERROR on record {index}:")
            print(record)
            print(f"Database error: {exc}")
            print("=" * 70)
            raise

    session.commit()

    print("[cfbd_qb] Finished processing.")
    print(f"[cfbd_qb] New records inserted: {written}")
    print(f"[cfbd_qb] Existing records skipped: {skipped}")

    if missing_game:
        print(
            f"[cfbd_qb] WARNING: {missing_game} records had no matching game. "
            f"Run team/game stats ingestion for {year} first."
        )

    if unmatched_team:
        print(
            f"[cfbd_qb] WARNING: {unmatched_team} records skipped because "
            f"the team name matched neither side of the game."
        )

    return written


# ---------------------------------------------------------------------------
# Logging + entry point
# ---------------------------------------------------------------------------

def log_run(
    session,
    source: str,
    record_count: int = 0,
    success: bool = True,
    error_message: str | None = None,
):
    """Record the result of an ingestion run in ingestion_log."""

    session.add(
        IngestionLog(
            source=source,
            run_at=_utcnow(),
            record_count=record_count,
            success=success,
            error_message=error_message,
        )
    )
    session.commit()


def _run_step(session, label: str, source: str, func, year: int):
    """
    Run one ingestion step. A failure is printed and logged but does not stop
    the other steps.
    """

    try:
        print(f"[{label}] Starting for {year}...")

        written = func(session, year)

        print(f"[{label}] Successfully wrote {written} records")

        log_run(session, source, written, True)

    except Exception as exc:
        session.rollback()

        kind = "UNAVAILABLE" if isinstance(exc, CFBDUnavailableError) else "FAILED"

        print(f"[{label}] {kind}: {exc}")

        try:
            log_run(session, source, 0, False, str(exc))
        except Exception:
            session.rollback()


def main():

    print("SCRIPT STARTED")

    parser = argparse.ArgumentParser(
        description="Run Just-Bet-It data ingestion."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=datetime.now(timezone.utc).year,
        help="Season year to ingest.",
    )

    # Both Odds API steps are opt-in so a normal ingestion run never spends
    # the team's limited Odds API credits.
    parser.add_argument(
        "--odds",
        action="store_true",
        help="Also snapshot game odds: moneyline, spread, total (3 Odds API credits).",
    )

    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help=argparse.SUPPRESS,  # old flag; odds are now off unless --odds
    )

    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip team/game ingestion.",
    )

    parser.add_argument(
        "--skip-lines",
        action="store_true",
        help="Skip CFBD game lines (spreads / over-unders).",
    )

    parser.add_argument(
        "--skip-qb",
        action="store_true",
        help="Skip QB stats ingestion.",
    )

    parser.add_argument(
        "--props",
        action="store_true",
        help="Also snapshot QB passing-yards prop lines for upcoming games "
             "(costs 1 Odds API credit per game).",
    )

    parser.add_argument(
        "--props-window-hours",
        type=int,
        default=PROPS_WINDOW_HOURS,
        help="With --props, only fetch games kicking off within this many hours.",
    )

    parser.add_argument(
        "--props-credit-reserve",
        type=int,
        default=PROPS_CREDIT_RESERVE,
        help="With --props, stop before remaining Odds API credits drop below this.",
    )

    parser.add_argument(
        "--props-max-games",
        type=int,
        default=PROPS_MAX_GAMES,
        help="With --props, make at most this many paid requests.",
    )

    args = parser.parse_args()

    init_db()

    session = SessionLocal()

    try:

        if args.odds:
            _run_step(session, "odds", "odds", ingest_odds, args.year)

        if not args.skip_stats:
            _run_step(
                session, "cfbd_stats", "cfbd_stats",
                ingest_team_stats, args.year,
            )

        if not args.skip_lines:
            _run_step(
                session, "cfbd_lines", "cfbd_lines",
                ingest_game_lines, args.year,
            )

        if not args.skip_qb:
            _run_step(
                session, "cfbd_qb", "cfbd_qb",
                ingest_qb_stats, args.year,
            )

        if args.props:
            _run_step(
                session, "props", "odds_props",
                partial(
                    ingest_qb_props,
                    window_hours=args.props_window_hours,
                    credit_reserve=args.props_credit_reserve,
                    max_games=args.props_max_games,
                ),
                args.year,
            )

    finally:

        print(f"Ingestion run complete for {args.year}.")

        session.close()


if __name__ == "__main__":
    main()
