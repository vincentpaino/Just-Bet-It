"""
One-off cleanup: merge the duplicate teams and games that odds ingestion
created before Odds API team names were mapped to CFBD names.

Before that fix, every Odds API name ("Ohio State Buckeyes") became its own
team, and every odds game became its own game with no CFBD ID. This script:
  1. maps each of those teams to its CFBD team (or renames it to the CFBD
     school if the table has no team by that name yet),
  2. moves each odds-only game's odds onto the matching CFBD game, flipping
     home/away values when the two list the teams the other way round, then
     deletes the odds-only game; one with no CFBD match is kept but pointed
     at the CFBD teams, so the next CFBD ingestion run adopts it,
  3. deletes the duplicate teams, which nothing references any more.

Everything runs in one transaction. Dry run by default; pass --apply to commit.
Run from the project root:

    python -m scripts.merge_odds_teams            # show what would change
    python -m scripts.merge_odds_teams --apply    # do it
"""

import argparse
import sys
from datetime import timedelta

from sqlalchemy import text

from src.db.models import Game
from src.db.session import SessionLocal
from src.ingestion.cfbd_ingest import fetch_teams
from src.ingestion.run_ingestion import _same_matchup
from src.ingestion.team_names import build_odds_name_map, resolve_odds_team_name

# Every column that references teams.team_id
TEAM_REFERENCES = [
    ("games", "home_team_id"),
    ("games", "away_team_id"),
    ("qb_game_stats", "team_id"),
    ("qb_game_stats", "opponent_team_id"),
    ("qb_game_features", "team_id"),
    ("qb_game_features", "opponent_team_id"),
    ("user_predictions", "predicted_winner"),
]

# Tables other than odds that reference games.game_id; an odds-only game
# that any of these point at is left in place rather than deleted.
OTHER_GAME_REFERENCES = [
    "features", "qb_game_stats", "qb_game_features", "qb_prop_lines", "user_predictions",
]


def map_duplicate_teams(session, name_map):
    """
    Return {duplicate team_id: CFBD team_id}. Only teams that appear solely
    in games without a CFBD ID (i.e. created by odds ingestion) are touched.
    """
    rows = session.execute(text("""
        SELECT t.team_id, t.team_name
        FROM teams t
        WHERE EXISTS (SELECT 1 FROM games g WHERE g.cfbd_game_id IS NULL
                      AND t.team_id IN (g.home_team_id, g.away_team_id))
          AND NOT EXISTS (SELECT 1 FROM games g WHERE g.cfbd_game_id IS NOT NULL
                          AND t.team_id IN (g.home_team_id, g.away_team_id))
          AND NOT EXISTS (SELECT 1 FROM qb_game_stats q
                          WHERE t.team_id IN (q.team_id, q.opponent_team_id))
    """)).all()

    team_ids = dict(session.execute(text("SELECT team_name, team_id FROM teams")).all())

    duplicates, renamed, unresolved = {}, [], []

    for team_id, team_name in rows:
        school = resolve_odds_team_name(team_name, name_map)

        if school is None:
            unresolved.append(team_name)
        elif school == team_name:
            continue
        elif school in team_ids:
            duplicates[team_id] = team_ids[school]
        else:
            session.execute(
                text("UPDATE teams SET team_name = :school WHERE team_id = :id"),
                {"school": school, "id": team_id},
            )
            team_ids[school] = team_id
            renamed.append(f"{team_name} -> {school}")

    return duplicates, renamed, unresolved


def merge_games(session, duplicates):
    """Move odds from odds-only games onto their CFBD games."""
    merged = flipped = dropped_snapshots = kept = ambiguous = blocked = 0

    stubs = session.query(Game).filter(Game.cfbd_game_id.is_(None)).all()

    for stub in stubs:
        home_id = duplicates.get(stub.home_team_id, stub.home_team_id)
        away_id = duplicates.get(stub.away_team_id, stub.away_team_id)

        matches = (
            session.query(Game)
            .filter(Game.cfbd_game_id.is_not(None))
            .filter(Game.game_date.between(
                stub.game_date - timedelta(days=1),
                stub.game_date + timedelta(days=1),
            ))
            .filter(_same_matchup(home_id, away_id))
            .all()
        )

        referenced = any(
            session.execute(
                text(f"SELECT 1 FROM {table} WHERE game_id = :id LIMIT 1"),
                {"id": stub.game_id},
            ).first()
            for table in OTHER_GAME_REFERENCES
        )

        if len(matches) != 1 or referenced:
            # No (unique) CFBD game yet: keep it, but on the CFBD teams.
            stub.home_team_id, stub.away_team_id = home_id, away_id
            kept += 1
            ambiguous += len(matches) > 1
            blocked += referenced
            continue

        target = matches[0]
        params = {"stub": stub.game_id, "target": target.game_id}

        # Odds rows are stored from the game's home side, so flip them if
        # the CFBD game lists the teams the other way round. (Postgres
        # evaluates every right-hand side with the old values, so a = b,
        # b = a swaps.)
        if target.home_team_id != home_id:
            session.execute(text("""
                UPDATE odds SET
                    home_moneyline = away_moneyline, away_moneyline = home_moneyline,
                    implied_prob_home = implied_prob_away, implied_prob_away = implied_prob_home,
                    spread = -spread
                WHERE game_id = :stub
            """), params)
            flipped += 1

        # Drop any snapshot the CFBD game already has (same book + time).
        dropped_snapshots += session.execute(text("""
            DELETE FROM odds o WHERE o.game_id = :stub AND EXISTS (
                SELECT 1 FROM odds t WHERE t.game_id = :target
                AND t.sportsbook = o.sportsbook AND t.fetched_at = o.fetched_at)
        """), params).rowcount

        session.execute(text("UPDATE odds SET game_id = :target WHERE game_id = :stub"), params)
        session.execute(text("DELETE FROM games WHERE game_id = :stub"), params)
        merged += 1

    session.flush()

    return {
        "odds-only games merged into CFBD games": merged,
        "  ...of which home/away flipped": flipped,
        "  duplicate snapshots dropped": dropped_snapshots,
        "odds-only games kept (no CFBD match yet)": kept,
        "  ...of which matched more than one CFBD game": ambiguous,
        "  ...of which referenced by other tables": blocked,
    }


def remove_duplicate_teams(session, duplicates):
    """Point any remaining references at the CFBD team, then delete duplicates."""
    for dup_id, cfbd_id in duplicates.items():
        for table, column in TEAM_REFERENCES:
            session.execute(
                text(f"UPDATE {table} SET {column} = :cfbd WHERE {column} = :dup"),
                {"cfbd": cfbd_id, "dup": dup_id},
            )
        session.execute(text("DELETE FROM teams WHERE team_id = :dup"), {"dup": dup_id})


def counts(session):
    return session.execute(text("""
        SELECT (SELECT COUNT(*) FROM teams),
               (SELECT COUNT(*) FROM games),
               (SELECT COUNT(*) FROM odds),
               (SELECT COUNT(DISTINCT o.game_id) FROM odds o
                JOIN games g USING (game_id) WHERE g.cfbd_game_id IS NOT NULL)
    """)).one()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="Commit the changes.")
    args = parser.parse_args()

    name_map = build_odds_name_map(fetch_teams())

    session = SessionLocal()

    try:
        before = counts(session)

        duplicates, renamed, unresolved = map_duplicate_teams(session, name_map)
        session.flush()
        game_stats = merge_games(session, duplicates)
        remove_duplicate_teams(session, duplicates)
        session.flush()

        after = counts(session)

        print(f"Duplicate teams merged into CFBD teams: {len(duplicates)}")
        print(f"Teams renamed to their CFBD school: {len(renamed)} {renamed or ''}")
        if unresolved:
            print(f"WARNING: no CFBD match (left alone, add to ODDS_NAME_OVERRIDES): {unresolved}")
        for label, value in game_stats.items():
            print(f"{label}: {value}")

        print()
        for label, b, a in zip(
            ["teams", "games", "odds rows", "games with odds linked to CFBD"], before, after
        ):
            print(f"{label:32} {b:>6} -> {a}")

        if args.apply:
            session.commit()
            print("\nCommitted.")
        else:
            session.rollback()
            print("\nDry run: nothing was changed. Re-run with --apply to commit.")

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
