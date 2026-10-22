"""
Maps The Odds API's team names ("Ohio State Buckeyes") onto the CFBD school
names the teams table uses ("Ohio State").

An Odds API name is the school followed by its mascot, so the lookup is built
from CFBD's team list: school (and each of its alternate names) + mascot.
Names are compared ignoring case, accents and punctuation, so "Hawaii" matches
"Hawai'i" and "William and Mary" matches "William & Mary".

Nothing in this file touches the database.
"""

import re
import unicodedata

# Odds API names that school/alternate name + mascot doesn't produce.
ODDS_NAME_OVERRIDES = {
    "Nicholls State Colonels": "Nicholls",
    "Southeastern Louisiana Lions": "SE Louisiana",
}


def normalize(name: str) -> str:
    """Lowercase ASCII letters and digits only; '&' counts as 'and'."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]", "", name)


def build_odds_name_map(cfbd_teams) -> dict[str, str]:
    """
    Build {normalized Odds API name: CFBD school} from CFBD team dicts
    (see cfbd_ingest.fetch_teams). A name that could mean more than one
    school is left out rather than guessed.
    """
    candidates = {}

    for team in cfbd_teams:
        school = team.get("school")
        mascot = team.get("mascot")

        if not school or not mascot:
            continue

        for name in {school, *(team.get("alternateNames") or [])}:
            key = normalize(f"{name} {mascot}")
            candidates.setdefault(key, set()).add(school)

    name_map = {
        key: next(iter(schools))
        for key, schools in candidates.items()
        if len(schools) == 1
    }

    for odds_name, school in ODDS_NAME_OVERRIDES.items():
        name_map[normalize(odds_name)] = school

    return name_map


def resolve_odds_team_name(odds_name: str, name_map: dict[str, str]):
    """The CFBD school for an Odds API team name, or None if it's unknown."""
    return name_map.get(normalize(odds_name))
