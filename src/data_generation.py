"""
Simulated player-season dataset.

WHY THIS EXISTS
---------------
FBref / Opta / StatsBomb season data cannot be redistributed with this
repository, so the app ships with a *simulated* player universe instead. It is
labelled as simulated everywhere it is used and nothing in the app claims these
are real footballers.

The simulation is not random noise. Every player is drawn from a latent-trait
model:

    1. each player belongs to a role profile (e.g. "ball-playing CB") which
       defines a mean vector over interpretable traits (defending, aerial,
       progression, carrying, creation, finishing, ...);
    2. traits are perturbed per player and per season;
    3. each observable per-90 rate is  base_rate(position) * exp(loadings . traits);
    4. season totals are then *sampled*  -  counts from a Poisson process over
       the player's actual minutes, success rates from a Binomial over their
       attempts.

Step 4 is the important one: it means a player with 300 minutes has a genuinely
noisy per-90 profile, exactly like real data, which is why the minimum-minutes
filter in the app matters.

Because the archetypes are known by construction, they are kept in the raw file
as `true_role_profile` and used **only** by scripts/validate_models.py to check
whether K-Means recovers real structure. No model, page or score in the app
reads that column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEAGUES, SEASONS

GROUP_ORDER = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "FW"]

# Squad shape used for every simulated club (20 players per team).
SQUAD_SHAPE = {"GK": 2, "CB": 4, "FB": 3, "DM": 2, "CM": 3, "AM": 2, "W": 2, "FW": 2}

DETAILED_CHOICES = {
    "GK": ["GK"],
    "CB": ["CB", "LCB", "RCB"],
    "FB": ["LB", "RB", "LWB", "RWB"],
    "DM": ["DM"],
    "CM": ["CM", "B2B"],
    "AM": ["AM", "SS"],
    "W": ["LW", "RW", "LM", "RM"],
    "FW": ["CF", "ST"],
}

TRAITS = [
    "defending",
    "aerial",
    "pass_volume",
    "pass_accuracy",
    "long_pass",
    "progression",
    "carrying",
    "dribbling",
    "creation",
    "crossing",
    "shooting",
    "finishing",
    "box_presence",
    "pressing",
    "gk_shotstop",
    "gk_claim",
    "gk_sweep",
    "gk_dist",
]

# --------------------------------------------------------------------------
# Role profiles -- the generative archetypes
# --------------------------------------------------------------------------
# Values are offsets in trait standard deviations. Anything unlisted is 0.

ROLE_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "GK": {
        "Shot-stopper": {"gk_shotstop": 1.1, "gk_dist": -0.5, "gk_sweep": -0.4},
        "Sweeper-keeper": {"gk_sweep": 1.2, "gk_dist": 0.8, "pass_accuracy": 0.7, "gk_shotstop": -0.2},
        "Commanding keeper": {"gk_claim": 1.2, "aerial": 0.9, "gk_shotstop": 0.3, "gk_dist": -0.3},
        "Traditional keeper": {"gk_dist": -0.8, "long_pass": 0.9, "gk_claim": 0.2},
    },
    "CB": {
        "Ball-playing centre-back": {
            "progression": 1.1, "pass_accuracy": 0.9, "carrying": 0.8, "pass_volume": 0.6,
            "defending": -0.2, "aerial": -0.1,
        },
        "Aggressive stopper": {
            "defending": 1.1, "pressing": 0.9, "aerial": 0.3, "progression": -0.6,
            "pass_accuracy": -0.5, "carrying": -0.4,
        },
        "Aerial dominator": {
            "aerial": 1.3, "defending": 0.5, "long_pass": 0.4, "carrying": -0.6, "progression": -0.5,
        },
        "No-frills defender": {
            "defending": 0.4, "aerial": 0.4, "pass_volume": -0.7, "progression": -0.8, "long_pass": 0.5,
        },
    },
    "FB": {
        "Attacking wing-back": {
            "crossing": 1.1, "creation": 0.9, "carrying": 0.8, "progression": 0.7,
            "box_presence": 0.5, "defending": -0.6,
        },
        "Inverted full-back": {
            "pass_volume": 1.0, "pass_accuracy": 0.9, "progression": 0.7, "crossing": -0.9,
            "dribbling": -0.3,
        },
        "Defensive full-back": {
            "defending": 1.1, "aerial": 0.6, "pressing": 0.5, "crossing": -0.4, "creation": -0.6,
        },
        "Overlapping runner": {
            "carrying": 1.0, "dribbling": 0.9, "crossing": 0.6, "progression": 0.5, "pass_accuracy": -0.4,
        },
    },
    "DM": {
        "Deep-lying playmaker": {
            "pass_volume": 1.1, "pass_accuracy": 1.0, "progression": 0.9, "long_pass": 0.8,
            "defending": -0.4, "pressing": -0.4,
        },
        "Ball-winning destroyer": {
            "defending": 1.2, "pressing": 1.0, "aerial": 0.4, "progression": -0.6, "creation": -0.5,
        },
        "Anchor man": {
            "defending": 0.7, "aerial": 0.8, "pass_accuracy": 0.5, "carrying": -0.7, "creation": -0.6,
        },
        "Progressive carrier": {
            "carrying": 1.1, "progression": 0.9, "dribbling": 0.8, "pressing": 0.3, "long_pass": -0.4,
        },
    },
    "CM": {
        "Box-to-box engine": {
            "pressing": 0.9, "defending": 0.6, "carrying": 0.6, "box_presence": 0.5, "shooting": 0.4,
        },
        "Creative midfielder": {
            "creation": 1.1, "progression": 0.8, "pass_accuracy": 0.5, "defending": -0.6, "pressing": -0.5,
        },
        "Metronome": {
            "pass_volume": 1.2, "pass_accuracy": 1.0, "progression": 0.4, "dribbling": -0.6, "shooting": -0.5,
        },
        "Late-arriving runner": {
            "box_presence": 1.1, "shooting": 0.9, "finishing": 0.5, "progression": 0.3, "pass_volume": -0.5,
        },
    },
    "AM": {
        "Classic number 10": {
            "creation": 1.2, "pass_accuracy": 0.6, "progression": 0.6, "defending": -0.7, "pressing": -0.6,
        },
        "Goalscoring second striker": {
            "shooting": 1.0, "finishing": 0.8, "box_presence": 1.0, "creation": -0.3, "pass_volume": -0.5,
        },
        "Dribbling playmaker": {
            "dribbling": 1.2, "carrying": 1.0, "creation": 0.6, "pass_accuracy": -0.4,
        },
        "Pressing forward-midfielder": {
            "pressing": 1.2, "defending": 0.7, "carrying": 0.3, "creation": -0.3,
        },
    },
    "W": {
        "Inside forward": {
            "shooting": 1.0, "finishing": 0.7, "box_presence": 0.9, "crossing": -0.7, "creation": -0.2,
        },
        "Direct dribbling winger": {
            "dribbling": 1.3, "carrying": 1.1, "crossing": 0.3, "pass_accuracy": -0.6,
        },
        "Creative wide playmaker": {
            "creation": 1.1, "crossing": 0.9, "pass_accuracy": 0.5, "shooting": -0.5, "dribbling": -0.2,
        },
        "Hard-working wide man": {
            "pressing": 1.1, "defending": 0.9, "shooting": -0.4, "dribbling": -0.5, "creation": -0.3,
        },
    },
    "FW": {
        "Poacher": {
            "finishing": 1.1, "box_presence": 1.2, "shooting": 0.6, "pass_volume": -0.9,
            "creation": -0.7, "carrying": -0.6,
        },
        "Target man": {
            "aerial": 1.4, "box_presence": 0.5, "dribbling": -0.7, "carrying": -0.6, "pressing": -0.3,
        },
        "Pressing forward": {
            "pressing": 1.3, "defending": 0.8, "carrying": 0.3, "finishing": -0.3,
        },
        "Complete / link-up forward": {
            "creation": 1.0, "pass_accuracy": 0.8, "carrying": 0.8, "dribbling": 0.6, "aerial": 0.2,
        },
    },
}

# --------------------------------------------------------------------------
# Base per-90 rates by position group
# --------------------------------------------------------------------------
# Order matches GROUP_ORDER. Values are approximate real-world per-90 means.

def _rates(values: list[float]) -> dict[str, float]:
    return dict(zip(GROUP_ORDER, values))


BASE_RATES: dict[str, dict[str, float]] = {
    #                        GK     CB    FB    DM    CM    AM    W     FW
    "shots":        _rates([0.01, 0.45, 0.55, 0.65, 1.00, 1.85, 1.90, 2.55]),
    "key_passes":   _rates([0.02, 0.25, 0.85, 0.65, 1.05, 1.85, 1.75, 1.05]),
    "passes_attempted": _rates([30.0, 58.0, 47.0, 58.0, 52.0, 38.0, 31.0, 25.0]),
    "progressive_passes": _rates([1.6, 3.3, 3.2, 4.4, 4.0, 3.0, 1.9, 1.3]),
    "passes_into_final_third": _rates([1.9, 3.2, 2.9, 4.4, 3.9, 2.6, 1.4, 0.9]),
    "passes_into_pen_area": _rates([0.02, 0.15, 0.85, 0.35, 0.65, 1.50, 1.35, 0.75]),
    "through_balls": _rates([0.01, 0.03, 0.10, 0.15, 0.22, 0.42, 0.28, 0.20]),
    "crosses":      _rates([0.02, 0.15, 2.40, 0.35, 0.75, 1.40, 3.10, 0.75]),
    "switches":     _rates([0.15, 0.25, 0.35, 0.55, 0.50, 0.35, 0.20, 0.10]),
    "long_passes_attempted": _rates([14.0, 7.0, 3.6, 5.0, 3.6, 2.2, 1.5, 1.5]),
    "progressive_receptions": _rates([0.05, 0.35, 3.20, 1.90, 3.40, 7.00, 8.50, 6.00]),
    "progressive_carries": _rates([0.05, 0.90, 2.40, 1.60, 2.30, 3.40, 4.20, 2.30]),
    "carries_into_pen_area": _rates([0.00, 0.02, 0.35, 0.12, 0.30, 0.90, 1.50, 1.20]),
    "dribbles_attempted": _rates([0.05, 0.40, 1.50, 0.90, 1.30, 2.60, 3.60, 2.10]),
    "touches_att_pen": _rates([0.02, 0.35, 0.80, 0.40, 0.90, 2.40, 3.30, 5.20]),
    "tackles":      _rates([0.05, 1.20, 1.90, 2.00, 1.70, 1.30, 1.30, 0.85]),
    "interceptions": _rates([0.10, 1.40, 1.30, 1.40, 1.00, 0.70, 0.55, 0.40]),
    "blocks":       _rates([0.05, 1.50, 1.20, 1.30, 1.00, 0.80, 0.60, 0.50]),
    "clearances":   _rates([0.60, 4.20, 1.90, 1.40, 0.90, 0.40, 0.35, 0.60]),
    "ball_recoveries": _rates([2.60, 5.20, 5.00, 6.40, 5.80, 4.60, 4.00, 3.20]),
    "pressures":    _rates([0.50, 9.00, 14.0, 16.0, 16.0, 18.0, 17.0, 16.0]),
    "aerials_contested": _rates([0.90, 4.20, 2.00, 2.30, 1.80, 1.40, 1.20, 3.60]),
    "fouls_committed": _rates([0.10, 1.10, 1.20, 1.60, 1.40, 1.30, 1.10, 1.20]),
    "errors":       _rates([0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]),
    "miscontrols":  _rates([0.10, 0.50, 1.10, 1.00, 1.30, 2.20, 2.60, 2.30]),
    "dispossessed": _rates([0.05, 0.30, 0.80, 0.70, 1.00, 1.80, 2.20, 1.70]),
    "touches":      _rates([40.0, 68.0, 62.0, 72.0, 68.0, 55.0, 48.0, 36.0]),
    # goalkeeping
    "gk_shots_on_target_against": _rates([3.60] + [0.0] * 7),
    "gk_crosses_faced": _rates([4.60] + [0.0] * 7),
    "gk_def_actions_outside_box": _rates([1.00] + [0.0] * 7),
    "gk_launches_attempted": _rates([11.0] + [0.0] * 7),
}

# Multiplicative trait loadings, applied on the log scale.
RATE_LOADINGS: dict[str, dict[str, float]] = {
    "shots": {"shooting": 0.40, "box_presence": 0.20},
    "key_passes": {"creation": 0.42, "crossing": 0.08},
    "passes_attempted": {"pass_volume": 0.28, "pass_accuracy": 0.06},
    "progressive_passes": {"progression": 0.35, "pass_volume": 0.14},
    "passes_into_final_third": {"progression": 0.32, "pass_volume": 0.14},
    "passes_into_pen_area": {"creation": 0.30, "progression": 0.15, "crossing": 0.12},
    "through_balls": {"creation": 0.42, "progression": 0.10},
    "crosses": {"crossing": 0.55, "creation": 0.10},
    "switches": {"long_pass": 0.45, "pass_volume": 0.10},
    "long_passes_attempted": {"long_pass": 0.45, "pass_volume": 0.12},
    "progressive_receptions": {"progression": 0.20, "box_presence": 0.25, "carrying": 0.10},
    "progressive_carries": {"carrying": 0.42, "dribbling": 0.15},
    "carries_into_pen_area": {"carrying": 0.20, "box_presence": 0.32, "dribbling": 0.12},
    "dribbles_attempted": {"dribbling": 0.50, "carrying": 0.12},
    "touches_att_pen": {"box_presence": 0.45, "shooting": 0.10},
    "tackles": {"defending": 0.35, "pressing": 0.20},
    "interceptions": {"defending": 0.35, "pressing": 0.05},
    "blocks": {"defending": 0.26},
    "clearances": {"defending": 0.25, "aerial": 0.20},
    "ball_recoveries": {"pressing": 0.22, "defending": 0.16},
    "pressures": {"pressing": 0.30},
    "aerials_contested": {"aerial": 0.35},
    "fouls_committed": {"defending": 0.15, "pressing": 0.16},
    "errors": {"pass_accuracy": -0.28},
    "miscontrols": {"dribbling": 0.25, "carrying": 0.20, "pass_accuracy": -0.20},
    "dispossessed": {"dribbling": 0.30, "carrying": 0.16, "pass_accuracy": -0.15},
    "touches": {"pass_volume": 0.22, "carrying": 0.10},
    "gk_shots_on_target_against": {},
    "gk_crosses_faced": {},
    "gk_def_actions_outside_box": {"gk_sweep": 0.55},
    "gk_launches_attempted": {"gk_dist": -0.35, "long_pass": 0.30},
}

# Success probabilities: (base probability by group, logit loadings).
BASE_PROBS: dict[str, dict[str, float]] = {
    #                       GK     CB    FB    DM    CM    AM    W     FW
    "pass_pct":     _rates([0.72, 0.86, 0.80, 0.87, 0.85, 0.80, 0.76, 0.72]),
    "long_pass_pct": _rates([0.42, 0.60, 0.55, 0.62, 0.60, 0.55, 0.50, 0.48]),
    "tackle_win_pct": _rates([0.70, 0.62, 0.60, 0.62, 0.61, 0.60, 0.59, 0.58]),
    "pressure_success_pct": _rates([0.30, 0.30, 0.29, 0.30, 0.29, 0.28, 0.28, 0.27]),
    "dribble_success_pct": _rates([0.50, 0.55, 0.50, 0.52, 0.51, 0.49, 0.48, 0.48]),
    "shot_accuracy_pct": _rates([0.30, 0.30, 0.31, 0.32, 0.33, 0.34, 0.34, 0.36]),
    "aerial_win_pct": _rates([0.75, 0.62, 0.50, 0.52, 0.50, 0.45, 0.42, 0.48]),
    "gk_save_pct": _rates([0.70] + [0.0] * 7),
    "gk_cross_stop_pct": _rates([0.065] + [0.0] * 7),
    "gk_launch_pct": _rates([0.42] + [0.0] * 7),
}

PROB_LOADINGS: dict[str, dict[str, float]] = {
    "pass_pct": {"pass_accuracy": 0.55, "long_pass": -0.25, "progression": -0.14},
    "long_pass_pct": {"long_pass": 0.35, "pass_accuracy": 0.25},
    "tackle_win_pct": {"defending": 0.30},
    "pressure_success_pct": {"pressing": 0.25},
    "dribble_success_pct": {"dribbling": 0.35},
    "shot_accuracy_pct": {"finishing": 0.20, "shooting": 0.10},
    "aerial_win_pct": {"aerial": 0.55},
    "gk_save_pct": {"gk_shotstop": 0.50},
    "gk_cross_stop_pct": {"gk_claim": 0.55},
    "gk_launch_pct": {"gk_dist": 0.40, "long_pass": 0.15},
}

# Mean non-penalty xG per shot by position group (shot quality).
BASE_XG_PER_SHOT = _rates([0.06, 0.07, 0.07, 0.07, 0.08, 0.09, 0.09, 0.11])
BASE_XA_PER_KEY_PASS = 0.105
BASE_HEIGHT = _rates([189.0, 187.0, 179.0, 182.0, 180.0, 178.0, 177.0, 184.0])

# --------------------------------------------------------------------------
# Name pools (fictional)
# --------------------------------------------------------------------------

FIRST_NAMES = [
    "Adrian", "Alvaro", "Andres", "Anton", "Arne", "Bastian", "Bruno", "Casper", "Cedric",
    "Damian", "Dario", "Diogo", "Dominik", "Eduardo", "Emil", "Enzo", "Fabio", "Felix",
    "Filip", "Gabriel", "Gustav", "Hugo", "Ibrahim", "Idris", "Ivan", "Jasper", "Joan",
    "Jonas", "Jorge", "Julen", "Kacper", "Karim", "Kasper", "Kevin", "Lars", "Lorenzo",
    "Luca", "Lukas", "Malik", "Marek", "Mateo", "Mathis", "Matteo", "Maxime", "Milan",
    "Mirko", "Moussa", "Nikola", "Noah", "Olivier", "Oscar", "Pau", "Pedro", "Rafael",
    "Rasmus", "Remi", "Ruben", "Samuel", "Sebastian", "Sergi", "Simon", "Stefan", "Sven",
    "Teddy", "Thiago", "Tobias", "Tomas", "Valentin", "Victor", "Vincent", "Yannick",
    "Youssef", "Zeno", "Callum", "Declan", "Elliot", "Harvey", "Jude", "Reece",
]

LAST_NAMES = [
    "Abrahams", "Adeyemi", "Almeida", "Andersen", "Antunes", "Baierl", "Barreto", "Beckmann",
    "Belanger", "Bergstrom", "Bianchi", "Bogdan", "Bortolo", "Brandt", "Cabral", "Cardoso",
    "Castellan", "Cavani-Ross", "Chandler", "Coppens", "Cortes", "Dahlberg", "Dembo", "Dijkstra",
    "Doncic", "Duarte", "Egberts", "Elmander", "Farrell", "Fenner", "Ferrandis", "Fontaine",
    "Gallardo", "Gasperi", "Gerber", "Girard", "Grunwald", "Haaskiv", "Halvorsen", "Hartmann",
    "Herrera", "Hoekstra", "Ionescu", "Jankovic", "Jelinek", "Kaminski", "Karlsen", "Keita",
    "Kovacic-Bell", "Krause", "Laurent", "Lindqvist", "Lombardi", "Maas", "Magnusson", "Mandic",
    "Marchetti", "Mensah", "Meunier", "Molnar", "Moreira", "Nasri-Cole", "Nilsen", "Novak",
    "Okafor", "Olsson", "Ortega", "Pavlovic", "Peeters", "Pereira", "Petrov", "Quintero",
    "Rakitic-Jones", "Renard", "Ribeiro", "Rossi", "Sanchez", "Schneider", "Silvestri", "Sorensen",
    "Stankovic", "Szabo", "Tavares", "Thorne", "Traore", "Vandenberg", "Varela", "Vermeulen",
    "Vlahovic-Reid", "Wagner", "Weimann", "Wojcik", "Zanetti", "Zielinski", "Ashworth", "Cadogan",
]

TOWN_NAMES = [
    "Alborough", "Brenham", "Calderon", "Dunmore", "Estrela", "Farnwood", "Grevenberg", "Halverd",
    "Iversund", "Jorda", "Kalstad", "Lomberg", "Marveil", "Norvik", "Oakhaven", "Pontarra",
    "Quintal", "Ravensholt", "Sarrion", "Tarnow", "Ulvedal", "Vendrell", "Westmere", "Ystad",
    "Zermatten", "Ardsley", "Bergheim", "Cortesa", "Delvaux", "Enskede", "Follonica", "Grimsby-le",
    "Havndal", "Ilberg", "Jansen", "Kirkwall", "Lindau", "Montoro", "Nordhaven", "Ostergaard",
    "Pallena", "Rovaniemi", "Sundberg", "Torrente", "Ubeda", "Valberg", "Wolfsheim", "Xativa",
    "Ynysgar", "Zandvoort", "Ashbourne", "Beaufort", "Cranleigh", "Dovergate", "Elmsworth",
    "Fairhurst", "Glenmara", "Holbeck", "Inverloch", "Jarrowfield", "Kelsford", "Langmere",
]

CLUB_SUFFIX = {
    "England": ["United", "City", "Town", "Rovers", "Athletic", "FC", "Wanderers"],
    "Spain": ["CF", "Deportivo", "Atletico", "Real", "UD", "Racing"],
    "Italy": ["Calcio", "AC", "US", "Sportiva", "FC"],
    "Germany": ["SV", "FC", "SC", "VfB", "Borussia"],
    "France": ["Olympique", "FC", "AS", "Stade", "RC"],
    "Netherlands": ["FC", "SC", "VV", "Sparta"],
    "Portugal": ["SC", "Sporting", "FC", "Uniao"],
    "Belgium": ["KV", "Royal", "FC", "Union"],
    "Turkey": ["SK", "Spor", "FK"],
    "Austria": ["SV", "FC", "SK"],
    "Denmark": ["BK", "IF", "FC"],
    "Switzerland": ["FC", "SC", "Grasshopper"],
    "Poland": ["KS", "Legia", "GKS"],
}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_teams(rng: np.random.Generator) -> pd.DataFrame:
    """One row per club, with a strength and an average possession share."""
    used: set[str] = set()
    rows = []
    for league in LEAGUES:
        for i in range(league["teams"]):
            while True:
                town = rng.choice(TOWN_NAMES)
                suffix = rng.choice(CLUB_SUFFIX[league["country"]])
                name = f"{town} {suffix}"
                if name not in used:
                    used.add(name)
                    break
            # Strength within the league, then shifted by the league's level.
            within = rng.normal(0, 1)
            strength = 0.75 * within + 2.2 * (league["strength"] - 0.80)
            rows.append(
                {
                    "team": name,
                    "league": league["name"],
                    "country": league["country"],
                    "league_tier": league["tier"],
                    "league_strength": league["strength"],
                    "team_strength": strength,
                    "team_possession": float(np.clip(50 + 6.5 * strength + rng.normal(0, 2.0), 34, 68)),
                }
            )
    return pd.DataFrame(rows)


def build_player_pool(rng: np.random.Generator, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per player identity: club, position, age, traits, latent ability."""
    used_names: set[str] = set()
    rows = []
    pid = 0
    for team in teams.itertuples():
        for group, count in SQUAD_SHAPE.items():
            profiles = list(ROLE_PROFILES[group])
            for _ in range(count):
                pid += 1
                while True:
                    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                    if name not in used_names:
                        used_names.add(name)
                        break
                profile = str(rng.choice(profiles))
                # Ability: better leagues and better clubs hold better players.
                ability = float(
                    rng.normal(0, 1) * 0.85
                    + 0.45 * team.team_strength
                    + 1.6 * (team.league_strength - 0.80)
                )
                age = float(np.clip(rng.normal(25.6, 4.3), 16, 39).round(1))
                rows.append(
                    {
                        "player_id": f"P{pid:05d}",
                        "player": name,
                        "position_group": group,
                        "position": str(rng.choice(DETAILED_CHOICES[group])),
                        "base_age": age,
                        "team": team.team,
                        "league": team.league,
                        "role_profile": profile,
                        "ability": ability,
                    }
                )
    return pd.DataFrame(rows)


def _draw_traits(
    rng: np.random.Generator, pool: pd.DataFrame, season_jitter: float
) -> pd.DataFrame:
    """Trait matrix: role-profile mean + ability uplift + player/season noise."""
    n = len(pool)
    traits = pd.DataFrame(rng.normal(0, 0.72, size=(n, len(TRAITS))), columns=TRAITS, index=pool.index)
    for (group, profile), idx in pool.groupby(["position_group", "role_profile"]).groups.items():
        offsets = ROLE_PROFILES[group][profile]
        for trait, value in offsets.items():
            traits.loc[idx, trait] += value
    # Better players are better at most things, but the role shape dominates.
    ability = pool["ability"].to_numpy()[:, None]
    uplift = np.array([0.30 if t not in {"pass_volume"} else 0.12 for t in TRAITS])[None, :]
    traits += ability * uplift
    traits += rng.normal(0, season_jitter, size=traits.shape)
    return traits


def _rate_matrix(stat: str, groups: np.ndarray, traits: pd.DataFrame) -> np.ndarray:
    base = np.array([BASE_RATES[stat][g] for g in groups])
    log_mult = np.zeros(len(groups))
    for trait, loading in RATE_LOADINGS.get(stat, {}).items():
        log_mult += loading * traits[trait].to_numpy()
    return base * np.exp(log_mult)


def _prob_vector(stat: str, groups: np.ndarray, traits: pd.DataFrame) -> np.ndarray:
    base = np.array([BASE_PROBS[stat][g] for g in groups])
    adj = _logit(np.clip(base, 1e-6, 1 - 1e-6))
    for trait, loading in PROB_LOADINGS.get(stat, {}).items():
        adj += loading * traits[trait].to_numpy()
    return _sigmoid(adj)


def simulate_season(
    rng: np.random.Generator,
    pool: pd.DataFrame,
    teams: pd.DataFrame,
    season: str,
    season_index: int,
) -> pd.DataFrame:
    """Sample one season of counting statistics for every player in the pool."""
    df = pool.merge(
        teams[["team", "league", "league_tier", "league_strength", "team_strength", "team_possession"]],
        on=["team", "league"],
        how="left",
    ).reset_index(drop=True)
    n = len(df)
    groups = df["position_group"].to_numpy()
    traits = _draw_traits(rng, df, season_jitter=0.30)

    df["season"] = season
    df["age"] = (df["base_age"] + season_index).round(1)
    df["height_cm"] = np.round(
        np.array([BASE_HEIGHT[g] for g in groups])
        + 4.2 * traits["aerial"].to_numpy()
        + rng.normal(0, 3.4, n)
    ).astype(int)

    # ---- Playing time -------------------------------------------------
    age = df["age"].to_numpy()
    # Availability peaks in the late twenties and tapers at both ends.
    age_curve = -0.055 * (age - 27.5) ** 2 / 2.0
    play_index = 1.05 * df["ability"].to_numpy() + age_curve + rng.normal(0, 0.55, n)
    share = _sigmoid(play_index)
    minutes = np.clip(
        np.round(3300 * share * rng.uniform(0.55, 1.15, n)), 0, 3420
    )
    df["minutes"] = minutes.astype(int)
    minutes_per_app = rng.uniform(52, 86, n)
    matches = np.clip(np.round(minutes / minutes_per_app), 0, 38).astype(int)
    df["matches"] = matches
    starts = np.minimum(matches, np.round(minutes / 90 * rng.uniform(0.82, 1.02, n))).astype(int)
    df["starts"] = np.clip(starts, 0, matches)

    exposure = minutes / 90.0  # per-90 exposure used by every Poisson draw

    # ---- Possession context -------------------------------------------
    # More team possession => more passes, fewer defensive actions.
    poss = df["team_possession"].to_numpy()
    poss_pass_mult = (poss / 50.0) ** 0.85
    poss_def_mult = ((100 - poss) / 50.0) ** 0.75

    def poisson(stat: str, extra_mult: np.ndarray | float = 1.0) -> np.ndarray:
        rate = _rate_matrix(stat, groups, traits) * extra_mult
        return rng.poisson(np.clip(rate, 0, None) * exposure)

    # ---- Passing -------------------------------------------------------
    df["passes_attempted"] = poisson("passes_attempted", poss_pass_mult)
    p_pass = _prob_vector("pass_pct", groups, traits)
    df["passes_completed"] = rng.binomial(df["passes_attempted"], p_pass)
    df["progressive_passes"] = np.minimum(
        df["passes_attempted"], poisson("progressive_passes", poss_pass_mult)
    )
    df["passes_into_final_third"] = np.minimum(
        df["passes_attempted"], poisson("passes_into_final_third", poss_pass_mult)
    )
    df["passes_into_pen_area"] = poisson("passes_into_pen_area", poss_pass_mult)
    df["through_balls"] = poisson("through_balls")
    df["crosses"] = poisson("crosses")
    df["switches"] = poisson("switches", poss_pass_mult)
    df["long_passes_attempted"] = np.minimum(
        df["passes_attempted"], poisson("long_passes_attempted")
    )
    df["long_passes_completed"] = rng.binomial(
        df["long_passes_attempted"], _prob_vector("long_pass_pct", groups, traits)
    )
    # Passing under pressure: a share of passes are contested, and completion
    # drops on those. Press-resistant players lose less.
    pressure_share = np.clip(
        0.24 * np.exp(-0.20 * traits["pass_volume"].to_numpy() + 0.15 * traits["carrying"].to_numpy())
        * (50.0 / np.clip(poss, 25, 75)),
        0.05, 0.55,
    )
    df["passes_under_pressure"] = rng.binomial(df["passes_attempted"], pressure_share)
    p_pressed = _sigmoid(
        _logit(np.clip(p_pass, 0.05, 0.95)) - 0.75 + 0.45 * traits["pass_accuracy"].to_numpy()
    )
    df["passes_completed_under_pressure"] = np.minimum(
        rng.binomial(df["passes_under_pressure"], p_pressed),
        df["passes_completed"],
    )
    df["progressive_receptions"] = poisson("progressive_receptions", poss_pass_mult)
    df["touches"] = df["passes_attempted"] + poisson("touches", 0.45 * poss_pass_mult)

    # ---- Carrying and dribbling ---------------------------------------
    df["progressive_carries"] = poisson("progressive_carries", poss_pass_mult)
    df["carries_into_final_third"] = rng.binomial(
        df["progressive_carries"], np.clip(rng.normal(0.45, 0.06, n), 0.15, 0.8)
    )
    df["carries_into_pen_area"] = poisson("carries_into_pen_area", poss_pass_mult)
    df["dribbles_attempted"] = poisson("dribbles_attempted")
    df["dribbles_completed"] = rng.binomial(
        df["dribbles_attempted"], _prob_vector("dribble_success_pct", groups, traits)
    )
    df["miscontrols"] = poisson("miscontrols")
    df["dispossessed"] = poisson("dispossessed")
    df["touches_att_pen"] = poisson("touches_att_pen", poss_pass_mult)

    # ---- Shooting and finishing ---------------------------------------
    df["shots"] = poisson("shots")
    xg_per_shot = np.clip(
        np.array([BASE_XG_PER_SHOT[g] for g in groups])
        * np.exp(0.28 * traits["box_presence"].to_numpy() - 0.10 * traits["shooting"].to_numpy()),
        0.02,
        0.35,
    )
    # xG accumulates shot by shot, so it is a sum of gamma-distributed values.
    shots = df["shots"].to_numpy()
    df["npxg"] = np.round(
        np.where(shots > 0, rng.gamma(np.maximum(shots, 1e-9) * 2.0, xg_per_shot / 2.0), 0.0), 2
    )
    conversion = np.clip(xg_per_shot * np.exp(0.22 * traits["finishing"].to_numpy()), 0.01, 0.6)
    df["np_goals"] = rng.binomial(shots, conversion)
    df["shots_on_target"] = rng.binomial(shots, _prob_vector("shot_accuracy_pct", groups, traits))
    pen_rate = np.exp(0.35 * traits["finishing"].to_numpy() + 0.35 * traits["box_presence"].to_numpy())
    pen_base = np.where(np.isin(groups, ["FW", "AM", "W"]), 0.055, 0.012)
    df["pens_taken"] = rng.poisson(pen_base * pen_rate * exposure)
    df["pens_scored"] = rng.binomial(df["pens_taken"], 0.78)
    df["goals"] = df["np_goals"] + df["pens_scored"]
    df["xg"] = (df["npxg"] + 0.79 * df["pens_taken"]).round(2)
    # Set-piece share of chances: taller, more aerial players feed on them.
    set_piece_share = np.clip(
        0.16 + 0.06 * traits["aerial"].to_numpy() - 0.04 * traits["dribbling"].to_numpy(), 0.02, 0.5
    )
    df["npxg_set_piece"] = (df["npxg"] * set_piece_share).round(2)
    df["npxg_open_play"] = (df["npxg"] - df["npxg_set_piece"]).round(2)

    # ---- Creation ------------------------------------------------------
    df["key_passes"] = poisson("key_passes", poss_pass_mult)
    kp = df["key_passes"].to_numpy()
    xa_per_kp = BASE_XA_PER_KEY_PASS * np.exp(0.20 * traits["creation"].to_numpy())
    df["xa"] = np.round(
        np.where(kp > 0, rng.gamma(np.maximum(kp, 1e-9) * 2.5, xa_per_kp / 2.5), 0.0), 2
    )
    # Assists depend on team-mates finishing the chances the player created.
    df["assists"] = rng.poisson(np.clip(df["xa"].to_numpy(), 0, None) * rng.lognormal(0, 0.22, n))
    df["sca"] = kp + rng.poisson(
        np.clip(
            (0.8 + 0.35 * df["progressive_carries"].to_numpy() / np.maximum(exposure, 1e-9))
            * exposure,
            0,
            None,
        )
    )
    df["gca"] = rng.binomial(df["sca"], 0.115)

    # ---- Defending -----------------------------------------------------
    df["tackles"] = poisson("tackles", poss_def_mult)
    df["tackles_won"] = rng.binomial(df["tackles"], _prob_vector("tackle_win_pct", groups, traits))
    df["interceptions"] = poisson("interceptions", poss_def_mult)
    df["blocks"] = poisson("blocks", poss_def_mult)
    df["clearances"] = poisson("clearances", poss_def_mult)
    df["ball_recoveries"] = poisson("ball_recoveries")
    df["pressures"] = poisson("pressures", poss_def_mult)
    df["pressures_successful"] = rng.binomial(
        df["pressures"], _prob_vector("pressure_success_pct", groups, traits)
    )
    aerials_total = poisson("aerials_contested")
    df["aerials_won"] = rng.binomial(aerials_total, _prob_vector("aerial_win_pct", groups, traits))
    df["aerials_lost"] = aerials_total - df["aerials_won"]
    df["fouls_committed"] = poisson("fouls_committed")
    df["errors"] = poisson("errors")

    # ---- Goalkeeping ---------------------------------------------------
    is_gk = groups == "GK"
    shots_faced_rate = np.where(
        is_gk, BASE_RATES["gk_shots_on_target_against"]["GK"] * np.exp(-0.30 * df["team_strength"]), 0.0
    )
    sota = rng.poisson(shots_faced_rate * exposure)
    save_p = np.where(is_gk, _prob_vector("gk_save_pct", groups, traits), 0.0)
    saves = rng.binomial(sota, np.clip(save_p, 0, 1))
    df["gk_shots_on_target_against"] = np.where(is_gk, sota, np.nan)
    df["gk_saves"] = np.where(is_gk, saves, np.nan)
    df["gk_goals_against"] = np.where(is_gk, sota - saves, np.nan)
    df["gk_psxg"] = np.where(
        is_gk, np.round(rng.gamma(np.maximum(sota, 1e-9) * 3.0, 0.345 / 3.0), 2), np.nan
    )
    crosses_faced = rng.poisson(np.where(is_gk, BASE_RATES["gk_crosses_faced"]["GK"], 0.0) * exposure)
    df["gk_crosses_faced"] = np.where(is_gk, crosses_faced, np.nan)
    df["gk_crosses_stopped"] = np.where(
        is_gk,
        rng.binomial(crosses_faced, np.clip(_prob_vector("gk_cross_stop_pct", groups, traits), 0, 1)),
        np.nan,
    )
    df["gk_def_actions_outside_box"] = np.where(is_gk, poisson("gk_def_actions_outside_box"), np.nan)
    launches = rng.poisson(
        np.where(is_gk, _rate_matrix("gk_launches_attempted", groups, traits), 0.0) * exposure
    )
    df["gk_launches_attempted"] = np.where(is_gk, launches, np.nan)
    df["gk_launches_completed"] = np.where(
        is_gk, rng.binomial(launches, np.clip(_prob_vector("gk_launch_pct", groups, traits), 0, 1)), np.nan
    )

    df = df.rename(columns={"role_profile": "true_role_profile"})
    return df.drop(columns=["base_age", "ability", "team_strength"])


def _inject_data_quality_issues(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic mess so the cleaning pipeline has real work to do.

    Real scouting feeds arrive with duplicated rows, missing optional columns
    and the occasional impossible value. The cleaning step in
    src/data_processing.py reports on exactly these.
    """
    df = df.copy()

    # 1. Duplicated rows (a player-season ingested twice).
    dupes = df.sample(frac=0.004, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Missing optional attributes.
    for column, frac in [("height_cm", 0.02), ("xa", 0.008), ("gk_psxg", 0.05)]:
        idx = df.sample(frac=frac, random_state=int(rng.integers(1e6))).index
        df.loc[idx, column] = np.nan

    # 3. Impossible values that must be caught, not silently modelled.
    bad = df.sample(n=12, random_state=int(rng.integers(1e6))).index
    df.loc[bad[:4], "minutes"] = df.loc[bad[:4], "minutes"] + 4000
    df.loc[bad[4:8], "shots"] = -1
    df.loc[bad[8:], "height_cm"] = 12

    return df.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)


def generate_dataset(seed: int = 7, seasons: list[str] | None = None, messy: bool = True) -> pd.DataFrame:
    """Generate the full multi-season player-season table."""
    seasons = seasons or list(SEASONS)
    rng = np.random.default_rng(seed)
    teams = build_teams(rng)
    pool = build_player_pool(rng, teams)

    frames = []
    for i, season in enumerate(seasons):
        if i > 0:
            # A slice of the league moves clubs between seasons.
            movers = pool.sample(frac=0.08, random_state=seed + i).index
            targets = teams.sample(n=len(movers), replace=True, random_state=seed + i)
            pool.loc[movers, "team"] = targets["team"].to_numpy()
            pool.loc[movers, "league"] = targets["league"].to_numpy()
        frames.append(simulate_season(rng, pool, teams, season, season_index=i))

    data = pd.concat(frames, ignore_index=True)
    data = data[data["minutes"] > 0].reset_index(drop=True)
    if messy:
        data = _inject_data_quality_issues(rng, data)
    return data
