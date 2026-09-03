# Model validation report

Pool: **2,626 player-seasons**, minimum **900 minutes**, seasons 2015-16, 2018, 2018-19, 2019-20, 2020-21, 2021-22, 2023, 2023-24.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 211 | 13 | 3 | 6 | 0.179 | 1876.2 | 57 |
| CB | 522 | 20 | 3 | 5 | 0.126 | 7901.7 | 144 |
| FB | 502 | 21 | 6 | 6 | 0.096 | 6800.8 | 40 |
| DM | 338 | 22 | 4 | 6 | 0.094 | 5375.5 | 57 |
| CM | 225 | 22 | 3 | 5 | 0.171 | 3524.4 | 37 |
| AM | 119 | 22 | 3 | 6 | 0.179 | 1823.8 | 23 |
| W | 410 | 23 | 3 | 6 | 0.167 | 6761.6 | 92 |
| FW | 299 | 22 | 5 | 6 | 0.115 | 4150.6 | 23 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.329 | 0.2 | 0.53 |
| CB | 0.255 | 0.153 | 0.408 |
| FB | 0.299 | 0.131 | 0.43 |
| DM | 0.259 | 0.122 | 0.381 |
| CM | 0.319 | 0.166 | 0.485 |
| AM | 0.285 | 0.222 | 0.506 |
| W | 0.332 | 0.179 | 0.511 |
| FW | 0.276 | 0.192 | 0.468 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 76 | 210 | 0.25 | 0.048 | 5.2 | 24 |
| CB | 100 | 521 | 0.46 | 0.019 | 24.0 | 13 |
| FB | 76 | 501 | 0.263 | 0.02 | 13.2 | 54 |
| DM | 24 | 337 | 0.667 | 0.03 | 22.5 | 5 |
| CM | 24 | 224 | 0.417 | 0.045 | 9.3 | 16 |
| AM | 14 | 118 | 0.786 | 0.085 | 9.3 | 5 |
| W | 70 | 409 | 0.3 | 0.024 | 12.3 | 19 |
| FW | 30 | 298 | 0.333 | 0.034 | 9.9 | 42 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.006 | 0.001 | 5.1 |
| CB | 0.031 | 0.004 | 8.2 |
| FB | 0.021 | 0.004 | 5.6 |
| DM | 0.016 | 0.004 | 4.0 |
| CM | 0.027 | 0.006 | 4.6 |
| AM | 0.009 | 0.003 | 3.1 |
| W | 0.014 | 0.004 | 3.2 |
| FW | 0.013 | 0.004 | 3.3 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.043 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Dribble success % | 0.054 | 1.24 |
| Pass completion % | 0.0531 | 1.22 |
| Pressures per 90 | 0.0522 | 1.2 |
| Progressive passes received per 90 | 0.0509 | 1.17 |
| Tackles per 90 | 0.0499 | 1.15 |
| Non-penalty xG per shot | 0.0474 | 1.09 |

**CB** - even share would be 0.050 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Interceptions per 90 | 0.0581 | 1.16 |
| Fouls committed per 90 | 0.0567 | 1.13 |
| Tackle success % | 0.0562 | 1.12 |
| Tackles per 90 | 0.056 | 1.12 |
| Errors leading to shot per 90 | 0.0553 | 1.11 |
| Aerial duel success % | 0.053 | 1.06 |

**DM** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Fouls committed per 90 | 0.0527 | 1.16 |
| Tackle success % | 0.0521 | 1.15 |
| Pressure success % | 0.051 | 1.12 |
| Aerial duel success % | 0.0503 | 1.11 |
| Pressures per 90 | 0.0496 | 1.09 |
| Ball recoveries per 90 | 0.0494 | 1.09 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.748 | 0.841 |
| Shots per 90 | 0.798 | 0.854 |
| Assists per 90 | 0.833 | 0.854 |
| xA (expected assists) per 90 | 0.861 | 0.915 |
| Non-penalty goals per 90 | 0.867 | 0.89 |
| Non-penalty xG from open play per 90 | 0.879 | 0.937 |
| Non-penalty xG per 90 | 0.887 | 0.947 |
| Touches in opposition box per 90 | 0.905 | 0.914 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Finishing | 0.586 | 0.587 |
| Ball Progression | 0.59 | 0.6 |
| Dribbling | 0.599 | 0.582 |
| Chance Creation | 0.634 | 0.683 |
| Defending | 0.653 | 0.679 |
| Box Threat | 0.67 | 0.726 |
| Passing | 0.701 | 0.733 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.66 | 0.68 |
| Blocks per 90 | 0.719 | 0.796 |
| Interceptions per 90 | 0.738 | 0.804 |
| Tackles per 90 | 0.74 | 0.791 |
| Ball recoveries per 90 | 0.74 | 0.782 |
| Clearances per 90 | 0.745 | 0.774 |
| Pressures per 90 | 0.775 | 0.822 |
| Pass completion under pressure % | 0.799 | 0.811 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.525 | 0.498 |
| Aerial | 0.582 | 0.661 |
| Passing | 0.594 | 0.696 |
| Ball Progression | 0.615 | 0.676 |
| Dribbling | 0.741 | 0.799 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Interceptions per 90 | 0.725 | 0.796 |
| Tackle success % | 0.739 | 0.769 |
| Ball recoveries per 90 | 0.75 | 0.778 |
| Blocks per 90 | 0.753 | 0.79 |
| Tackles per 90 | 0.764 | 0.812 |
| Share of passes made under pressure % | 0.785 | 0.861 |
| Clearances per 90 | 0.816 | 0.855 |
| Pass completion under pressure % | 0.836 | 0.892 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.513 | 0.549 |
| Passing | 0.589 | 0.629 |
| Ball Progression | 0.643 | 0.679 |
| Aerial | 0.655 | 0.727 |
| Dribbling | 0.795 | 0.881 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Non-penalty xG from open play per 90 | Non-penalty xG per 90 | 0.966 |
| Key passes per 90 | Shot-creating actions per 90 | 0.93 |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.914 |
| Progressive carries per 90 | Carries into final third per 90 | 0.904 |
| xA (expected assists) per 90 | Key passes per 90 | 0.871 |
| Key passes per 90 | Passes into penalty area per 90 | 0.862 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.965 |
| Pass completion % | Long-pass completion % | 0.875 |
| Progressive passes per 90 | Passes into final third per 90 | 0.851 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Long passes attempted per 90 | Switches of play per 90 | 0.896 |
| Progressive passes per 90 | Passes into final third per 90 | 0.875 |
| Passes attempted per 90 | Passes into final third per 90 | 0.868 |
| Pass completion % | Long-pass completion % | 0.852 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.