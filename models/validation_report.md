# Model validation report

Pool: **2,626 player-seasons**, minimum **900 minutes**, seasons 2015-16, 2018, 2018-19, 2019-20, 2020-21, 2021-22, 2023, 2023-24.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 211 | 11 | 3 | 6 | 0.216 | 1504.7 | 50 |
| CB | 522 | 19 | 3 | 5 | 0.131 | 7486.3 | 125 |
| FB | 502 | 20 | 5 | 5 | 0.099 | 6679.3 | 44 |
| DM | 338 | 20 | 3 | 6 | 0.105 | 5222.9 | 87 |
| CM | 225 | 20 | 3 | 5 | 0.178 | 3147.0 | 37 |
| AM | 119 | 20 | 3 | 3 | 0.176 | 1660.2 | 23 |
| W | 410 | 22 | 4 | 5 | 0.143 | 5918.6 | 37 |
| FW | 299 | 21 | 6 | 6 | 0.111 | 3758.5 | 7 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.382 | 0.188 | 0.57 |
| CB | 0.247 | 0.161 | 0.408 |
| FB | 0.306 | 0.134 | 0.44 |
| DM | 0.243 | 0.129 | 0.372 |
| CM | 0.345 | 0.153 | 0.497 |
| AM | 0.301 | 0.203 | 0.504 |
| W | 0.333 | 0.164 | 0.497 |
| FW | 0.272 | 0.183 | 0.455 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 76 | 210 | 0.237 | 0.048 | 5.0 | 19 |
| CB | 100 | 521 | 0.47 | 0.019 | 24.5 | 13 |
| FB | 76 | 501 | 0.263 | 0.02 | 13.2 | 46 |
| DM | 24 | 337 | 0.708 | 0.03 | 23.9 | 4 |
| CM | 24 | 224 | 0.458 | 0.045 | 10.3 | 17 |
| AM | 14 | 118 | 0.857 | 0.085 | 10.1 | 4 |
| W | 70 | 409 | 0.329 | 0.024 | 13.4 | 18 |
| FW | 30 | 298 | 0.333 | 0.034 | 9.9 | 44 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.012 | 0.001 | 10.5 |
| CB | 0.04 | 0.004 | 10.7 |
| FB | 0.023 | 0.004 | 6.0 |
| DM | 0.017 | 0.004 | 4.2 |
| CM | 0.033 | 0.006 | 5.5 |
| AM | 0.015 | 0.003 | 5.1 |
| W | 0.017 | 0.004 | 4.0 |
| FW | 0.015 | 0.004 | 3.9 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Dribble success % | 0.0556 | 1.22 |
| Pass completion % | 0.0548 | 1.2 |
| Pressures per 90 | 0.0538 | 1.18 |
| Progressive passes received per 90 | 0.0527 | 1.16 |
| Tackles per 90 | 0.0515 | 1.13 |
| Non-penalty xG per shot | 0.0501 | 1.1 |

**CB** - even share would be 0.053 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Interceptions per 90 | 0.0611 | 1.16 |
| Fouls committed per 90 | 0.0596 | 1.13 |
| Tackle success % | 0.0593 | 1.13 |
| Tackles per 90 | 0.0585 | 1.11 |
| Errors leading to shot per 90 | 0.0577 | 1.1 |
| Aerial duel success % | 0.0557 | 1.06 |

**DM** - even share would be 0.050 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Fouls committed per 90 | 0.0576 | 1.15 |
| Tackle success % | 0.0569 | 1.14 |
| Pressure success % | 0.0556 | 1.11 |
| Aerial duel success % | 0.055 | 1.1 |
| Pressures per 90 | 0.0544 | 1.09 |
| Ball recoveries per 90 | 0.0542 | 1.08 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.746 | 0.798 |
| Shots per 90 | 0.811 | 0.861 |
| Assists per 90 | 0.858 | 0.848 |
| Non-penalty goals per 90 | 0.859 | 0.897 |
| Key passes per 90 | 0.866 | 0.925 |
| Non-penalty xG per 90 | 0.873 | 0.924 |
| xA (expected assists) per 90 | 0.883 | 0.91 |
| Touches in opposition box per 90 | 0.887 | 0.915 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Finishing | 0.587 | 0.609 |
| Ball Progression | 0.601 | 0.621 |
| Dribbling | 0.605 | 0.579 |
| Defending | 0.638 | 0.679 |
| Chance Creation | 0.64 | 0.684 |
| Box Threat | 0.658 | 0.759 |
| Passing | 0.697 | 0.762 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.651 | 0.69 |
| Blocks per 90 | 0.702 | 0.792 |
| Tackles per 90 | 0.717 | 0.798 |
| Clearances per 90 | 0.723 | 0.736 |
| Ball recoveries per 90 | 0.731 | 0.761 |
| Interceptions per 90 | 0.738 | 0.827 |
| Pressures per 90 | 0.764 | 0.811 |
| Aerial duels won per 90 | 0.838 | 0.895 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.524 | 0.495 |
| Aerial | 0.561 | 0.652 |
| Ball Progression | 0.615 | 0.643 |
| Passing | 0.643 | 0.701 |
| Dribbling | 0.724 | 0.782 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Interceptions per 90 | 0.727 | 0.781 |
| Blocks per 90 | 0.742 | 0.795 |
| Ball recoveries per 90 | 0.743 | 0.755 |
| Tackle success % | 0.75 | 0.794 |
| Pressure success % | 0.755 | 0.801 |
| Tackles per 90 | 0.765 | 0.802 |
| Pressures per 90 | 0.771 | 0.829 |
| Clearances per 90 | 0.801 | 0.83 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.506 | 0.485 |
| Passing | 0.62 | 0.653 |
| Ball Progression | 0.646 | 0.64 |
| Aerial | 0.671 | 0.716 |
| Dribbling | 0.805 | 0.852 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
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