# Model validation report

Pool: **1,633 player-seasons**, minimum **900 minutes**, seasons 2021-22.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 122 | 15 | 3 | 5 | 0.149 | 1292.9 | 34 |
| CB | 338 | 19 | 4 | 5 | 0.118 | 4451.3 | 68 |
| FB | 286 | 20 | 3 | 6 | 0.134 | 4211.1 | 26 |
| DM | 149 | 21 | 3 | 5 | 0.116 | 2375.6 | 45 |
| CM | 223 | 21 | 4 | 5 | 0.129 | 3062.8 | 25 |
| AM | 112 | 20 | 3 | 6 | 0.137 | 1552.4 | 35 |
| W | 208 | 22 | 4 | 5 | 0.124 | 2965.4 | 24 |
| FW | 195 | 21 | 3 | 6 | 0.124 | 3005.4 | 20 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.287 | 0.212 | 0.498 |
| CB | 0.245 | 0.165 | 0.41 |
| FB | 0.318 | 0.133 | 0.451 |
| DM | 0.248 | 0.161 | 0.409 |
| CM | 0.304 | 0.196 | 0.5 |
| AM | 0.312 | 0.2 | 0.512 |
| W | 0.355 | 0.152 | 0.507 |
| FW | 0.307 | 0.157 | 0.464 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.011 | 0.003 | 3.5 |
| CB | 0.034 | 0.008 | 4.3 |
| FB | 0.026 | 0.008 | 3.3 |
| DM | 0.017 | 0.007 | 2.6 |
| CM | 0.021 | 0.008 | 2.5 |
| AM | 0.017 | 0.007 | 2.5 |
| W | 0.022 | 0.008 | 2.7 |
| FW | 0.009 | 0.007 | 1.3 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pressures per 90 | 0.0585 | 1.29 |
| Pass completion % | 0.0556 | 1.22 |
| Dribble success % | 0.0546 | 1.2 |
| Tackles per 90 | 0.0539 | 1.19 |
| Crosses per 90 | 0.0503 | 1.11 |
| Non-penalty goals per 90 | 0.0488 | 1.07 |

**CB** - even share would be 0.053 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Tackle success % | 0.0611 | 1.16 |
| Aerial duel success % | 0.0585 | 1.11 |
| Fouls committed per 90 | 0.0566 | 1.07 |
| Interceptions per 90 | 0.0564 | 1.07 |
| Errors leading to shot per 90 | 0.0554 | 1.05 |
| Clearances per 90 | 0.0548 | 1.04 |

**DM** - even share would be 0.048 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pressures per 90 | 0.0527 | 1.11 |
| Share of passes made under pressure % | 0.0524 | 1.1 |
| Tackle success % | 0.0516 | 1.08 |
| Blocks per 90 | 0.0516 | 1.08 |
| Tackles per 90 | 0.0511 | 1.07 |
| Interceptions per 90 | 0.0505 | 1.06 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty goals per 90 | 0.795 | 0.844 |
| Shots per 90 | 0.812 | 0.852 |
| Assists per 90 | 0.827 | 0.862 |
| Touches in opposition box per 90 | 0.844 | 0.887 |
| Non-penalty xG per 90 | 0.858 | 0.877 |
| Key passes per 90 | 0.87 | 0.88 |
| xA (expected assists) per 90 | 0.871 | 0.887 |
| Non-penalty xG per shot | 0.896 | 0.946 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Dribbling | 0.598 | 0.597 |
| Chance Creation | 0.61 | 0.624 |
| Finishing | 0.621 | 0.646 |
| Defending | 0.645 | 0.681 |
| Ball Progression | 0.67 | 0.677 |
| Box Threat | 0.68 | 0.715 |
| Passing | 0.727 | 0.796 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.698 | 0.751 |
| Interceptions per 90 | 0.712 | 0.764 |
| Blocks per 90 | 0.731 | 0.738 |
| Clearances per 90 | 0.778 | 0.818 |
| Ball recoveries per 90 | 0.779 | 0.824 |
| Pressures per 90 | 0.781 | 0.863 |
| Tackles per 90 | 0.793 | 0.845 |
| Aerial duels won per 90 | 0.815 | 0.882 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.519 | 0.514 |
| Aerial | 0.613 | 0.662 |
| Passing | 0.628 | 0.714 |
| Ball Progression | 0.688 | 0.748 |
| Dribbling | 0.802 | 0.875 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Blocks per 90 | 0.784 | 0.783 |
| Tackles per 90 | 0.792 | 0.798 |
| Tackle success % | 0.793 | 0.798 |
| Interceptions per 90 | 0.802 | 0.832 |
| Share of passes made under pressure % | 0.803 | 0.799 |
| Ball recoveries per 90 | 0.82 | 0.87 |
| Pressures per 90 | 0.839 | 0.837 |
| Clearances per 90 | 0.851 | 0.867 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.565 | 0.485 |
| Passing | 0.653 | 0.666 |
| Ball Progression | 0.656 | 0.597 |
| Aerial | 0.695 | 0.75 |
| Dribbling | 0.786 | 0.833 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.944 |
| Key passes per 90 | Shot-creating actions per 90 | 0.866 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.926 |
| Pass completion % | Long-pass completion % | 0.881 |
| Progressive passes per 90 | Passes into final third per 90 | 0.852 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Long passes attempted per 90 | Switches of play per 90 | 0.878 |
| Passes attempted per 90 | Passes into final third per 90 | 0.854 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.