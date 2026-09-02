# Model validation report

Pool: **5,504 player-seasons**, minimum **900 minutes**, seasons 2023-24, 2024-25.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster | adjusted_rand_vs_true_role | cluster_purity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 554 | 12 | 5 | 5 | 0.127 | 4338.9 | 75 | 0.127 | 0.493 |
| CB | 1090 | 19 | 5 | 4 | 0.12 | 13214.9 | 166 | 0.248 | 0.605 |
| FB | 813 | 20 | 3 | 5 | 0.144 | 11916.7 | 207 | 0.166 | 0.502 |
| DM | 543 | 20 | 4 | 6 | 0.124 | 7413.2 | 91 | 0.304 | 0.659 |
| CM | 798 | 20 | 4 | 5 | 0.151 | 11024.3 | 109 | 0.21 | 0.579 |
| AM | 552 | 20 | 6 | 5 | 0.149 | 6139.0 | 25 | 0.258 | 0.687 |
| W | 574 | 22 | 4 | 4 | 0.204 | 7789.1 | 91 | 0.28 | 0.66 |
| FW | 580 | 21 | 3 | 6 | 0.169 | 8800.3 | 112 | 0.154 | 0.495 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.236 | 0.172 | 0.408 |
| CB | 0.279 | 0.153 | 0.431 |
| FB | 0.269 | 0.173 | 0.442 |
| DM | 0.294 | 0.158 | 0.452 |
| CM | 0.218 | 0.206 | 0.424 |
| AM | 0.291 | 0.193 | 0.483 |
| W | 0.251 | 0.218 | 0.469 |
| FW | 0.236 | 0.21 | 0.446 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2. Does similarity find players who do the same job?

| position_group | roles | top10_same_role | chance_baseline | lift |
| --- | --- | --- | --- | --- |
| GK | 4 | 0.54 | 0.251 | 2.15 |
| CB | 4 | 0.573 | 0.25 | 2.29 |
| FB | 4 | 0.636 | 0.252 | 2.53 |
| DM | 4 | 0.577 | 0.256 | 2.25 |
| CM | 4 | 0.605 | 0.253 | 2.39 |
| AM | 4 | 0.727 | 0.253 | 2.88 |
| W | 4 | 0.707 | 0.252 | 2.81 |
| FW | 4 | 0.697 | 0.256 | 2.73 |

`top10_same_role` is the share of a player's ten nearest neighbours drawn from the same generative role; `chance_baseline` is what random picking would produce given the role mix in that position. A lift above 1 means the model is recovering role, not noise.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pass completion % | 0.0583 | 1.28 |
| Tackles per 90 | 0.0575 | 1.26 |
| Progressive passes per 90 | 0.0563 | 1.24 |
| Pressures per 90 | 0.0554 | 1.22 |
| Dribble success % | 0.0531 | 1.17 |
| Non-penalty xG per shot | 0.0497 | 1.09 |

**CB** - even share would be 0.053 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Errors leading to shot per 90 | 0.0602 | 1.14 |
| Pass completion % | 0.0585 | 1.11 |
| Tackle success % | 0.0582 | 1.11 |
| Progressive carries per 90 | 0.0563 | 1.07 |
| Long-pass completion % | 0.0561 | 1.07 |
| Long passes attempted per 90 | 0.055 | 1.04 |

**DM** - even share would be 0.050 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pass completion % | 0.0565 | 1.13 |
| Aerial duel success % | 0.0562 | 1.12 |
| Tackle success % | 0.0555 | 1.11 |
| Long-pass completion % | 0.0545 | 1.09 |
| Pressure success % | 0.052 | 1.04 |
| Aerial duels won per 90 | 0.0516 | 1.03 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.769 | 0.775 |
| Assists per 90 | 0.809 | 0.854 |
| Shots per 90 | 0.815 | 0.886 |
| Non-penalty goals per 90 | 0.842 | 0.896 |
| Touches in opposition box per 90 | 0.857 | 0.904 |
| Non-penalty xG per 90 | 0.875 | 0.921 |
| Key passes per 90 | 0.878 | 0.911 |
| xA (expected assists) per 90 | 0.9 | 0.946 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Finishing | 0.608 | 0.654 |
| Ball Progression | 0.621 | 0.704 |
| Chance Creation | 0.633 | 0.68 |
| Defending | 0.639 | 0.681 |
| Dribbling | 0.651 | 0.743 |
| Box Threat | 0.677 | 0.729 |
| Passing | 0.69 | 0.755 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.634 | 0.663 |
| Pressures per 90 | 0.693 | 0.719 |
| Blocks per 90 | 0.697 | 0.768 |
| Interceptions per 90 | 0.745 | 0.805 |
| Clearances per 90 | 0.76 | 0.846 |
| Ball recoveries per 90 | 0.803 | 0.797 |
| Aerial duels won per 90 | 0.817 | 0.903 |
| Tackles per 90 | 0.831 | 0.876 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.485 | 0.516 |
| Passing | 0.549 | 0.572 |
| Ball Progression | 0.57 | 0.576 |
| Dribbling | 0.645 | 0.723 |
| Aerial | 0.65 | 0.749 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.733 | 0.769 |
| Blocks per 90 | 0.766 | 0.865 |
| Pressure success % | 0.778 | 0.809 |
| Clearances per 90 | 0.789 | 0.837 |
| Pressures per 90 | 0.789 | 0.824 |
| Interceptions per 90 | 0.824 | 0.861 |
| Ball recoveries per 90 | 0.863 | 0.925 |
| Tackles per 90 | 0.886 | 0.913 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Passing | 0.528 | 0.549 |
| Defending | 0.536 | 0.519 |
| Ball Progression | 0.567 | 0.643 |
| Aerial | 0.669 | 0.673 |
| Dribbling | 0.729 | 0.817 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.99 |
| xA (expected assists) per 90 | Key passes per 90 | 0.977 |
| Progressive carries per 90 | Carries into final third per 90 | 0.95 |
| Non-penalty xG per 90 | Shots per 90 | 0.901 |
| Non-penalty xG per 90 | Touches in opposition box per 90 | 0.89 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.988 |
| Progressive passes per 90 | Passes into final third per 90 | 0.929 |
| Aerial duels won per 90 | Aerial duel success % | 0.885 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.943 |
| Aerial duels won per 90 | Aerial duel success % | 0.862 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.