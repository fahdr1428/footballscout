# Model validation report

Pool: **339 player-seasons**, minimum **900 minutes**, seasons 2025-26.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 24 | 8 | 2 | 2 | 0.38 | 121.3 | 7 |
| DEF | 128 | 15 | 3 | 6 | 0.18 | 1331.9 | 16 |
| MID | 153 | 16 | 3 | 6 | 0.163 | 1510.6 | 36 |
| FWD | 34 | 13 | 2 | 2 | 0.219 | 299.7 | 16 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.569 | 0.262 | 0.831 |
| DEF | 0.281 | 0.202 | 0.483 |
| MID | 0.361 | 0.203 | 0.564 |
| FWD | 0.374 | 0.29 | 0.664 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.008 | 0.014 | 0.6 |
| DEF | 0.074 | 0.044 | 1.7 |
| MID | 0.064 | 0.046 | 1.4 |
| FWD | 0.026 | 0.034 | 0.8 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

## 3. Is the model dominated by a few metrics?

**MID** - even share would be 0.062 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Share of appearances that were starts % | 0.0822 | 1.32 |
| Ball recoveries per 90 | 0.0766 | 1.23 |
| Yellow cards per 90 | 0.0739 | 1.18 |
| Expected goals conceded (team, while on pitch) per 90 | 0.0726 | 1.16 |
| Tackles per 90 | 0.0703 | 1.12 |
| Goals per 90 | 0.0676 | 1.08 |

**DEF** - even share would be 0.067 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Ball recoveries per 90 | 0.0778 | 1.17 |
| Tackles per 90 | 0.0753 | 1.13 |
| Yellow cards per 90 | 0.0745 | 1.12 |
| Share of appearances that were starts % | 0.0717 | 1.08 |
| Influence (Opta index) per 90 | 0.0699 | 1.05 |
| Defensive contribution actions per 90 | 0.067 | 1.01 |

**FWD** - even share would be 0.077 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Share of appearances that were starts % | 0.1061 | 1.38 |
| Assists per 90 | 0.0998 | 1.3 |
| Tackles per 90 | 0.0936 | 1.22 |
| Defensive contribution actions per 90 | 0.0867 | 1.13 |
| Threat (Opta index) per 90 | 0.0847 | 1.1 |
| Bonus points system score per 90 | 0.0792 | 1.03 |


## 4. Sensitivity of the similarity rankings

### MID - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.751 | 0.803 |
| Creativity (Opta index) per 90 | 0.824 | 0.87 |
| Goals per 90 | 0.83 | 0.877 |
| Influence (Opta index) per 90 | 0.835 | 0.884 |
| xA (expected assists) per 90 | 0.847 | 0.902 |
| xG per 90 | 0.849 | 0.871 |
| Threat (Opta index) per 90 | 0.86 | 0.88 |
| Non-penalty xG per 90 | 1.0 | 1.0 |

### MID - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.653 | 0.664 |
| Defensive Work | 0.658 | 0.715 |
| Availability | 0.684 | 0.754 |
| Goal Threat | 0.697 | 0.78 |
| Defensive Solidity | 0.724 | 0.753 |
| Overall Rating | 0.776 | 0.852 |
| Build-up Involvement | 0.8 | 0.865 |

### DEF - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball recoveries per 90 | 0.723 | 0.706 |
| Tackles per 90 | 0.783 | 0.767 |
| xG per 90 | 0.8 | 0.855 |
| Goals per 90 | 0.801 | 0.838 |
| Expected goals conceded (team, while on pitch) per 90 | 0.829 | 0.849 |
| Defensive contribution actions per 90 | 0.842 | 0.879 |
| Clearances, blocks and interceptions per 90 | 0.852 | 0.897 |
| xA (expected assists) per 90 | 0.853 | 0.881 |

### DEF - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.632 | 0.68 |
| Defensive Work | 0.632 | 0.642 |
| Goal Threat | 0.675 | 0.686 |
| Availability | 0.728 | 0.782 |
| Overall Rating | 0.771 | 0.843 |
| Defensive Solidity | 0.779 | 0.812 |
| Build-up Involvement | 0.798 | 0.846 |

### FWD - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.815 | 0.683 |
| Threat (Opta index) per 90 | 0.859 | 0.914 |
| xG per 90 | 0.865 | 0.904 |
| Influence (Opta index) per 90 | 0.873 | 0.931 |
| Goals per 90 | 0.875 | 0.934 |
| xA (expected assists) per 90 | 0.897 | 0.861 |
| Creativity (Opta index) per 90 | 0.901 | 0.894 |
| Non-penalty xG per 90 | 1.0 | 1.0 |

### FWD - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.675 | 0.672 |
| Defensive Work | 0.702 | 0.667 |
| Goal Threat | 0.772 | 0.734 |
| Availability | 0.78 | 0.76 |
| Overall Rating | 0.844 | 0.891 |
| Build-up Involvement | 0.86 | 0.897 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**MID** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| xA (expected assists) per 90 | Creativity (Opta index) per 90 | 0.882 |
| Influence (Opta index) per 90 | Bonus points system score per 90 | 0.858 |

**DEF** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Clearances, blocks and interceptions per 90 | Defensive contribution actions per 90 | 0.954 |
| xA (expected assists) per 90 | Creativity (Opta index) per 90 | 0.863 |

**FWD** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Influence (Opta index) per 90 | Bonus points system score per 90 | 0.936 |
| Goals per 90 | Influence (Opta index) per 90 | 0.92 |
| Goals per 90 | Bonus points system score per 90 | 0.886 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.