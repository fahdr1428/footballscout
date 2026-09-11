# Model validation report

Pool: **20,357 player-seasons**, minimum **900 minutes**, seasons 2014-15, 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Not run. This source does not record positions specific enough to test - it publishes broad buckets rather than 'Left-Back' and 'Second Striker'.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEF | 9137 | 12 | 3 | 5 | 0.306 | 61601.3 | 524 |
| MID | 4297 | 14 | 4 | 6 | 0.232 | 32342.0 | 396 |
| FWD | 5390 | 13 | 3 | 5 | 0.244 | 41835.4 | 1179 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| DEF | 0.526 | 0.177 | 0.703 |
| MID | 0.503 | 0.137 | 0.64 |
| FWD | 0.482 | 0.233 | 0.715 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 33574 | 9136 | 0.017 | 0.001 | 15.1 | 1499 |
| MID | 13146 | 4296 | 0.033 | 0.002 | 14.3 | 659 |
| FWD | 17724 | 5389 | 0.027 | 0.002 | 14.7 | 823 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| DEF | 0.001 | 0.001 | 1.9 |
| MID | 0.003 | 0.001 | 4.4 |
| FWD | 0.001 | 0.001 | 1.0 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 9,487 of 13,030 player-seasons (73%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 946 | 11.95 | 8000000.0 | 0.8 | 0.342 |
| 2 | 951 | 30.4 | 12000000.0 | 0.71 | 0.251 |
| 3 | 948 | 35.6 | 10000000.0 | 0.8 | 0.328 |
| 4 | 956 | 39.4 | 10000000.0 | 0.8 | 0.351 |
| 5 | 924 | 42.5 | 8000000.0 | 0.83 | 0.356 |
| 6 | 952 | 45.5 | 8000000.0 | 0.92 | 0.404 |
| 7 | 966 | 48.7 | 7000000.0 | 1.0 | 0.466 |
| 8 | 945 | 52.2 | 5000000.0 | 1.0 | 0.497 |
| 9 | 951 | 56.5 | 4000000.0 | 1.33 | 0.593 |
| 10 | 948 | 64.05 | 2500000.0 | 1.8 | 0.703 |

Median value change runs from **x0.8** in the bottom decile to **x1.8** in the top, and the share of players whose value rose climbs from 34% to 70%. Rank correlation of score against growth: **0.297**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (20 cells, 9,487 players, minimum 40 each) gives a correlation of **0.112** - roughly half the headline figure, positive in 18 of 20 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**MID** - even share would be 0.071 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Yellow cards per 90 | 0.1138 | 1.59 |
| Non-penalty xG per shot | 0.0839 | 1.17 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0834 | 1.17 |
| Non-penalty goals - xG per 90 | 0.0806 | 1.13 |
| Key passes per 90 | 0.0799 | 1.12 |
| Shots per 90 | 0.0786 | 1.1 |

**DEF** - even share would be 0.083 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Yellow cards per 90 | 0.1466 | 1.76 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.1053 | 1.26 |
| Key passes per 90 | 0.0928 | 1.11 |
| Assists per 90 | 0.0847 | 1.02 |
| xGChain (possessions ending in a shot) per 90 | 0.0837 | 1.0 |
| Shots per 90 | 0.0786 | 0.94 |

**FWD** - even share would be 0.077 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Non-penalty goals - xG per 90 | 0.0996 | 1.3 |
| Non-penalty xG per shot | 0.0903 | 1.17 |
| Key passes per 90 | 0.089 | 1.16 |
| Assists per 90 | 0.0887 | 1.15 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0858 | 1.12 |
| Shots per 90 | 0.0856 | 1.11 |


## 4. Sensitivity of the similarity rankings

### MID - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.535 | 0.6 |
| Non-penalty xG per shot | 0.646 | 0.677 |
| xA (expected assists) per 90 | 0.729 | 0.797 |
| Non-penalty xG per 90 | 0.872 | 0.939 |
| xG per 90 | 0.874 | 0.935 |
| Goals per 90 | 0.875 | 0.931 |
| Non-penalty goals per 90 | 0.884 | 0.927 |
| Expected goal involvements per 90 | 0.895 | 0.952 |

### MID - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.571 | 0.56 |
| Build-up Involvement | 0.639 | 0.73 |
| Goal Threat | 0.706 | 0.745 |

### DEF - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.493 | 0.597 |
| Key passes per 90 | 0.533 | 0.596 |
| xA (expected assists) per 90 | 0.71 | 0.74 |
| Non-penalty goals per 90 | 0.761 | 0.844 |
| Goals per 90 | 0.766 | 0.872 |
| Non-penalty xG per 90 | 0.769 | 0.818 |
| xG per 90 | 0.803 | 0.847 |
| Expected goal involvements per 90 | 0.873 | 0.887 |

### DEF - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.585 | 0.567 |
| Goal Threat | 0.59 | 0.63 |
| Build-up Involvement | 0.66 | 0.771 |

### FWD - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.623 | 0.713 |
| Shots per 90 | 0.639 | 0.769 |
| xA (expected assists) per 90 | 0.659 | 0.753 |
| Non-penalty goals - xG per 90 | 0.66 | 0.743 |
| Goals per 90 | 0.873 | 0.898 |
| Non-penalty xG per 90 | 0.873 | 0.948 |
| xG per 90 | 0.875 | 0.92 |
| Non-penalty goals per 90 | 0.881 | 0.937 |

### FWD - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.564 | 0.633 |
| Build-up Involvement | 0.638 | 0.703 |
| Goal Threat | 0.702 | 0.705 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**MID** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Goals per 90 | Non-penalty goals per 90 | 0.944 |
| xG per 90 | Non-penalty xG per 90 | 0.937 |
| xG per 90 | Expected goal involvements per 90 | 0.882 |
| xA (expected assists) per 90 | Expected goal involvements per 90 | 0.861 |
| xA (expected assists) per 90 | Key passes per 90 | 0.861 |

**DEF** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Goals per 90 | Non-penalty goals per 90 | 0.975 |
| xG per 90 | Non-penalty xG per 90 | 0.972 |
| xGChain (possessions ending in a shot) per 90 | xGBuildup (xGChain excluding shots and key passes) per 90 | 0.885 |
| xA (expected assists) per 90 | Key passes per 90 | 0.879 |
| xA (expected assists) per 90 | Expected goal involvements per 90 | 0.856 |

**FWD** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| xG per 90 | Non-penalty xG per 90 | 0.959 |
| Goals per 90 | Non-penalty goals per 90 | 0.958 |
| xG per 90 | Expected goal involvements per 90 | 0.909 |
| Non-penalty xG per 90 | Expected goal involvements per 90 | 0.867 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.