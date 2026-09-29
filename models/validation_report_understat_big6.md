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
| DEF | 33574 | 9136 | 0.018 | 0.001 | 16.8 | 1302 |
| MID | 13146 | 4296 | 0.038 | 0.002 | 16.3 | 581 |
| FWD | 17724 | 5389 | 0.031 | 0.002 | 16.8 | 725 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2e. Which metrics describe the player, and which the season?

Each metric's season-to-season correlation for the same player in the same position (900+ minutes in both), measured over the source's whole history with values standardised within each season. Its similarity weight is that correlation squared, so a metric that repeats at 0.8 counts four times as much as one at 0.4, and one that barely repeats at all - mostly the season's noise - hardly counts. Chosen by cross-validation over players: held-out players' own other season landed in their top ten more often under it on every source tested, and it held for players who had changed club in between, so the weights are not simply recognising clubs. Metrics marked (club) - goals conceded, clean sheets, a keeper's saves - describe the team in front of him, so they take the median weight whatever their figure.

| position_group | metrics_measured | median_repeatability | most_repeatable | least_repeatable | top_weight_share |
| --- | --- | --- | --- | --- | --- |
| DEF | 12 of 12 | 0.61 | Key passes per 90 0.82, xGChain (possessions ending in a shot) per 90 0.72, Shots per 90 0.72 | Non-penalty goals per 90 0.31, Goals per 90 0.34, Yellow cards per 90 0.35 | 0.156 |
| MID | 14 of 14 | 0.67 | Shots per 90 0.78, Key passes per 90 0.76, Expected goal involvements per 90 0.76 | Non-penalty goals - xG per 90 0.03, Yellow cards per 90 0.44, Non-penalty goals per 90 0.44 | 0.114 |
| FWD | 13 of 13 | 0.67 | xG per 90 0.72, Key passes per 90 0.71, Shots per 90 0.71 | Non-penalty goals - xG per 90 0.07, Assists per 90 0.36, Non-penalty goals per 90 0.52 | 0.106 |

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| DEF | 0.003 | 0.001 | 3.9 |
| MID | 0.003 | 0.001 | 4.4 |
| FWD | 0.001 | 0.001 | 2.1 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 9,487 of 13,030 player-seasons (73%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 956 | 26.5 | 10250000.0 | 0.7 | 0.236 |
| 2 | 929 | 32.1 | 10000000.0 | 0.75 | 0.279 |
| 3 | 956 | 36.5 | 10000000.0 | 0.83 | 0.348 |
| 4 | 964 | 40.0 | 10000000.0 | 0.83 | 0.367 |
| 5 | 951 | 43.1 | 8000000.0 | 0.83 | 0.359 |
| 6 | 949 | 46.1 | 7500000.0 | 1.0 | 0.436 |
| 7 | 942 | 49.1 | 7000000.0 | 1.0 | 0.462 |
| 8 | 947 | 52.6 | 4500000.0 | 1.08 | 0.506 |
| 9 | 943 | 56.8 | 3500000.0 | 1.31 | 0.592 |
| 10 | 950 | 64.4 | 2500000.0 | 1.8 | 0.711 |

Median value change runs from **x0.7** in the bottom decile to **x1.8** in the top, and the share of players whose value rose climbs from 24% to 71%. Rank correlation of score against growth: **0.324**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (20 cells, 9,481 players, minimum 40 each) gives a correlation of **0.12** - roughly half the headline figure, positive in 18 of 20 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**MID** - even share would be 0.071 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Shots per 90 | 0.1136 | 0.1241 | 1.74 |
| Key passes per 90 | 0.1101 | 0.1197 | 1.68 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0908 | 0.1042 | 1.46 |
| xGChain (possessions ending in a shot) per 90 | 0.1058 | 0.0929 | 1.3 |
| xA (expected assists) per 90 | 0.097 | 0.0817 | 1.14 |
| Expected goal involvements per 90 | 0.1087 | 0.0781 | 1.09 |

**DEF** - even share would be 0.083 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Key passes per 90 | 0.1558 | 0.1573 | 1.89 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.1059 | 0.1295 | 1.55 |
| Shots per 90 | 0.1208 | 0.1171 | 1.41 |
| xGChain (possessions ending in a shot) per 90 | 0.1215 | 0.117 | 1.4 |
| xA (expected assists) per 90 | 0.1188 | 0.0996 | 1.2 |
| Expected goal involvements per 90 | 0.1132 | 0.0745 | 0.89 |

**FWD** - even share would be 0.077 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Key passes per 90 | 0.105 | 0.1238 | 1.61 |
| Shots per 90 | 0.1035 | 0.1149 | 1.49 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0914 | 0.1037 | 1.35 |
| Non-penalty xG per shot | 0.0841 | 0.1017 | 1.32 |
| xG per 90 | 0.1059 | 0.0859 | 1.12 |
| Non-penalty xG per 90 | 0.1032 | 0.084 | 1.09 |


## 4. Sensitivity of the similarity rankings

### MID - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.616 | 0.695 |
| Non-penalty xG per shot | 0.704 | 0.788 |
| xA (expected assists) per 90 | 0.709 | 0.777 |
| Non-penalty goals per 90 | 0.782 | 0.861 |
| Goals per 90 | 0.783 | 0.851 |
| xG per 90 | 0.863 | 0.931 |
| Non-penalty xG per 90 | 0.864 | 0.923 |
| Expected goal involvements per 90 | 0.891 | 0.944 |

### MID - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.548 | 0.638 |
| Goal Threat | 0.636 | 0.674 |
| Build-up Involvement | 0.653 | 0.751 |

### DEF - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Key passes per 90 | 0.444 | 0.535 |
| Assists per 90 | 0.564 | 0.632 |
| xA (expected assists) per 90 | 0.666 | 0.734 |
| Non-penalty xG per 90 | 0.776 | 0.835 |
| Goals per 90 | 0.776 | 0.838 |
| xG per 90 | 0.794 | 0.866 |
| Non-penalty goals per 90 | 0.796 | 0.874 |
| Expected goal involvements per 90 | 0.837 | 0.897 |

### DEF - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.579 | 0.634 |
| Goal Threat | 0.588 | 0.633 |
| Build-up Involvement | 0.666 | 0.747 |

### FWD - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Shots per 90 | 0.59 | 0.734 |
| Non-penalty xG per shot | 0.648 | 0.691 |
| xA (expected assists) per 90 | 0.667 | 0.734 |
| Non-penalty goals per 90 | 0.753 | 0.833 |
| Goals per 90 | 0.759 | 0.857 |
| xG per 90 | 0.838 | 0.896 |
| Non-penalty xG per 90 | 0.864 | 0.902 |
| Non-penalty goals - xG per 90 | 0.989 | 0.997 |

### FWD - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.55 | 0.654 |
| Build-up Involvement | 0.629 | 0.715 |
| Goal Threat | 0.668 | 0.703 |

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

- This pool is **Six leagues 2014/15-2024/25 (Understat)** - real players, real seasons. See the caveats on the Home page and in `src/config.py` for exactly what this source does and does not measure.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.