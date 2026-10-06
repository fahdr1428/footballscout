# Model validation report

Pool: **22,184 player-seasons**, minimum **900 minutes**, seasons 2014-15, 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25, 2025-26, 2026-27.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Not run. This source does not record positions specific enough to test - it publishes broad buckets rather than 'Left-Back' and 'Second Striker'.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DEF | 8601 | 12 | 3 | 5 | 0.296 | 64197.3 | 1006 |
| MID | 7033 | 14 | 4 | 6 | 0.225 | 52267.0 | 661 |
| FWD | 4881 | 13 | 3 | 5 | 0.252 | 38034.4 | 946 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| DEF | 0.471 | 0.205 | 0.676 |
| MID | 0.516 | 0.133 | 0.649 |
| FWD | 0.481 | 0.231 | 0.712 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| DEF | 35336 | 8600 | 0.017 | 0.001 | 14.6 | 1284 |
| MID | 29218 | 7032 | 0.028 | 0.001 | 20.0 | 914 |
| FWD | 18884 | 4880 | 0.029 | 0.002 | 14.0 | 763 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2e. Which metrics describe the player, and which the season?

Each metric's season-to-season correlation for the same player in the same position (900+ minutes in both), measured over the source's whole history with values standardised within each season. Its similarity weight is that correlation squared, so a metric that repeats at 0.8 counts four times as much as one at 0.4, and one that barely repeats at all - mostly the season's noise - hardly counts. Chosen by cross-validation over players: held-out players' own other season landed in their top ten more often under it on every source tested, and it held for players who had changed club in between, so the weights are not simply recognising clubs. Metrics marked (club) - goals conceded, clean sheets, a keeper's saves - describe the team in front of him, so they take the median weight whatever their figure.

| position_group | metrics_measured | median_repeatability | most_repeatable | least_repeatable | top_weight_share |
| --- | --- | --- | --- | --- | --- |
| DEF | 12 of 12 | 0.54 | Key passes per 90 0.81, xGChain (possessions ending in a shot) per 90 0.71, xA (expected assists) per 90 0.69 | Non-penalty goals per 90 0.23, Goals per 90 0.25, Yellow cards per 90 0.34 | 0.176 |
| MID | 14 of 14 | 0.7 | Shots per 90 0.80, Expected goal involvements per 90 0.79, Key passes per 90 0.78 | Non-penalty goals - xG per 90 0.06, Yellow cards per 90 0.44, Assists per 90 0.49 | 0.111 |
| FWD | 13 of 13 | 0.62 | xG per 90 0.69, Key passes per 90 0.69, Shots per 90 0.69 | Non-penalty goals - xG per 90 0.07, Assists per 90 0.34, Non-penalty goals per 90 0.50 | 0.107 |

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| DEF | 0.001 | 0.001 | 1.1 |
| MID | 0.002 | 0.001 | 3.5 |
| FWD | 0.002 | 0.001 | 3.7 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 14,134 of 20,221 player-seasons (70%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 1402 | 26.4 | 10000000.0 | 0.64 | 0.193 |
| 2 | 1428 | 32.4 | 10000000.0 | 0.71 | 0.256 |
| 3 | 1428 | 36.8 | 11000000.0 | 0.78 | 0.295 |
| 4 | 1373 | 40.3 | 10000000.0 | 0.8 | 0.327 |
| 5 | 1434 | 43.3 | 9000000.0 | 0.77 | 0.312 |
| 6 | 1437 | 46.2 | 8000000.0 | 0.8 | 0.346 |
| 7 | 1410 | 49.2 | 8000000.0 | 0.83 | 0.367 |
| 8 | 1393 | 52.6 | 6000000.0 | 0.88 | 0.391 |
| 9 | 1418 | 56.4 | 4500000.0 | 0.84 | 0.37 |
| 10 | 1411 | 63.5 | 3000000.0 | 1.0 | 0.456 |

Median value change runs from **x0.64** in the bottom decile to **x1.0** in the top, and the share of players whose value rose climbs from 19% to 46%. Rank correlation of score against growth: **0.153**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (20 cells, 14,134 players, minimum 40 each) gives a correlation of **0.027** - roughly half the headline figure, positive in 14 of 20 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**MID** - even share would be 0.071 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Shots per 90 | 0.1107 | 0.1175 | 1.64 |
| Key passes per 90 | 0.1047 | 0.1123 | 1.57 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0819 | 0.1044 | 1.46 |
| xGChain (possessions ending in a shot) per 90 | 0.0966 | 0.0894 | 1.25 |
| xA (expected assists) per 90 | 0.093 | 0.0829 | 1.16 |
| Expected goal involvements per 90 | 0.1085 | 0.0773 | 1.08 |

**DEF** - even share would be 0.083 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Key passes per 90 | 0.1762 | 0.1676 | 2.01 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.1212 | 0.1272 | 1.53 |
| xGChain (possessions ending in a shot) per 90 | 0.1365 | 0.1206 | 1.45 |
| Shots per 90 | 0.11 | 0.1192 | 1.43 |
| xA (expected assists) per 90 | 0.1301 | 0.1025 | 1.23 |
| Expected goal involvements per 90 | 0.11 | 0.0799 | 0.96 |

**FWD** - even share would be 0.077 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Key passes per 90 | 0.1066 | 0.125 | 1.62 |
| Shots per 90 | 0.1063 | 0.1153 | 1.5 |
| Non-penalty xG per shot | 0.0829 | 0.1054 | 1.37 |
| xGBuildup (xGChain excluding shots and key passes) per 90 | 0.0854 | 0.0962 | 1.25 |
| xG per 90 | 0.1069 | 0.0875 | 1.14 |
| Non-penalty xG per 90 | 0.1018 | 0.0838 | 1.09 |


## 4. Sensitivity of the similarity rankings

### MID - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.564 | 0.705 |
| Non-penalty xG per shot | 0.671 | 0.73 |
| xA (expected assists) per 90 | 0.724 | 0.779 |
| Non-penalty goals per 90 | 0.755 | 0.808 |
| Goals per 90 | 0.765 | 0.803 |
| Non-penalty xG per 90 | 0.844 | 0.901 |
| xG per 90 | 0.865 | 0.909 |
| Expected goal involvements per 90 | 0.9 | 0.943 |

### MID - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.547 | 0.643 |
| Goal Threat | 0.632 | 0.653 |
| Build-up Involvement | 0.661 | 0.753 |

### DEF - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Key passes per 90 | 0.43 | 0.512 |
| Assists per 90 | 0.588 | 0.631 |
| xA (expected assists) per 90 | 0.712 | 0.767 |
| Goals per 90 | 0.802 | 0.867 |
| Non-penalty goals per 90 | 0.827 | 0.887 |
| Non-penalty xG per 90 | 0.837 | 0.896 |
| Expected goal involvements per 90 | 0.84 | 0.865 |
| xG per 90 | 0.843 | 0.906 |

### DEF - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.588 | 0.627 |
| Goal Threat | 0.592 | 0.641 |
| Build-up Involvement | 0.698 | 0.734 |

### FWD - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Shots per 90 | 0.557 | 0.619 |
| xA (expected assists) per 90 | 0.656 | 0.696 |
| Non-penalty xG per shot | 0.668 | 0.759 |
| Goals per 90 | 0.712 | 0.771 |
| Non-penalty goals per 90 | 0.748 | 0.743 |
| xG per 90 | 0.789 | 0.825 |
| Non-penalty xG per 90 | 0.864 | 0.898 |
| Non-penalty goals - xG per 90 | 0.991 | 0.992 |

### FWD - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.556 | 0.547 |
| Goal Threat | 0.616 | 0.703 |
| Build-up Involvement | 0.655 | 0.681 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**MID** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Goals per 90 | Non-penalty goals per 90 | 0.952 |
| xG per 90 | Non-penalty xG per 90 | 0.947 |
| xG per 90 | Expected goal involvements per 90 | 0.892 |
| xA (expected assists) per 90 | Key passes per 90 | 0.865 |
| xA (expected assists) per 90 | Expected goal involvements per 90 | 0.86 |

**DEF** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Goals per 90 | Non-penalty goals per 90 | 0.973 |
| xG per 90 | Non-penalty xG per 90 | 0.964 |
| xGChain (possessions ending in a shot) per 90 | xGBuildup (xGChain excluding shots and key passes) per 90 | 0.927 |
| xA (expected assists) per 90 | Key passes per 90 | 0.872 |
| xA (expected assists) per 90 | Expected goal involvements per 90 | 0.856 |

**FWD** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| xG per 90 | Non-penalty xG per 90 | 0.957 |
| Goals per 90 | Non-penalty goals per 90 | 0.956 |
| xG per 90 | Expected goal involvements per 90 | 0.909 |
| Non-penalty xG per 90 | Expected goal involvements per 90 | 0.867 |
| Expected goal involvements per 90 | xGChain (possessions ending in a shot) per 90 | 0.861 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- This pool is **Six leagues 2014/15 to date (Understat)** - real players, real seasons. See the caveats on the Home page and in `src/config.py` for exactly what this source does and does not measure.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.