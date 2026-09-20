# Model validation report

Pool: **12,914 player-seasons**, minimum **900 minutes**, seasons 2017-18, 2018-19, 2019-20, 2020-21, 2021-22, 2022-23, 2023-24, 2024-25.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Before asking whether the models are any good, the groups they are fitted on have to be the right ones. Each row trains a cross-validated classifier to tell two specific positions apart on their own model features. **Balanced accuracy**, so 0.50 is a coin flip whatever the class imbalance; the two control rows are pairs nobody doubts are different jobs, and exist to show the measurement works.

| pair | kind | players | balanced_accuracy | chance | majority_class | modelled_separately | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Second striker vs attacking midfield | candidate | 870 | 0.745 | 0.5 | 0.822 | True | borderline |
| Wide midfield vs winger | candidate | 839 | 0.742 | 0.5 | 0.856 | True | borderline |
| Left-back vs right-back | candidate | 2208 | 0.595 | 0.5 | 0.537 | False | the same job - splitting would only cost peers |
| Left wing vs right wing | candidate | 1503 | 0.572 | 0.5 | 0.522 | False | the same job - splitting would only cost peers |
| Centre-back vs defensive midfield | control | 3792 | 0.936 | 0.5 | 0.706 | True | different jobs - deserves its own model |
| Defensive vs attacking midfield | control | 1830 | 0.957 | 0.5 | 0.609 | True | different jobs - deserves its own model |

A pair the classifier cannot separate is one job under two names: giving them separate peer groups would halve the sample and buy nothing. A pair it separates easily is two jobs, and measuring one against the other's percentiles is a bias no sample size fixes. The taxonomy in `src/config.py` follows this table - second strikers and wide midfielders are modelled apart, left and right are not - and the verdict column says so explicitly when the code and the evidence disagree.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 954 | 15 | 4 | 5 | 0.164 | 9505.7 | 95 |
| CB | 2677 | 19 | 3 | 5 | 0.18 | 35592.6 | 309 |
| FB | 2208 | 20 | 3 | 6 | 0.185 | 32487.8 | 241 |
| DM | 1115 | 21 | 3 | 5 | 0.161 | 17198.8 | 112 |
| CM | 1806 | 21 | 4 | 5 | 0.167 | 25592.7 | 212 |
| AM | 715 | 19 | 3 | 5 | 0.201 | 9546.9 | 95 |
| SS | 155 | 23 | 3 | 6 | 0.115 | 2734.5 | 23 |
| WM | 196 | 24 | 3 | 5 | 0.139 | 3628.5 | 43 |
| W | 1503 | 21 | 3 | 5 | 0.173 | 22569.2 | 221 |
| FW | 1572 | 21 | 3 | 6 | 0.164 | 25453.7 | 170 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.247 | 0.182 | 0.429 |
| CB | 0.269 | 0.186 | 0.455 |
| FB | 0.255 | 0.176 | 0.431 |
| DM | 0.232 | 0.189 | 0.421 |
| CM | 0.258 | 0.171 | 0.428 |
| AM | 0.305 | 0.168 | 0.473 |
| SS | 0.238 | 0.16 | 0.398 |
| WM | 0.235 | 0.158 | 0.393 |
| W | 0.322 | 0.149 | 0.472 |
| FW | 0.256 | 0.153 | 0.41 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 3390 | 953 | 0.099 | 0.01 | 9.4 | 204 |
| CB | 9140 | 2676 | 0.099 | 0.004 | 26.5 | 351 |
| FB | 7220 | 2207 | 0.12 | 0.005 | 26.5 | 233 |
| DM | 3772 | 1114 | 0.193 | 0.009 | 21.5 | 125 |
| CM | 6170 | 1805 | 0.146 | 0.006 | 26.3 | 184 |
| AM | 2338 | 714 | 0.185 | 0.014 | 13.2 | 85 |
| SS | 548 | 154 | 0.387 | 0.065 | 6.0 | 20 |
| WM | 588 | 195 | 0.398 | 0.051 | 7.8 | 15 |
| W | 4664 | 1502 | 0.162 | 0.007 | 24.4 | 132 |
| FW | 5098 | 1571 | 0.154 | 0.006 | 24.2 | 141 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.004 | 0.0 | 10.8 |
| CB | 0.01 | 0.001 | 9.8 |
| FB | 0.013 | 0.001 | 13.7 |
| DM | 0.009 | 0.001 | 10.2 |
| CM | 0.009 | 0.001 | 9.2 |
| AM | 0.002 | 0.001 | 2.2 |
| SS | 0.002 | 0.001 | 2.0 |
| WM | 0.008 | 0.002 | 4.3 |
| W | 0.005 | 0.001 | 5.4 |
| FW | 0.005 | 0.001 | 5.4 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 4,641 of 9,537 player-seasons (49%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 465 | 33.2 | 18000000.0 | 0.67 | 0.144 |
| 2 | 469 | 39.1 | 17000000.0 | 0.7 | 0.186 |
| 3 | 464 | 43.1 | 15000000.0 | 0.7 | 0.22 |
| 4 | 452 | 46.1 | 12000000.0 | 0.73 | 0.285 |
| 5 | 473 | 48.8 | 12000000.0 | 0.75 | 0.275 |
| 6 | 450 | 51.5 | 10000000.0 | 0.78 | 0.307 |
| 7 | 474 | 54.2 | 8000000.0 | 0.75 | 0.302 |
| 8 | 471 | 57.3 | 6500000.0 | 0.8 | 0.318 |
| 9 | 459 | 60.6 | 6000000.0 | 0.93 | 0.431 |
| 10 | 464 | 66.05 | 5000000.0 | 1.0 | 0.468 |

Median value change runs from **x0.67** in the bottom decile to **x1.0** in the top, and the share of players whose value rose climbs from 14% to 47%. Rank correlation of score against growth: **0.209**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (19 cells, 4,619 players, minimum 40 each) gives a correlation of **0.084** - roughly half the headline figure, positive in 17 of 19 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.048 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pass completion % | 0.0577 | 1.21 |
| Dribble success % | 0.0574 | 1.2 |
| Tackles per 90 | 0.0555 | 1.16 |
| Non-penalty xG per shot | 0.0544 | 1.14 |
| Crosses per 90 | 0.0516 | 1.08 |
| Assists per 90 | 0.0505 | 1.06 |

**CB** - even share would be 0.053 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Aerial duel success % | 0.0602 | 1.14 |
| Tackle success % | 0.0601 | 1.14 |
| Long-pass completion % | 0.0587 | 1.11 |
| Long passes attempted per 90 | 0.0572 | 1.09 |
| Errors leading to shot per 90 | 0.0559 | 1.06 |
| Pressures per 90 | 0.0558 | 1.06 |

**DM** - even share would be 0.048 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Tackle success % | 0.0552 | 1.16 |
| Aerial duel success % | 0.0551 | 1.16 |
| Long-pass completion % | 0.0529 | 1.11 |
| Pressure success % | 0.0517 | 1.09 |
| Share of passes made under pressure % | 0.0498 | 1.05 |
| Interceptions per 90 | 0.0495 | 1.04 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.695 | 0.762 |
| Shots per 90 | 0.713 | 0.832 |
| Assists per 90 | 0.728 | 0.782 |
| Non-penalty goals per 90 | 0.751 | 0.803 |
| Key passes per 90 | 0.791 | 0.81 |
| xA (expected assists) per 90 | 0.797 | 0.853 |
| Touches in opposition box per 90 | 0.822 | 0.885 |
| Non-penalty xG per 90 | 0.831 | 0.903 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.533 | 0.575 |
| Finishing | 0.572 | 0.62 |
| Dribbling | 0.58 | 0.647 |
| Ball Progression | 0.598 | 0.614 |
| Box Threat | 0.623 | 0.66 |
| Passing | 0.686 | 0.759 |
| Defending | 0.734 | 0.833 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.554 | 0.609 |
| Interceptions per 90 | 0.674 | 0.781 |
| Blocks per 90 | 0.736 | 0.801 |
| Pressures per 90 | 0.752 | 0.8 |
| Ball recoveries per 90 | 0.778 | 0.834 |
| Clearances per 90 | 0.788 | 0.816 |
| Tackles per 90 | 0.8 | 0.835 |
| Aerial duels won per 90 | 0.813 | 0.879 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.464 | 0.447 |
| Aerial | 0.555 | 0.594 |
| Ball Progression | 0.576 | 0.624 |
| Passing | 0.576 | 0.628 |
| Dribbling | 0.771 | 0.84 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.654 | 0.709 |
| Interceptions per 90 | 0.702 | 0.794 |
| Share of passes made under pressure % | 0.732 | 0.784 |
| Clearances per 90 | 0.76 | 0.814 |
| Pressures per 90 | 0.774 | 0.811 |
| Blocks per 90 | 0.784 | 0.814 |
| Tackles per 90 | 0.8 | 0.842 |
| Ball recoveries per 90 | 0.854 | 0.892 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.491 | 0.507 |
| Aerial | 0.602 | 0.663 |
| Passing | 0.616 | 0.645 |
| Ball Progression | 0.619 | 0.657 |
| Dribbling | 0.811 | 0.85 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.934 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.955 |
| Progressive passes per 90 | Passes into final third per 90 | 0.875 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.881 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- This pool is **Big five leagues 2017/18-2025/26 (FBref + Transfermarkt)** - real players, real seasons. See the caveats on the Home page and in `src/config.py` for exactly what this source does and does not measure.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.