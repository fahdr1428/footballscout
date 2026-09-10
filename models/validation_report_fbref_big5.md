# Model validation report

Pool: **8,088 player-seasons**, minimum **900 minutes**, seasons 2017-18, 2018-19, 2019-20, 2020-21, 2021-22.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 596 | 15 | 3 | 7 | 0.145 | 6500.7 | 190 |
| CB | 1677 | 19 | 4 | 5 | 0.114 | 22750.4 | 346 |
| FB | 1384 | 20 | 5 | 6 | 0.101 | 18318.5 | 114 |
| DM | 723 | 21 | 8 | 6 | 0.081 | 9820.5 | 71 |
| CM | 1145 | 21 | 3 | 5 | 0.162 | 17327.3 | 257 |
| AM | 520 | 20 | 4 | 5 | 0.137 | 6672.5 | 48 |
| W | 1034 | 22 | 3 | 6 | 0.143 | 16078.1 | 102 |
| FW | 1007 | 21 | 4 | 5 | 0.116 | 14512.6 | 143 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.279 | 0.186 | 0.464 |
| CB | 0.233 | 0.153 | 0.386 |
| FB | 0.312 | 0.132 | 0.444 |
| DM | 0.238 | 0.121 | 0.36 |
| CM | 0.306 | 0.191 | 0.497 |
| AM | 0.34 | 0.173 | 0.513 |
| W | 0.369 | 0.135 | 0.505 |
| FW | 0.301 | 0.154 | 0.455 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 1408 | 595 | 0.165 | 0.017 | 9.8 | 78 |
| CB | 3844 | 1676 | 0.172 | 0.006 | 28.8 | 100 |
| FB | 2988 | 1383 | 0.247 | 0.007 | 34.2 | 57 |
| DM | 1646 | 722 | 0.386 | 0.014 | 27.9 | 22 |
| CM | 2568 | 1144 | 0.307 | 0.009 | 35.1 | 39 |
| AM | 1160 | 519 | 0.368 | 0.019 | 19.1 | 22 |
| W | 2130 | 1033 | 0.35 | 0.01 | 36.1 | 29 |
| FW | 2302 | 1006 | 0.279 | 0.01 | 28.1 | 42 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.009 | 0.001 | 14.6 |
| CB | 0.019 | 0.002 | 12.5 |
| FB | 0.009 | 0.002 | 6.0 |
| DM | 0.005 | 0.001 | 4.1 |
| CM | 0.005 | 0.002 | 2.8 |
| AM | 0.003 | 0.001 | 2.7 |
| W | 0.012 | 0.002 | 7.4 |
| FW | 0.003 | 0.001 | 2.8 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 3,503 of 4,776 player-seasons (73%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 347 | 30.8 | 15000000.0 | 0.58 | 0.135 |
| 2 | 355 | 37.5 | 15000000.0 | 0.62 | 0.186 |
| 3 | 346 | 41.3 | 14000000.0 | 0.62 | 0.208 |
| 4 | 348 | 44.5 | 12000000.0 | 0.67 | 0.276 |
| 5 | 361 | 47.3 | 10000000.0 | 0.67 | 0.28 |
| 6 | 350 | 50.0 | 9750000.0 | 0.73 | 0.317 |
| 7 | 347 | 52.8 | 7000000.0 | 0.71 | 0.3 |
| 8 | 352 | 56.0 | 6500000.0 | 0.8 | 0.338 |
| 9 | 346 | 59.2 | 6500000.0 | 0.8 | 0.37 |
| 10 | 351 | 64.8 | 5000000.0 | 1.07 | 0.516 |

Median value change runs from **x0.58** in the bottom decile to **x1.07** in the top, and the share of players whose value rose climbs from 14% to 52%. Rank correlation of score against growth: **0.238**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (19 cells, 3,487 players, minimum 40 each) gives a correlation of **0.117** - roughly half the headline figure, positive in 16 of 19 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Dribble success % | 0.0587 | 1.29 |
| Pressures per 90 | 0.0555 | 1.22 |
| Tackles per 90 | 0.055 | 1.21 |
| Pass completion % | 0.0549 | 1.21 |
| Crosses per 90 | 0.051 | 1.12 |
| Assists per 90 | 0.0485 | 1.07 |

**CB** - even share would be 0.053 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Tackle success % | 0.0605 | 1.15 |
| Interceptions per 90 | 0.0597 | 1.13 |
| Aerial duel success % | 0.0569 | 1.08 |
| Fouls committed per 90 | 0.0565 | 1.07 |
| Ball recoveries per 90 | 0.056 | 1.06 |
| Tackles per 90 | 0.0557 | 1.06 |

**DM** - even share would be 0.048 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Tackle success % | 0.0554 | 1.16 |
| Interceptions per 90 | 0.0533 | 1.12 |
| Pressure success % | 0.052 | 1.09 |
| Aerial duel success % | 0.0512 | 1.08 |
| Share of passes made under pressure % | 0.0505 | 1.06 |
| Ball recoveries per 90 | 0.0505 | 1.06 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Assists per 90 | 0.71 | 0.844 |
| Non-penalty goals per 90 | 0.724 | 0.766 |
| Shots per 90 | 0.789 | 0.843 |
| Non-penalty xG per shot | 0.798 | 0.871 |
| Touches in opposition box per 90 | 0.811 | 0.891 |
| Key passes per 90 | 0.825 | 0.847 |
| xA (expected assists) per 90 | 0.834 | 0.889 |
| Non-penalty xG per 90 | 0.837 | 0.895 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Dribbling | 0.537 | 0.586 |
| Finishing | 0.554 | 0.568 |
| Defending | 0.565 | 0.575 |
| Ball Progression | 0.573 | 0.576 |
| Chance Creation | 0.591 | 0.619 |
| Box Threat | 0.622 | 0.673 |
| Passing | 0.673 | 0.786 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.604 | 0.647 |
| Blocks per 90 | 0.611 | 0.685 |
| Interceptions per 90 | 0.617 | 0.678 |
| Ball recoveries per 90 | 0.656 | 0.659 |
| Tackles per 90 | 0.687 | 0.744 |
| Clearances per 90 | 0.7 | 0.793 |
| Pressures per 90 | 0.756 | 0.84 |
| Aerial duels won per 90 | 0.789 | 0.873 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.466 | 0.406 |
| Aerial | 0.579 | 0.61 |
| Passing | 0.621 | 0.701 |
| Ball Progression | 0.641 | 0.664 |
| Dribbling | 0.78 | 0.852 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Interceptions per 90 | 0.707 | 0.783 |
| Tackle success % | 0.713 | 0.731 |
| Share of passes made under pressure % | 0.723 | 0.757 |
| Blocks per 90 | 0.731 | 0.759 |
| Tackles per 90 | 0.741 | 0.8 |
| Ball recoveries per 90 | 0.753 | 0.795 |
| Clearances per 90 | 0.765 | 0.797 |
| Pressures per 90 | 0.772 | 0.799 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.47 | 0.5 |
| Passing | 0.58 | 0.619 |
| Aerial | 0.642 | 0.656 |
| Ball Progression | 0.655 | 0.645 |
| Dribbling | 0.771 | 0.807 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.952 |
| Key passes per 90 | Shot-creating actions per 90 | 0.868 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.953 |
| Pass completion % | Long-pass completion % | 0.87 |
| Progressive passes per 90 | Passes into final third per 90 | 0.857 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.881 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.