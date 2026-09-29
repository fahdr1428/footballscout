# Model validation report

Pool: **8,088 player-seasons**, minimum **900 minutes**, seasons 2017-18, 2018-19, 2019-20, 2020-21, 2021-22.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Before asking whether the models are any good, the groups they are fitted on have to be the right ones. Each row trains a cross-validated classifier to tell two specific positions apart on their own model features. **Balanced accuracy**, so 0.50 is a coin flip whatever the class imbalance; the two control rows are pairs nobody doubts are different jobs, and exist to show the measurement works.

| pair | kind | players | balanced_accuracy | chance | majority_class | modelled_separately | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Second striker vs attacking midfield | candidate | 520 | 0.793 | 0.5 | 0.794 | True | different jobs - deserves its own model |
| Wide midfield vs winger | candidate | 524 | 0.742 | 0.5 | 0.832 | True | borderline |
| Left-back vs right-back | candidate | 1384 | 0.616 | 0.5 | 0.546 | False | the same job - splitting would only cost peers |
| Left wing vs right wing | candidate | 901 | 0.591 | 0.5 | 0.516 | False | the same job - splitting would only cost peers |
| Centre-back vs defensive midfield | control | 2401 | 0.953 | 0.5 | 0.698 | True | different jobs - deserves its own model |
| Defensive vs attacking midfield | control | 1137 | 0.97 | 0.5 | 0.637 | True | different jobs - deserves its own model |

A pair the classifier cannot separate is one job under two names: giving them separate peer groups would halve the sample and buy nothing. A pair it separates easily is two jobs, and measuring one against the other's percentiles is a bias no sample size fixes. The taxonomy in `src/config.py` follows this table - second strikers and wide midfielders are modelled apart, left and right are not - and the verdict column says so explicitly when the code and the evidence disagree.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 596 | 15 | 5 | 6 | 0.146 | 5771.9 | 29 |
| CB | 1677 | 19 | 3 | 5 | 0.127 | 24340.0 | 436 |
| FB | 1384 | 20 | 6 | 6 | 0.106 | 17485.5 | 96 |
| DM | 724 | 21 | 9 | 5 | 0.082 | 9531.1 | 52 |
| CM | 1145 | 21 | 3 | 5 | 0.162 | 17338.6 | 265 |
| AM | 413 | 20 | 4 | 5 | 0.135 | 5357.3 | 37 |
| SS | 107 | 23 | 2 | 4 | 0.205 | 1998.6 | 30 |
| WM | 132 | 24 | 2 | 5 | 0.202 | 2595.7 | 37 |
| W | 901 | 22 | 3 | 5 | 0.134 | 14020.1 | 99 |
| FW | 1007 | 21 | 4 | 5 | 0.112 | 14495.1 | 144 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.278 | 0.179 | 0.456 |
| CB | 0.239 | 0.158 | 0.397 |
| FB | 0.316 | 0.127 | 0.443 |
| DM | 0.246 | 0.114 | 0.36 |
| CM | 0.301 | 0.196 | 0.496 |
| AM | 0.347 | 0.143 | 0.491 |
| SS | 0.288 | 0.181 | 0.469 |
| WM | 0.264 | 0.185 | 0.45 |
| W | 0.369 | 0.14 | 0.509 |
| FW | 0.295 | 0.16 | 0.455 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 1408 | 595 | 0.175 | 0.017 | 10.4 | 68 |
| CB | 3844 | 1676 | 0.215 | 0.006 | 36.1 | 77 |
| FB | 2988 | 1383 | 0.273 | 0.007 | 37.8 | 46 |
| DM | 1646 | 723 | 0.405 | 0.014 | 29.3 | 18 |
| CM | 2568 | 1144 | 0.32 | 0.009 | 36.7 | 36 |
| AM | 874 | 412 | 0.397 | 0.024 | 16.4 | 17 |
| SS | 252 | 106 | 0.587 | 0.094 | 6.2 | 6 |
| WM | 252 | 131 | 0.683 | 0.076 | 8.9 | 4 |
| W | 1872 | 900 | 0.386 | 0.011 | 34.7 | 23 |
| FW | 2302 | 1006 | 0.355 | 0.01 | 35.7 | 27 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2e. Which metrics describe the player, and which the season?

Each metric's season-to-season correlation for the same player in the same position (900+ minutes in both), measured over the source's whole history with values standardised within each season. Its similarity weight is that correlation squared, so a metric that repeats at 0.8 counts four times as much as one at 0.4, and one that barely repeats at all - mostly the season's noise - hardly counts. Chosen by cross-validation over players: held-out players' own other season landed in their top ten more often under it on every source tested, and it held for players who had changed club in between, so the weights are not simply recognising clubs. Metrics marked (club) - goals conceded, clean sheets, a keeper's saves - describe the team in front of him, so they take the median weight whatever their figure.

| position_group | metrics_measured | median_repeatability | most_repeatable | least_repeatable | top_weight_share |
| --- | --- | --- | --- | --- | --- |
| GK | 15 of 15 | 0.49 | Pass completion % 0.76, Long goal-kicks / launches per 90 0.76, Long passes attempted per 90 0.73 | Share of appearances that were starts % 0.19, Goals prevented vs post-shot xG per 90 0.25, Save rate (saves / shots faced) % 0.33 | 0.13 |
| CB | 19 of 19 | 0.65 | Passes attempted per 90 0.77, Pass completion % 0.76, Long passes attempted per 90 0.70 | Errors leading to shot per 90 0.05, Tackle success % 0.12, Blocks per 90 0.39 | 0.085 |
| FB | 20 of 20 | 0.66 | Passes attempted per 90 0.78, Progressive carries per 90 0.78, Pass completion % 0.78 | Tackle success % 0.15, Blocks per 90 0.43, Interceptions per 90 0.46 | 0.074 |
| DM | 21 of 21 | 0.65 | Aerial duels won per 90 0.82, Passes attempted per 90 0.82, Long passes attempted per 90 0.80 | Tackle success % 0.06, Pressure success % 0.45, Blocks per 90 0.46 | 0.077 |
| CM | 21 of 21 | 0.7 | Passes attempted per 90 0.84, Pass completion % 0.79, Passes into final third per 90 0.79 | Pressure success % 0.41, Aerial duel success % 0.46, Interceptions per 90 0.53 | 0.073 |
| AM | 20 of 20 | 0.65 | Passes attempted per 90 0.81, Progressive carries per 90 0.76, Pass completion % 0.74 | Dribble success % 0.37, Non-penalty goals per 90 0.39, Assists per 90 0.39 | 0.08 |
| SS | 23 of 23 | 0.58 | Successful dribbles per 90 0.82, Passes attempted per 90 0.81, Shot-creating actions per 90 0.76 | Shot accuracy % 0.13, Non-penalty goals per 90 0.16, Dribble success % 0.31 | 0.087 |
| WM | 24 of 24 | 0.65 | Crosses per 90 0.84, Pass completion % 0.81, Pressures per 90 0.80 | Share of tackles in the attacking third % 0.20, Dribble success % 0.37, Non-penalty goals per 90 0.42 | 0.069 |
| W | 22 of 22 | 0.69 | Dribble attempts per 90 0.81, Touches in opposition box per 90 0.80, Progressive carries per 90 0.80 | Dribble success % 0.33, Assists per 90 0.35, Non-penalty goals per 90 0.49 | 0.064 |
| FW | 21 of 21 | 0.68 | Aerial duels won per 90 0.89, Passes attempted per 90 0.83, Progressive carries per 90 0.81 | Shot accuracy % 0.19, Dribble success % 0.20, Assists per 90 0.28 | 0.088 |

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.009 | 0.001 | 14.6 |
| CB | 0.011 | 0.002 | 6.9 |
| FB | 0.009 | 0.002 | 5.6 |
| DM | 0.001 | 0.001 | 1.0 |
| CM | 0.005 | 0.002 | 3.1 |
| AM | 0.005 | 0.001 | 3.3 |
| SS | 0.004 | 0.002 | 2.1 |
| WM | 0.011 | 0.003 | 3.4 |
| W | 0.008 | 0.002 | 4.5 |
| FW | 0.005 | 0.001 | 3.9 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Hidden-gem score in season *t*, against the player's Transfermarkt valuation **2 seasons later**. The score sees only season *t*, so nothing about the outcome enters it. 3,503 of 4,776 player-seasons (73%) could be followed up.

| decile | players | median_score | median_value_eur | median_growth_x | share_that_rose |
| --- | --- | --- | --- | --- | --- |
| 1 | 353 | 31.2 | 15000000.0 | 0.57 | 0.116 |
| 2 | 346 | 37.6 | 15000000.0 | 0.64 | 0.197 |
| 3 | 356 | 41.5 | 14500000.0 | 0.62 | 0.228 |
| 4 | 348 | 44.7 | 13000000.0 | 0.67 | 0.259 |
| 5 | 345 | 47.4 | 11000000.0 | 0.7 | 0.293 |
| 6 | 350 | 50.0 | 10000000.0 | 0.69 | 0.32 |
| 7 | 359 | 53.0 | 7500000.0 | 0.71 | 0.304 |
| 8 | 342 | 56.1 | 6500000.0 | 0.8 | 0.325 |
| 9 | 351 | 59.5 | 5500000.0 | 0.8 | 0.393 |
| 10 | 353 | 65.0 | 5000000.0 | 1.0 | 0.493 |

Median value change runs from **x0.57** in the bottom decile to **x1.0** in the top, and the share of players whose value rose climbs from 12% to 49%. Rank correlation of score against growth: **0.238**.

**But most of a monotone table like that can be an artefact.** The top decile is also younger and cheaper, and a cheap twenty-year-old rises in percentage terms for reasons the model can take no credit for. Asking the same question inside cells of similar age *and* similar starting price (19 cells, 3,487 players, minimum 40 each) gives a correlation of **0.114** - roughly half the headline figure, positive in 17 of 19 cells.

So: about half the apparent signal is youth and a low starting price, and about half is left over. A modest edge that survives both controls is a believable result for a model built from public data; the headline number on its own would be an overclaim.

Three limits. **Survivorship** - a player who left the big five has no later valuation and drops out, and those are disproportionately the ones who did not work out, so absolute growth figures flatter every decile. **Market value is Transfermarkt's estimate**, not a fee anyone paid, and it is partly informed by the same public data the model reads. **One market regime**, five seasons, one continent.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.045 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Pass completion % | 0.0597 | 0.0722 | 1.59 |
| Dribble attempts per 90 | 0.0642 | 0.0589 | 1.3 |
| Touches in opposition box per 90 | 0.0628 | 0.0579 | 1.27 |
| Tackles per 90 | 0.0448 | 0.0574 | 1.26 |
| Pressures per 90 | 0.041 | 0.0558 | 1.23 |
| Carries into penalty area per 90 | 0.0565 | 0.0552 | 1.22 |

**CB** - even share would be 0.053 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Pass completion % | 0.0824 | 0.0801 | 1.52 |
| Passes attempted per 90 | 0.0848 | 0.074 | 1.41 |
| Long passes attempted per 90 | 0.0707 | 0.0727 | 1.38 |
| Pressures per 90 | 0.0657 | 0.0679 | 1.29 |
| Long-pass completion % | 0.0681 | 0.0671 | 1.27 |
| Ball recoveries per 90 | 0.0613 | 0.0669 | 1.27 |

**DM** - even share would be 0.048 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Aerial duels won per 90 | 0.0767 | 0.0746 | 1.57 |
| Pass completion % | 0.0703 | 0.0687 | 1.44 |
| Long passes attempted per 90 | 0.0739 | 0.0656 | 1.38 |
| Passes attempted per 90 | 0.0765 | 0.0608 | 1.28 |
| Progressive carries per 90 | 0.0623 | 0.0583 | 1.22 |
| Progressive passes per 90 | 0.0657 | 0.058 | 1.22 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Shots per 90 | 0.749 | 0.755 |
| Touches in opposition box per 90 | 0.782 | 0.831 |
| Non-penalty xG per shot | 0.796 | 0.835 |
| Key passes per 90 | 0.804 | 0.873 |
| Non-penalty xG per 90 | 0.831 | 0.886 |
| xA (expected assists) per 90 | 0.837 | 0.893 |
| Non-penalty goals per 90 | 0.839 | 0.906 |
| Assists per 90 | 0.872 | 0.921 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball Progression | 0.559 | 0.6 |
| Finishing | 0.581 | 0.617 |
| Chance Creation | 0.582 | 0.589 |
| Box Threat | 0.599 | 0.667 |
| Defending | 0.601 | 0.648 |
| Dribbling | 0.602 | 0.711 |
| Passing | 0.685 | 0.817 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball recoveries per 90 | 0.603 | 0.709 |
| Clearances per 90 | 0.648 | 0.687 |
| Pressures per 90 | 0.693 | 0.743 |
| Tackles per 90 | 0.695 | 0.756 |
| Interceptions per 90 | 0.7 | 0.734 |
| Aerial duels won per 90 | 0.787 | 0.852 |
| Blocks per 90 | 0.793 | 0.803 |
| Tackle success % | 0.97 | 0.982 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.519 | 0.41 |
| Passing | 0.569 | 0.615 |
| Aerial | 0.579 | 0.621 |
| Ball Progression | 0.609 | 0.647 |
| Dribbling | 0.769 | 0.825 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Clearances per 90 | 0.696 | 0.729 |
| Share of passes made under pressure % | 0.726 | 0.769 |
| Tackles per 90 | 0.742 | 0.798 |
| Pressures per 90 | 0.753 | 0.783 |
| Interceptions per 90 | 0.769 | 0.757 |
| Ball recoveries per 90 | 0.773 | 0.784 |
| Blocks per 90 | 0.795 | 0.844 |
| Tackle success % | 0.991 | 0.997 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.505 | 0.508 |
| Passing | 0.55 | 0.522 |
| Ball Progression | 0.585 | 0.585 |
| Aerial | 0.602 | 0.675 |
| Dribbling | 0.734 | 0.791 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.949 |
| Key passes per 90 | Shot-creating actions per 90 | 0.869 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.932 |
| Pass completion % | Long-pass completion % | 0.87 |
| Progressive passes per 90 | Passes into final third per 90 | 0.857 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.883 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- This pool is **Big five leagues 2017/18-2024/25 (FBref + Transfermarkt)** - real players, real seasons. See the caveats on the Home page and in `src/config.py` for exactly what this source does and does not measure.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.