# Model validation report

Pool: **5,745 player-seasons**, minimum **900 minutes**, seasons 2023-24, 2024-25.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Before asking whether the models are any good, the groups they are fitted on have to be the right ones. Each row trains a cross-validated classifier to tell two specific positions apart on their own model features. **Balanced accuracy**, so 0.50 is a coin flip whatever the class imbalance; the two control rows are pairs nobody doubts are different jobs, and exist to show the measurement works.

| pair | kind | players | balanced_accuracy | chance | majority_class | modelled_separately | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Second striker vs attacking midfield | candidate | 574 | 0.49 | 0.5 | 0.559 | True | the same job - splitting would only cost peers - BUT the taxonomy does the opposite |
| Wide midfield vs winger | candidate | 277 | 0.596 | 0.5 | 0.509 | True | the same job - splitting would only cost peers - BUT the taxonomy does the opposite |
| Left-back vs right-back | candidate | 444 | 0.535 | 0.5 | 0.583 | False | the same job - splitting would only cost peers |
| Left wing vs right wing | candidate | 277 | 0.558 | 0.5 | 0.509 | False | the same job - splitting would only cost peers |
| Centre-back vs defensive midfield | control | 939 | 0.999 | 0.5 | 0.607 | True | different jobs - deserves its own model |
| Defensive vs attacking midfield | control | 823 | 1.0 | 0.5 | 0.693 | True | different jobs - deserves its own model |

A pair the classifier cannot separate is one job under two names: giving them separate peer groups would halve the sample and buy nothing. A pair it separates easily is two jobs, and measuring one against the other's percentiles is a bias no sample size fixes. The taxonomy in `src/config.py` follows this table - second strikers and wide midfielders are modelled apart, left and right are not - and the verdict column says so explicitly when the code and the evidence disagree.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster | adjusted_rand_vs_true_role | cluster_purity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 557 | 14 | 4 | 6 | 0.128 | 5351.8 | 93 | 0.115 | 0.472 |
| CB | 1181 | 20 | 4 | 5 | 0.13 | 16107.9 | 242 | 0.277 | 0.606 |
| FB | 877 | 21 | 5 | 5 | 0.117 | 12076.1 | 104 | 0.238 | 0.617 |
| DM | 570 | 22 | 4 | 5 | 0.127 | 8388.7 | 122 | 0.323 | 0.667 |
| CM | 863 | 22 | 4 | 5 | 0.121 | 13805.1 | 120 | 0.194 | 0.572 |
| AM | 574 | 22 | 3 | 6 | 0.176 | 9096.1 | 77 | 0.162 | 0.505 |
| W | 565 | 23 | 4 | 4 | 0.192 | 8083.5 | 86 | 0.232 | 0.623 |
| FW | 558 | 22 | 3 | 5 | 0.181 | 8788.2 | 95 | 0.173 | 0.513 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.279 | 0.153 | 0.432 |
| CB | 0.259 | 0.153 | 0.412 |
| FB | 0.244 | 0.153 | 0.396 |
| DM | 0.295 | 0.145 | 0.44 |
| CM | 0.199 | 0.181 | 0.38 |
| AM | 0.265 | 0.214 | 0.48 |
| W | 0.262 | 0.215 | 0.476 |
| FW | 0.268 | 0.198 | 0.465 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2. Does similarity find players who do the same job?

| position_group | roles | top10_same_role | chance_baseline | lift |
| --- | --- | --- | --- | --- |
| GK | 4 | 0.535 | 0.25 | 2.13 |
| CB | 4 | 0.609 | 0.252 | 2.42 |
| FB | 4 | 0.674 | 0.25 | 2.69 |
| DM | 4 | 0.648 | 0.251 | 2.58 |
| CM | 4 | 0.625 | 0.251 | 2.49 |
| AM | 4 | 0.687 | 0.251 | 2.74 |
| W | 4 | 0.74 | 0.254 | 2.92 |
| FW | 4 | 0.726 | 0.252 | 2.88 |

`top10_same_role` is the share of a player's ten nearest neighbours drawn from the same generative role; `chance_baseline` is what random picking would produce given the role mix in that position. A lift above 1 means the model is recovering role, not noise.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 440 | 556 | 0.095 | 0.018 | 5.3 | 120 |
| CB | 950 | 1180 | 0.045 | 0.008 | 5.3 | 256 |
| FB | 710 | 876 | 0.045 | 0.011 | 3.9 | 185 |
| DM | 482 | 569 | 0.083 | 0.018 | 4.7 | 93 |
| CM | 694 | 862 | 0.052 | 0.012 | 4.5 | 169 |
| AM | 452 | 573 | 0.086 | 0.017 | 4.9 | 89 |
| W | 450 | 564 | 0.087 | 0.018 | 4.9 | 92 |
| FW | 446 | 557 | 0.09 | 0.018 | 5.0 | 72 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2e. Which metrics describe the player, and which the season?

Each metric's season-to-season correlation for the same player in the same position (900+ minutes in both), measured over the source's whole history with values standardised within each season. Its similarity weight is that correlation squared, so a metric that repeats at 0.8 counts four times as much as one at 0.4, and one that barely repeats at all - mostly the season's noise - hardly counts. Chosen by cross-validation over players: held-out players' own other season landed in their top ten more often under it on every source tested, and it held for players who had changed club in between, so the weights are not simply recognising clubs. Metrics marked (club) - goals conceded, clean sheets, a keeper's saves - describe the team in front of him, so they take the median weight whatever their figure.

| position_group | metrics_measured | median_repeatability | most_repeatable | least_repeatable | top_weight_share |
| --- | --- | --- | --- | --- | --- |
| GK | 14 of 14 | 0.26 | Saves per 90 0.53 (club), Goals conceded per 90 0.44 (club), Long goal-kicks / launches per 90 0.34 | Long passes attempted per 90 0.06, Share of appearances that were starts % 0.10, Progressive passes per 90 0.18 | 0.142 |
| CB | 20 of 20 | 0.31 | Passes into final third per 90 0.55, Progressive passes per 90 0.54, Ball recoveries per 90 0.42 | Errors leading to shot per 90 0.01, Tackle success % 0.08, Pass completion % 0.08 | 0.157 |
| FB | 21 of 21 | 0.29 | Crosses per 90 0.47, Passes attempted per 90 0.40, Shot-creating actions per 90 0.39 | Aerial duel success % 0.09, Pressures per 90 0.10, Tackle success % 0.12 | 0.118 |
| DM | 22 of 22 | 0.32 | Progressive carries per 90 0.53, Share of passes made under pressure % 0.47, Passes attempted per 90 0.47 | Pass completion % 0.04, Aerial duel success % 0.04, Aerial duels won per 90 0.10 | 0.115 |
| CM | 22 of 22 | 0.28 | Passes attempted per 90 0.51, Touches in opposition box per 90 0.39, Share of passes made under pressure % 0.39 | Aerial duel success % 0.06, Successful dribbles per 90 0.11, Pass completion % 0.12 | 0.136 |
| AM | 22 of 22 | 0.41 | Ball recoveries per 90 0.57, Non-penalty xG per 90 0.54, Touches in opposition box per 90 0.54 | Dribble success % 0.09, Progressive passes per 90 0.16, Pass completion % 0.18 | 0.09 |
| W | 23 of 23 | 0.36 | Non-penalty xG from open play per 90 0.53, Shots per 90 0.51, Non-penalty xG per 90 0.50 | Non-penalty xG per shot 0.04, Progressive passes per 90 0.05, Assists per 90 0.22 | 0.088 |
| FW | 22 of 22 | 0.41 | Non-penalty goals per 90 0.56, Shot-creating actions per 90 0.56, Progressive carries per 90 0.54 | Pass completion % 0.01, Progressive passes per 90 0.16, Non-penalty xG per shot 0.20 | 0.093 |

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.001 | 0.001 | 1.3 |
| CB | 0.001 | 0.001 | 0.9 |
| FB | 0.001 | 0.001 | 0.5 |
| DM | 0.002 | 0.001 | 1.7 |
| CM | 0.003 | 0.001 | 1.9 |
| AM | 0.003 | 0.001 | 2.3 |
| W | 0.002 | 0.001 | 2.0 |
| FW | 0.001 | 0.001 | 1.2 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Not run. This needs a source with market values and at least three seasons loaded in the pool - select more seasons in the sidebar.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.043 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Shots per 90 | 0.0822 | 0.0754 | 1.73 |
| Non-penalty xG from open play per 90 | 0.0885 | 0.0693 | 1.59 |
| Shot-creating actions per 90 | 0.0692 | 0.0689 | 1.59 |
| Passes into penalty area per 90 | 0.0643 | 0.0679 | 1.56 |
| Non-penalty xG per 90 | 0.0806 | 0.0629 | 1.45 |
| Touches in opposition box per 90 | 0.0648 | 0.0613 | 1.41 |

**CB** - even share would be 0.050 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Passes into final third per 90 | 0.1565 | 0.1171 | 2.34 |
| Progressive passes per 90 | 0.152 | 0.1143 | 2.29 |
| Ball recoveries per 90 | 0.0925 | 0.0962 | 1.92 |
| Aerial duel success % | 0.07 | 0.0772 | 1.54 |
| Aerial duels won per 90 | 0.0723 | 0.0709 | 1.42 |
| Tackles per 90 | 0.0681 | 0.0675 | 1.35 |

**DM** - even share would be 0.045 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Progressive carries per 90 | 0.1151 | 0.1138 | 2.5 |
| Share of passes made under pressure % | 0.0921 | 0.0911 | 2.0 |
| Passes attempted per 90 | 0.0917 | 0.0848 | 1.87 |
| Ball recoveries per 90 | 0.0744 | 0.0725 | 1.6 |
| Progressive passes received per 90 | 0.0676 | 0.07 | 1.54 |
| Passes into final third per 90 | 0.0762 | 0.0691 | 1.52 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Shots per 90 | 0.762 | 0.84 |
| Touches in opposition box per 90 | 0.817 | 0.839 |
| Non-penalty goals per 90 | 0.829 | 0.862 |
| Non-penalty xG from open play per 90 | 0.873 | 0.917 |
| Non-penalty xG per 90 | 0.902 | 0.941 |
| xA (expected assists) per 90 | 0.907 | 0.949 |
| Assists per 90 | 0.921 | 0.953 |
| Non-penalty xG per shot | 0.984 | 0.997 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.594 | 0.668 |
| Chance Creation | 0.62 | 0.614 |
| Dribbling | 0.647 | 0.681 |
| Finishing | 0.676 | 0.728 |
| Box Threat | 0.678 | 0.682 |
| Ball Progression | 0.695 | 0.725 |
| Passing | 0.752 | 0.757 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball recoveries per 90 | 0.628 | 0.643 |
| Tackles per 90 | 0.747 | 0.772 |
| Interceptions per 90 | 0.752 | 0.812 |
| Blocks per 90 | 0.781 | 0.817 |
| Clearances per 90 | 0.785 | 0.801 |
| Pressures per 90 | 0.788 | 0.827 |
| Pass completion under pressure % | 0.898 | 0.928 |
| Tackle success % | 0.934 | 0.969 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.541 | 0.526 |
| Ball Progression | 0.574 | 0.558 |
| Passing | 0.581 | 0.627 |
| Aerial | 0.637 | 0.718 |
| Dribbling | 0.67 | 0.716 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Share of passes made under pressure % | 0.732 | 0.786 |
| Ball recoveries per 90 | 0.79 | 0.877 |
| Blocks per 90 | 0.84 | 0.881 |
| Interceptions per 90 | 0.87 | 0.9 |
| Tackles per 90 | 0.875 | 0.903 |
| Clearances per 90 | 0.894 | 0.909 |
| Pass completion under pressure % | 0.909 | 0.933 |
| Tackle success % | 0.968 | 0.971 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball Progression | 0.605 | 0.555 |
| Passing | 0.616 | 0.584 |
| Defending | 0.634 | 0.592 |
| Dribbling | 0.712 | 0.737 |
| Aerial | 0.92 | 0.94 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Non-penalty xG from open play per 90 | Non-penalty xG per 90 | 0.993 |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.989 |
| xA (expected assists) per 90 | Key passes per 90 | 0.973 |
| Progressive carries per 90 | Carries into final third per 90 | 0.956 |
| Non-penalty xG per 90 | Shots per 90 | 0.913 |
| Non-penalty xG from open play per 90 | Shots per 90 | 0.906 |
| Non-penalty xG per 90 | Touches in opposition box per 90 | 0.87 |
| Non-penalty xG from open play per 90 | Touches in opposition box per 90 | 0.863 |
| Assists per 90 | xA (expected assists) per 90 | 0.852 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.989 |
| Pass completion under pressure % | Pass completion % | 0.944 |
| Progressive passes per 90 | Passes into final third per 90 | 0.937 |
| Aerial duels won per 90 | Aerial duel success % | 0.888 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.945 |
| Pass completion under pressure % | Pass completion % | 0.931 |
| Aerial duels won per 90 | Aerial duel success % | 0.872 |
| Tackles per 90 | Ball recoveries per 90 | 0.862 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.