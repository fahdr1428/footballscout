# Model validation report

Pool: **5,512 player-seasons**, minimum **900 minutes**, seasons 2023-24, 2024-25.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster | adjusted_rand_vs_true_role | cluster_purity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 554 | 14 | 3 | 5 | 0.129 | 5896.6 | 131 | 0.161 | 0.495 |
| CB | 1104 | 20 | 4 | 5 | 0.126 | 14884.4 | 221 | 0.296 | 0.625 |
| FB | 803 | 21 | 3 | 5 | 0.137 | 12479.0 | 184 | 0.18 | 0.518 |
| DM | 547 | 22 | 6 | 5 | 0.119 | 7581.9 | 60 | 0.22 | 0.631 |
| CM | 793 | 22 | 4 | 6 | 0.129 | 12498.7 | 151 | 0.244 | 0.614 |
| AM | 546 | 22 | 4 | 5 | 0.178 | 7826.3 | 52 | 0.213 | 0.586 |
| W | 573 | 23 | 4 | 4 | 0.22 | 7723.2 | 96 | 0.332 | 0.7 |
| FW | 592 | 22 | 3 | 6 | 0.17 | 9495.5 | 115 | 0.15 | 0.522 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.273 | 0.149 | 0.422 |
| CB | 0.277 | 0.144 | 0.421 |
| FB | 0.268 | 0.156 | 0.423 |
| DM | 0.269 | 0.144 | 0.413 |
| CM | 0.211 | 0.185 | 0.396 |
| AM | 0.269 | 0.206 | 0.475 |
| W | 0.263 | 0.226 | 0.489 |
| FW | 0.26 | 0.194 | 0.454 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2. Does similarity find players who do the same job?

| position_group | roles | top10_same_role | chance_baseline | lift |
| --- | --- | --- | --- | --- |
| GK | 4 | 0.526 | 0.251 | 2.1 |
| CB | 4 | 0.596 | 0.251 | 2.38 |
| FB | 4 | 0.659 | 0.251 | 2.62 |
| DM | 4 | 0.63 | 0.254 | 2.48 |
| CM | 4 | 0.633 | 0.252 | 2.51 |
| AM | 4 | 0.705 | 0.252 | 2.8 |
| W | 4 | 0.714 | 0.252 | 2.83 |
| FW | 4 | 0.663 | 0.255 | 2.6 |

`top10_same_role` is the share of a player's ten nearest neighbours drawn from the same generative role; `chance_baseline` is what random picking would produce given the role mix in that position. A lift above 1 means the model is recovering role, not noise.

## 2b. Does the engine recognise the same player twice?

| position_group | player_seasons_tested | candidates | own_season_in_top10 | chance | lift | median_rank |
| --- | --- | --- | --- | --- | --- | --- |
| GK | 454 | 553 | 0.07 | 0.018 | 3.9 | 121 |
| CB | 876 | 1103 | 0.029 | 0.009 | 3.1 | 277 |
| FB | 642 | 802 | 0.047 | 0.012 | 3.7 | 157 |
| DM | 424 | 546 | 0.061 | 0.018 | 3.3 | 134 |
| CM | 620 | 792 | 0.063 | 0.013 | 5.0 | 151 |
| AM | 440 | 545 | 0.091 | 0.018 | 5.0 | 81 |
| W | 464 | 572 | 0.073 | 0.017 | 4.2 | 87 |
| FW | 480 | 591 | 0.065 | 0.017 | 3.8 | 96 |

For every player with two seasons in the pool, this asks where his *other* season ranks among his nearest neighbours. It is the only case where the right answer is known without any labels, which makes it the check that also works on real data. `chance` is what random ordering would give.

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.005 | 0.001 | 3.8 |
| CB | 0.0 | 0.002 | 0.0 |
| FB | 0.001 | 0.001 | 0.5 |
| DM | 0.002 | 0.001 | 1.7 |
| CM | 0.004 | 0.002 | 2.6 |
| AM | 0.004 | 0.001 | 3.1 |
| W | 0.002 | 0.001 | 1.5 |
| FW | 0.001 | 0.001 | 1.2 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

## 3. Is the model dominated by a few metrics?

**W** - even share would be 0.043 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Pass completion % | 0.0556 | 1.28 |
| Tackles per 90 | 0.0554 | 1.28 |
| Pressures per 90 | 0.0553 | 1.27 |
| Dribble success % | 0.053 | 1.22 |
| Progressive passes per 90 | 0.0522 | 1.2 |
| Non-penalty xG per shot | 0.0464 | 1.07 |

**CB** - even share would be 0.050 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Tackle success % | 0.0574 | 1.15 |
| Errors leading to shot per 90 | 0.0571 | 1.14 |
| Long-pass completion % | 0.0559 | 1.12 |
| Pass completion under pressure % | 0.0526 | 1.05 |
| Progressive carries per 90 | 0.0524 | 1.05 |
| Passes attempted per 90 | 0.0511 | 1.02 |

**DM** - even share would be 0.045 per feature.

| metric | mean_distance_share | vs_even_share |
| --- | --- | --- |
| Aerial duel success % | 0.0498 | 1.1 |
| Long-pass completion % | 0.0491 | 1.08 |
| Blocks per 90 | 0.0485 | 1.07 |
| Tackle success % | 0.0484 | 1.06 |
| Share of passes made under pressure % | 0.0483 | 1.06 |
| Progressive passes received per 90 | 0.0475 | 1.05 |


## 4. Sensitivity of the similarity rankings

### W - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Non-penalty xG per shot | 0.756 | 0.842 |
| Assists per 90 | 0.809 | 0.881 |
| Non-penalty goals per 90 | 0.861 | 0.889 |
| Shots per 90 | 0.873 | 0.913 |
| xA (expected assists) per 90 | 0.887 | 0.931 |
| Touches in opposition box per 90 | 0.912 | 0.945 |
| Non-penalty xG per 90 | 0.936 | 0.965 |
| Non-penalty xG from open play per 90 | 0.94 | 0.961 |

### W - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball Progression | 0.596 | 0.616 |
| Defending | 0.615 | 0.64 |
| Finishing | 0.634 | 0.641 |
| Dribbling | 0.637 | 0.695 |
| Chance Creation | 0.65 | 0.653 |
| Box Threat | 0.686 | 0.737 |
| Passing | 0.696 | 0.751 |

### CB - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Tackle success % | 0.66 | 0.687 |
| Pressures per 90 | 0.725 | 0.816 |
| Blocks per 90 | 0.763 | 0.814 |
| Ball recoveries per 90 | 0.799 | 0.852 |
| Pass completion under pressure % | 0.804 | 0.867 |
| Interceptions per 90 | 0.804 | 0.86 |
| Tackles per 90 | 0.845 | 0.914 |
| Clearances per 90 | 0.846 | 0.883 |

### CB - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defending | 0.525 | 0.569 |
| Passing | 0.588 | 0.59 |
| Ball Progression | 0.599 | 0.624 |
| Aerial | 0.642 | 0.745 |
| Dribbling | 0.72 | 0.776 |

### DM - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Share of passes made under pressure % | 0.763 | 0.875 |
| Tackle success % | 0.773 | 0.797 |
| Blocks per 90 | 0.777 | 0.826 |
| Pass completion under pressure % | 0.804 | 0.832 |
| Clearances per 90 | 0.812 | 0.868 |
| Ball recoveries per 90 | 0.823 | 0.905 |
| Interceptions per 90 | 0.836 | 0.887 |
| Tackles per 90 | 0.875 | 0.947 |

### DM - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Passing | 0.529 | 0.564 |
| Defending | 0.551 | 0.544 |
| Ball Progression | 0.596 | 0.625 |
| Aerial | 0.657 | 0.797 |
| Dribbling | 0.768 | 0.812 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**W** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Non-penalty xG from open play per 90 | Non-penalty xG per 90 | 0.993 |
| Successful dribbles per 90 | Dribble attempts per 90 | 0.99 |
| xA (expected assists) per 90 | Key passes per 90 | 0.975 |
| Progressive carries per 90 | Carries into final third per 90 | 0.958 |
| Non-penalty xG per 90 | Touches in opposition box per 90 | 0.907 |
| Non-penalty xG from open play per 90 | Touches in opposition box per 90 | 0.899 |
| Non-penalty xG per 90 | Shots per 90 | 0.896 |
| Non-penalty xG from open play per 90 | Shots per 90 | 0.888 |
| Non-penalty goals per 90 | Non-penalty xG per 90 | 0.865 |
| Non-penalty xG from open play per 90 | Non-penalty goals per 90 | 0.863 |

**CB** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Aerial duels won per 90 | Aerial duels contested per 90 | 0.987 |
| Pass completion under pressure % | Pass completion % | 0.943 |
| Progressive passes per 90 | Passes into final third per 90 | 0.935 |
| Aerial duels won per 90 | Aerial duel success % | 0.888 |

**DM** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Progressive passes per 90 | Passes into final third per 90 | 0.94 |
| Pass completion under pressure % | Pass completion % | 0.928 |
| Aerial duels won per 90 | Aerial duel success % | 0.866 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- The bundled dataset is **simulated**. Absolute values are plausible but they are not real players, and no conclusion about a real footballer can be drawn from them.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.