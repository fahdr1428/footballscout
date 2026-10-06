# Model validation report

Pool: **339 player-seasons**, minimum **900 minutes**, seasons 2025-26.
All models are fitted per position group. Nothing below is tuned to make the numbers look better; where a result is weak it is reported and interpreted.

## 0. Are the position groups the right shape?

Before asking whether the models are any good, the groups they are fitted on have to be the right ones. Each row trains a cross-validated classifier to tell two specific positions apart on their own model features. **Balanced accuracy**, so 0.50 is a coin flip whatever the class imbalance; the two control rows are pairs nobody doubts are different jobs, and exist to show the measurement works.

| pair | kind | players | balanced_accuracy | chance | majority_class | modelled_separately | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Centre-back vs defensive midfield | control | 77 | 0.947 | 0.5 | 0.584 | True | different jobs - deserves its own model |

A pair the classifier cannot separate is one job under two names: giving them separate peer groups would halve the sample and buy nothing. A pair it separates easily is two jobs, and measuring one against the other's percentiles is a bias no sample size fixes. The taxonomy in `src/config.py` follows this table - second strikers and wide midfielders are modelled apart, left and right are not - and the verdict column says so explicitly when the code and the evidence disagree.

## 1. Clustering

| position_group | players | features | k | elbow_k | silhouette | inertia | smallest_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK | 24 | 8 | 2 | 2 | 0.38 | 121.3 | 7 |
| DEF | 128 | 16 | 3 | 6 | 0.169 | 1391.3 | 24 |
| MID | 153 | 16 | 4 | 5 | 0.149 | 1436.0 | 29 |
| FWD | 34 | 13 | 2 | 2 | 0.245 | 312.7 | 15 |

Silhouette scores in the 0.10-0.25 range are typical for football style data and should be read honestly: playing styles form a **continuum**, not well-separated groups. K-Means here is a useful summary of that continuum, not evidence that discrete player types exist. `k` is chosen as the largest k whose silhouette stays within 10% of the best score, with the inertia elbow reported alongside as a cross-check.

`adjusted_rand_vs_true_role` and `cluster_purity` compare the clusters with the role profiles used to *generate* the simulated dataset. They are only computable because the sample data is simulated - on a real feed there is no ground truth, and these columns would be absent.

### Variance shown by the 2-D archetype map

| position_group | pc1_variance | pc2_variance | total_shown |
| --- | --- | --- | --- |
| GK | 0.569 | 0.262 | 0.831 |
| DEF | 0.302 | 0.204 | 0.506 |
| MID | 0.391 | 0.191 | 0.581 |
| FWD | 0.406 | 0.274 | 0.679 |

The cluster map compresses 12-22 features into two axes, so a large share of the variance is not on screen. Two players sitting close together on the map are not necessarily close in the full feature space - the similarity table is the authority.

## 2e. Which metrics describe the player, and which the season?

Each metric's season-to-season correlation for the same player in the same position (900+ minutes in both), measured over the source's whole history with values standardised within each season. Its similarity weight is that correlation squared, so a metric that repeats at 0.8 counts four times as much as one at 0.4, and one that barely repeats at all - mostly the season's noise - hardly counts. Chosen by cross-validation over players: held-out players' own other season landed in their top ten more often under it on every source tested, and it held for players who had changed club in between, so the weights are not simply recognising clubs. Metrics marked (club) - goals conceded, clean sheets, a keeper's saves - describe the team in front of him, so they take the median weight whatever their figure.

| position_group | metrics_measured | median_repeatability | most_repeatable | least_repeatable | top_weight_share |
| --- | --- | --- | --- | --- | --- |
| GK | 8 of 8 | 0.34 | Goals conceded per 90 0.60 (club), Expected goals conceded (team, while on pitch) per 90 0.49 (club), Saves per 90 0.41 (club) | Share of appearances that were starts % -0.05, Bonus points system score per 90 0.17, Save rate (saves / shots faced) % 0.26 | 0.203 |
| DEF | 12 of 16 | 0.51 | Creativity (Opta index) per 90 0.88, xA (expected assists) per 90 0.72, Expected goal involvements per 90 0.65 | Goals per 90 0.20, Yellow cards per 90 0.37, Share of appearances that were starts % 0.38 | 0.157 |
| MID | 12 of 16 | 0.61 | Threat (Opta index) per 90 0.87, Creativity (Opta index) per 90 0.84, Expected goal involvements per 90 0.81 | Share of appearances that were starts % 0.35, Expected goals conceded (team, while on pitch) per 90 0.43 (club), Assists per 90 0.50 | 0.109 |
| FWD | 10 of 13 | 0.58 | Share of appearances that were starts % 0.64, Threat (Opta index) per 90 0.62, Creativity (Opta index) per 90 0.61 | Assists per 90 0.26, xA (expected assists) per 90 0.39, Goals per 90 0.53 | 0.099 |

## 2c. Is the engine matching on club rather than player?

| position_group | team_mates_in_top10 | chance | lift |
| --- | --- | --- | --- |
| GK | 0.008 | 0.014 | 0.6 |
| DEF | 0.073 | 0.044 | 1.7 |
| MID | 0.065 | 0.046 | 1.4 |
| FWD | 0.026 | 0.034 | 0.8 |

Team style leaks into individual numbers: a defender in a possession side passes more because of the side. Some over-representation of team-mates is expected and correct; a large lift would mean the model is partly clustering clubs.

**Read the absolute column, not the ratio.** In a wide pool - five leagues in one season - two team-mates are a vanishing share of the candidates, so `chance` is tiny and `lift` divides by it. A lift of 4 on a 1% observed share still means a top-ten list contains one-tenth of a team-mate on average, which is not a model clustering clubs. The ratio only becomes worrying when the observed share itself climbs into double figures.

## 2d. Did the market later agree? (forward test)

Not run. This needs a source with market values and at least three seasons loaded in the pool - select more seasons in the sidebar.

## 3. Is the model dominated by a few metrics?

**MID** - even share would be 0.062 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Threat (Opta index) per 90 | 0.1087 | 0.0963 | 1.54 |
| Creativity (Opta index) per 90 | 0.1021 | 0.0914 | 1.46 |
| Ball recoveries per 90 | 0.0577 | 0.073 | 1.17 |
| Expected goal involvements per 90 | 0.0935 | 0.0716 | 1.15 |
| Expected goals conceded (team, while on pitch) per 90 | 0.0577 | 0.0704 | 1.13 |
| xG per 90 | 0.0821 | 0.0698 | 1.12 |

**DEF** - even share would be 0.062 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Creativity (Opta index) per 90 | 0.1566 | 0.1255 | 2.01 |
| xA (expected assists) per 90 | 0.1047 | 0.0834 | 1.33 |
| Ball recoveries per 90 | 0.0634 | 0.0789 | 1.26 |
| Tackles per 90 | 0.0634 | 0.078 | 1.25 |
| Bonus points system score per 90 | 0.069 | 0.0716 | 1.15 |
| Influence (Opta index) per 90 | 0.0634 | 0.0712 | 1.14 |

**FWD** - even share would be 0.077 per feature; `weight_share` is what the repeatability weights alone give each metric.

| metric | weight_share | mean_distance_share | vs_even_share |
| --- | --- | --- | --- |
| Share of appearances that were starts % | 0.0994 | 0.1256 | 1.63 |
| Tackles per 90 | 0.0844 | 0.0981 | 1.28 |
| Threat (Opta index) per 90 | 0.0936 | 0.0932 | 1.21 |
| Defensive contribution actions per 90 | 0.0844 | 0.0906 | 1.18 |
| Creativity (Opta index) per 90 | 0.0914 | 0.0864 | 1.12 |
| Bonus points system score per 90 | 0.09 | 0.0851 | 1.11 |


## 4. Sensitivity of the similarity rankings

### MID - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Creativity (Opta index) per 90 | 0.778 | 0.786 |
| Threat (Opta index) per 90 | 0.778 | 0.84 |
| Goals per 90 | 0.826 | 0.877 |
| xA (expected assists) per 90 | 0.847 | 0.887 |
| Influence (Opta index) per 90 | 0.847 | 0.903 |
| xG per 90 | 0.849 | 0.877 |
| Assists per 90 | 0.86 | 0.837 |
| Expected goal involvements per 90 | 0.896 | 0.93 |

### MID - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Chance Creation | 0.672 | 0.727 |
| Defensive Work | 0.675 | 0.681 |
| Goal Threat | 0.716 | 0.754 |
| Defensive Solidity | 0.718 | 0.753 |
| Overall Rating | 0.776 | 0.811 |
| Availability | 0.814 | 0.861 |
| Build-up Involvement | 0.843 | 0.871 |

### DEF - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Ball recoveries per 90 | 0.711 | 0.678 |
| Tackles per 90 | 0.771 | 0.786 |
| Expected goals conceded (team, while on pitch) per 90 | 0.784 | 0.868 |
| Defensive contribution actions per 90 | 0.835 | 0.884 |
| xA (expected assists) per 90 | 0.844 | 0.869 |
| Clearances, blocks and interceptions per 90 | 0.857 | 0.913 |
| xG per 90 | 0.888 | 0.938 |
| Goals per 90 | 0.956 | 0.978 |

### DEF - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defensive Work | 0.615 | 0.679 |
| Chance Creation | 0.662 | 0.694 |
| Goal Threat | 0.703 | 0.773 |
| Overall Rating | 0.756 | 0.817 |
| Availability | 0.764 | 0.811 |
| Defensive Solidity | 0.778 | 0.823 |
| Build-up Involvement | 0.797 | 0.841 |

### FWD - removing one metric

| removed | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Creativity (Opta index) per 90 | 0.86 | 0.883 |
| Expected goal involvements per 90 | 0.884 | 0.927 |
| Threat (Opta index) per 90 | 0.891 | 0.892 |
| Influence (Opta index) per 90 | 0.911 | 0.944 |
| xG per 90 | 0.916 | 0.927 |
| Goals per 90 | 0.917 | 0.945 |
| Assists per 90 | 0.952 | 0.954 |
| xA (expected assists) per 90 | 0.963 | 0.954 |

### FWD - tripling the weight on one category

| category_weighted_x3 | top_k_overlap | rank_correlation |
| --- | --- | --- |
| Defensive Work | 0.705 | 0.628 |
| Availability | 0.756 | 0.783 |
| Goal Threat | 0.773 | 0.819 |
| Chance Creation | 0.795 | 0.769 |
| Overall Rating | 0.854 | 0.901 |
| Build-up Involvement | 0.883 | 0.916 |

`top_k_overlap` is the Jaccard overlap of the top ten before and after the change; `rank_correlation` is the Spearman correlation of the survivors' ordering. A metric whose removal drops the overlap below ~0.5 is effectively steering that position's model.

## 5. Correlated features

**MID** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| xA (expected assists) per 90 | Creativity (Opta index) per 90 | 0.882 |
| xG per 90 | Expected goal involvements per 90 | 0.873 |
| Influence (Opta index) per 90 | Bonus points system score per 90 | 0.858 |

**DEF** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| Clearances, blocks and interceptions per 90 | Defensive contribution actions per 90 | 0.954 |
| xA (expected assists) per 90 | Creativity (Opta index) per 90 | 0.863 |

**FWD** - pairs with |r| >= 0.85:

| metric A | metric B | r |
| --- | --- | --- |
| xG per 90 | Expected goal involvements per 90 | 0.966 |
| Influence (Opta index) per 90 | Bonus points system score per 90 | 0.936 |
| Goals per 90 | Influence (Opta index) per 90 | 0.92 |
| Goals per 90 | Bonus points system score per 90 | 0.886 |

Highly correlated features double-count one idea inside a Euclidean distance. They are reported rather than silently dropped, because for a scout 'progressive passes' and 'passes into the final third' are different questions even when they move together.

## Known limitations

- This pool is **Premier League 2016/17-2025/26 (FPL + Understat)** - real players, real seasons. See the caveats on the Home page and in `src/config.py` for exactly what this source does and does not measure.
- League strength coefficients are editable assumptions in `src/config.py`, not measured quantities. Every score that uses them says so.
- One season of finishing (goals minus xG) is noisy and is treated as descriptive only.
- Positions come from the dataset. A player who changed role mid-season is compared against the peer group of his listed position, which will understate him.
- The Hidden Gem score contains no fee or wage data and is therefore not a valuation.