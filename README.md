# DSA210-Spring2026-Project
Project for DSA

## Project overview

League of Legends is a frequently updated five-versus-five competitive video game. These updates, known as patches, can modify the relative power of champions (playable characters) and can modify the play of professional matches. This project will monitor one of the most successful professional League of Legends players, Faker, and examine whether he is more consistent or not in performance compared to a small comparison group of elite mid-lane players after significant patch releases.

## Research question

**Does Faker have a more consistent performance than a small comparison group of elite professional League of Legends mid-lane players following significant game updates, and is that associated with champion-pool diversity?** 

In this case, the **champion-pool diversity** is the number of playable characters that a player can effectively apply during professional matches.

## Sources of data and enrichment

The following analysis will be a synthesis of two sources:

- Main dataset: Elixir-style professional match exports of Oracle in data/raw/
  - game_metadata.csv
  - game_players_stats.csv
- Enrichment dataset: data/raw/champions_simple.json: metadata of Riot champions.

The match data furnishes professionalized match information like tournament, date, player, team, role, selected champion, kills, deaths, assists, gold, damage, and match result. The Riot metadata enables champions to be categorised into larger play-style groups and be associated with individual patches.

## Samples used

- Time period: 2024-01-01 to 2024-09-14
- Position: Mid lane only.
- Faker, Chovy, ShowMaker, Caps, knight, Bdd.
- Competitions that the filtered sample includes: LCK, LPL, LEC, MSI, Esports World Cup, Demacia Cup.
- Final match-level sample size: 732 player-match rows
- Patch windows studied: 14.5, 14.13, 14.16

## Data preparation and data collection

I joined match metadata and player statistics by game ID, eliminated any rows of the same player-game, and limited the sample to mid-lane players. Faker is the main subject but in comparison with the small benchmark group of other high-level professional players in the mid lane.

To examine the adaptation behavior in significant game patches I designed 28 days pre-patch and post-patch windows surrounding patch versions 14.5, 14.13 and 14.16.

I cleaned and merged the data and came up with the following derived metrics:

- KDA = (kills + assists) / max(1, deaths)
- Creep score per minute.
- Damage per minute
- Gold per minute
- Kill participation
- Uniform composite performance index.
- Champion-pool diversity
- New champion rate following a patch.
- Before and after change in performance with patching.

## Data characteristics

The last data is player-match data whereby a row is a representation of one professional player in a single match. It includes:

### Categorical variables
- player name
- team
- league/tournament
- patch version
- role
- champion name
- champion style/tag

### Numerical variables
- kills
- deaths
- assists
- gold earned
- damage dealt
- creep score
- kill participation
- damage per minute
- gold per minute
- composite performance index
- match outcome

I also generated player-patch summary tables to enable performance stability prior to and following significant updates to be assessed at a more elevated level than the match-by-match statistics.

## Exploratory data analysis (EDA)

The exploratory analysis was concerned with whether Faker seems to be more stable than peers following patch transitions and whether this could be connected to champion-pool diversity.

## EDA highlights
- The rolling performance visualization shows performance variation across the 2024 season, with visible changes around certain patch dates.
- Faker does not consistently perform best among all the players in each period, and the trend is competitive with the benchmark group.
- The mean absolute patch-change plot indicates that the post-patch changes in performance of Faker are slightly less than the average changes of the peers.
- There is no close monotonic relationship of the diversity-versus-stability plot in the present sample.
- The new-champion-rate plot indicates that Faker is more likely to present fewer champions that were not used before right after major patches compared to the peer group.

## Hypothesis tests

There were four hypothesis tests:

1. Faker exhibits lesser average changes in absolute post-patch performance as compared to peers.
2. Faker is more diverse in terms of pre-patch champion-pool than others.
3. The larger the pre-patch diversity, the smaller the post-patch performance changes.
4. Faker has less really new champions right after patches in comparison to peers.

## Results summary
- H1: directionally consistent with the hypothesis, but not statistically significant in the present sample (p = 0.436)
- H2: directionally consistent with the hypothesis, but not statistically significant in the present sample (p = 0.285).
- H3: no meaningful relationship in the current sample (rho = -0.038, p = 0.880)
- H4: had the greatest directional evidence in the current sample, although it was not statistically significant (p = 0.063).

## Key results:

- The mean change in the absolute performance of Faker after patch was 0.251 (p=0.291) in comparison to peers.
- The mean pre-patch champion diversity entropy of Faker was 2.399, which is compared to 2.197 among peers.
- The most evident pattern of the current sample is that Faker had fewer actually new champions right after patches in comparison to peers (0.228 vs 0.481) but the current sample is only near-significant.
- The tests do not yet give strong statistical support of the larger claim with the current milestone-sized sample.

## Interpretation

At this point, the evidence is favoring a tentative conclusion. Faker seems to be more of a conservative adopter of patches with a reduction in the use of entirely new champions, and his performance change is a little smaller than the average of peers. Nevertheless, the existing sample is not yet convincing statistically that Faker is more stable uniquely, due to the champion-pool diversity.

## Repository contents

### Processed data
- `data/processed/mid_players_2024.csv`
- `data/processed/player_patch_summary.csv`
- `data/processed/overall_player_summary.csv`
- `data/processed/hypothesis_tests.csv`

### Figures
- `figures/rolling_performance_2024.png`
- `figures/avg_abs_patch_change.png`
- `figures/diversity_vs_stability.png`
- `figures/new_champion_rate.png`
- `figures/style_share_faker_vs_peers.png`
- `figures/patch_delta_heatmap.png`

### Code and documentation
- `scripts/run_milestone_analysis.py`
- `requirements.txt`
- `README.md`

## Reproducing the analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis pipeline from the project root:

```bash
python scripts/run_milestone_analysis.py
```

## AI Usage

 - AI was used for python code, debugging, where to find databases, how to navigate Riot API.


