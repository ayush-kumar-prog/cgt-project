# COMPREHENSIVE PROMPT: Create COMP34612 Computational Game Theory Presentation

## MISSION
Create a presentation for COMP34612 Computational Game Theory group project. This presentation is worth **40% of the project mark** (which is 50% of the entire course unit = 20% of final grade). The goal is to MAXIMISE the rubric score. Every slide must earn its space. Every element must serve the grade.

## THE RUBRIC (from the spec, word for word)
The presentation is graded on **CONTENT ONLY** (presentation skills are NOT evaluated):
1. **Originality of your idea** — going beyond the textbook OLS+RLS approach
2. **How well it integrates knowledge from the course** — must reference Lectures 1-7 concepts explicitly
3. **Any relevant external literature or research** — cite papers, dual control theory, online learning
4. **Activities your group undertook and lessons learned** — the journey, experiments, failures, iterations
5. **Each group member must actively participate in speaking and answering questions** — this is how individual marks are distributed

## THE PROJECT (what we built)
**Task**: Build a program that plays a repeated 2-person Stackelberg pricing game as the leader, under imperfect information.

**Game Structure**:
- Leader announces price u_L first, follower responds with u_F
- 100 days historical data, then 30 days of gameplay (days 101-130)
- Leader demand: S_L(u_L, u_F) = 100 - 5*u_L + 3*u_F
- Leader daily profit: (u_L - 1) * S_L, unit cost c_L = 1
- Objective: maximize accumulated 30-day profit
- 3 visible followers (MK1, MK2, MK3) + 3 hidden variants (MK4, MK5, MK6)
- Follower payoff functions are UNKNOWN; parameters may change over time
- Strategy spaces: [1, +inf) for MK1/MK2/MK4/MK5; [1, 15] for MK3/MK6

**Our Solution** (final code: 124 lines, `game/src/leaders.py`):
- `AdaptiveLeader` for MK1/MK2/MK4/MK5 (unbounded)
- `BoundedAdaptiveLeader` for MK3/MK6 (capped at 15)
- Both use a shared `_AdaptiveCore` mixin with:
  1. Historical data loading (100 days via `get_price_from_date()`)
  2. MAD-based outlier filtering
  3. Time trend detection (correlation > 0.7 triggers time regressor)
  4. OLS reaction function fitting: u_F = alpha + beta*u_L [+ gamma*t]
  5. RMSE/MAPE/R² metrics computation
  6. Adaptive exploration: probe at u_L=12 on day 101 to learn slope outside historical range [1.72, 1.90]
  7. Hybrid OLS+RLS online updating (batch refit every 3 days, RLS between)
  8. Stackelberg-optimal price with safety bounds and steep-slope ramp-up
  9. Forgetting factor lambda=0.97

**Performance Results**:
| Follower | TEST mode | MARK mode (20 runs) | Benchmark |
|----------|:---------:|:-------------------:|:---------:|
| MK1 | 28,841 (99.2%) | 98.8% ± 1.1% | 29,061 |
| MK2 | 31,571 (99.4%) | 99.6% ± 0.1% | 31,748 |
| MK3 | 16,030 (100.0%) | 98.5% ± 0.5% | 16,034 |
| **Avg** | **99.5%** | **99.0%** | |

**Benchmark Comparison** (why our approach matters):
| Leader | MK1 | MK2 | MK3 | Avg |
|--------|:---:|:---:|:---:|:---:|
| **AdaptiveLeader (ours)** | 99.2% | 99.4% | 100.0% | **99.5%** |
| NaiveLeader (textbook OLS) | 81.1% | 75.4% | 99.5% | **85.3%** |

**The key insight other groups miss**: Historical data only has u_L in [1.72, 1.90]. The true optimal prices are u_L ≈ 20 for MK1/MK2 and u_L ≈ 11 for MK3. Without exploration, you extrapolate from a 0.18-wide range to a price 10x higher. Our adaptive exploration on day 101 discovers the true slope and converges to optimal in 2-3 days.

**Version Evolution** (for the "journey" narrative):
```
v1 (baseline):          98.2%  — OLS + RLS, probe at u_L=5
v2 (+better explore):   99.3%  — probe at u_L=10, batch refit every 5 days
v3 (+time trend):       99.3%  — auto-detect trending followers
v4 (simplified):        99.5%  — probe at u_L=12, refit every 3 days, cleaner code
v5 (+robustness):       99.5%  — outlier filter, RMSE/MAPE metrics, bounded variant
v6 (+GUI fix):          99.6%  — mixin pattern for Colab dropdown compatibility
v7 (+steep slope fix):  99.6%  — gradual ramp-up for beta > 5/3 edge case
```

**Stress Test Results** (18 adversarial scenarios):
- 17/18 score ≥ 99%, avg 99.5%
- Handles: negative slopes (100%), different trends (99.1-99.7%), nonlinear reactions (99.9-100%), high noise (99.5%)
- Only weakness: very steep slope beta=2.0 (97.2%) — fixed in v7 with ramp-up

**Things We Tried That Didn't Work** (for the "failures" section):
- Lambda=0.99 (increased MK1 variance from 245 to 390)
- OLS-only without RLS (higher variance, 98.8% vs 99.0%)
- 2-day exploration probe (lost more in exploitation than gained in precision)
- Thompson Sampling (identical performance to deterministic approach in this setting — 99.5% both)

## COURSE INTEGRATION POINTS (must reference these explicitly)
- **Lecture 1**: Stackelberg game framework, two-step problem, reaction functions
- **Lecture 3**: Linear reaction functions justified by linear demand models
- **Lecture 4**: Learning under imperfect information via OLS, two-step approximate Stackelberg strategy, model selection (if linear poor, try polynomial/NN)
- **Lecture 5**: RMSE, MAE, MAPE, R² for model accuracy evaluation
- **Lecture 6**: RLS with forgetting factor for time-varying environments, moving window vs RLS comparison, weighted RMSE/MAPE
- **Lecture 7**: Best reaction functions, Stackelberg equilibrium derivation, existence theorems

## EXTERNAL LITERATURE TO CITE
- Feldbaum (1960s): Dual control theory — certainty-equivalence with cautious probing
- Letchford et al. (2009): Learning and approximation in Stackelberg games
- Balcan et al. (2015): Commitment without regret: Online Stackelberg games
- Thompson (1933): On the likelihood that one unknown probability exceeds another (Thompson Sampling)
- Bar-Shalom & Tse (1974): Dual effect, certainty equivalence and separation in stochastic control

## INSPIRATION: HEX PROJECT PRESENTATION ANALYSIS
We previously did a similar project (AI & Games, Hex agent) that scored ~75% on presentation and 100% on journal. The full Hex project is at:
- **Repository**: `/Users/kumar/Documents/University/Year3/AI_games/COMP34111-AI-Games-Hex/`
- **Presentation PDF**: `/Users/kumar/Documents/University/Year3/AI_games/COMP34111_presentation.pdf`
- **Also at**: `/Users/kumar/Documents/University/Year3/AI_games/COMP34111-AI-Games-Hex/journal/presentation.pdf`
- **Journal**: `/Users/kumar/Documents/University/Year3/AI_games/COMP34111-AI-Games-Hex/journal/`

### What worked in the Hex presentation (keep doing):
- Strong chronological journey narrative (10 phases)
- Extensive empirical evidence (200+ tournament games)
- Honest treatment of failures (4 approaches that failed)
- Code snippets demonstrate technical depth
- Every claim backed by data tables
- "Feature | Why" tables to justify decisions
- Literature references integrated into decision narrative
- Terminal output screenshots for authenticity

### What to improve for CGT (push above 75%):
- **Fewer slides** (15 max, not 35)
- **Less raw code** (3-5 focused snippets, not 15+)
- **Add visualizations** (the Hex presentation had ZERO diagrams — we need time series plots, architecture diagrams, convergence charts)
- **Don't front-load the conclusion** in executive summary
- **Consolidate early phases** into 1-2 slides
- **Add comparison benchmarks** (NaiveLeader vs AdaptiveLeader table)
- **Cleaner visual design** (not just walls of text and tables)

## SUGGESTED SLIDE STRUCTURE (15 slides)
1. **Title** — COMP34612, project title, group number, subtitle framing the journey
2. **Problem Overview** — Stackelberg game setup, demand model, what's known vs unknown
3. **The Naive Approach** — what textbook OLS gives (85.3%), why it fails (extrapolation gap)
4. **Course Theory** — OLS (L4) → RLS (L6) → metrics (L5), the two-step Stackelberg method
5. **The Exploration Problem** — historical data only covers [1.72, 1.90], optimal is u_L≈20
6. **Our Innovation: Adaptive Exploration** — probe at u_L=12, detect slope, converge in 2 days
7. **Time Trend Detection** — auto-detect MK2-type followers, fit u_F = α + βu_L + γt
8. **Online Learning: OLS + RLS Hybrid** — why hybrid beats pure RLS, forgetting factor tuning
9. **What Didn't Work** — lambda tuning, OLS-only, 2-day probe, Thompson Sampling comparison
10. **Robustness** — 18 stress scenarios, MK4/5/6 handling, steep slope fix
11. **Ablation Study** — AdaptiveLeader vs NaiveLeader vs BoundedAdaptive comparison table
12. **Final Results** — profit numbers, RMSE/MAPE metrics, convergence plot
13. **External Literature** — dual control theory, online Stackelberg learning references
14. **Key Lessons** — exploration asymmetry, robustness > precision, the 0.18-range insight
15. **Conclusion & Q&A**

## FORMATTING & DESIGN REQUIREMENTS
- **Dimensions**: Standard 16:9 widescreen slides
- **Every element must earn its space** — no decorative filler, no generic stock images
- **Visualizations needed**:
  - Price convergence plot (u_L over days 101-130 for each follower)
  - NaiveLeader vs AdaptiveLeader profit comparison bar chart
  - Architecture/flow diagram of the leader's decision pipeline
  - Exploration regret analysis chart (why probe at 12 not 5)
- **Code snippets**: 3-5 max, dark background, syntax highlighted, 5-10 lines each, focused on key innovations (exploration protocol, OLS+RLS hybrid, slope detection)
- **Tables**: clean formatting, highlight key numbers in bold/color
- **Consistent color scheme**: professional, academic
- **Font**: large enough to read from back of room

## ITERATION PROCESS (Ralph Wiggum Self-Improving Loop)
Run this loop recursively until you cannot squeeze out any more marks:
1. Generate initial presentation (PPTX + PDF)
2. Screenshot each slide
3. Compare EVERY slide against the rubric point by point
4. For each rubric criterion, score yourself 1-10 and identify gaps
5. Ask: "If I were a marker seeing 30 presentations, would this one stand out?"
6. Ask: "Does every element earn its space on this page?"
7. Check: formatting issues? overlapping text/boxes? too much empty space? too crowded?
8. Identify the weakest slide, improve it
9. Spin up agent teams to be creative — devil's advocate, design critic, content reviewer
10. Commit iteration to git, note what changed and why
11. REPEAT from step 2 until score plateaus
12. Generate final PPTX and PDF

## THE PROJECT SPEC PDF
Located at: `/Users/kumar/Documents/University/Year3/cgt/game/COMP34612 - Project - 2026.pdf`
Read this to understand the exact game rules, assessment criteria, and requirements.

## COURSE MATERIALS (for lecture references)
Located at: `/Users/kumar/Documents/University/Year3/cgt/course_details/`
Contains lecture slides (1-12) and example class questions with solutions.
Key files for presentation references:
- `lecture 4 slides.pdf` — Imperfect information, OLS learning
- `lecture 6 slides.pdf` — RLS, forgetting factor, online learning
- `Lecture 4 - Example Class Questions with Solutions.pdf` — model selection guidance
- `Lecture 6 - Example Class Questions with Solutions.pdf` — RLS vs moving window

## GIT REPOSITORY
`/Users/kumar/Documents/University/Year3/cgt/` — push all presentation iterations here.

## GROUP NUMBER
12
