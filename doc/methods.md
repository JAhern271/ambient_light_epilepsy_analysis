# Ambient light exposure and rest–activity rhythm disruption in people with epilepsy: NHANES

> **Status:** normative specification for this study. If the code disagrees with this
> document, the code is wrong — that gap is tracked in
> [implementation-status.md](implementation-status.md), not resolved by editing this file
> to match the code.
>
> **Revised:** 2026-08-27. **Supersedes:** [archive/protocol-original.md](archive/protocol-original.md).
> **Revision in progress:** scope is being narrowed to cycle H; see
> [implementation-status.md](implementation-status.md) for what still refers to the pooled design.
>
> Citation confidence is flagged throughout; see §11.

---

## 1. Background and rationale

People with epilepsy (PWE) show disrupted rest–activity rhythms: lower interdaily stability (IS), higher intradaily variability (IV), lower relative amplitude (RA) and lower daytime activity (M10) compared with controls [Liguori_2022; Tang_2024]. These differences appear in recordings unlikely to contain seizures [Liguori_2022], which argues that circadian disruption in epilepsy is not purely a downstream consequence of seizure activity.

If disruption is not seizure-driven, something upstream is driving it. Light is the primary zeitgeber of the human circadian system, and there is precedent from another neurological population for light being the operative factor: in schizophrenia, sleep disruption was explained principally by reduced light exposure rather than by an altered intrinsic circadian period [Skeldon_2022]. Limited and indirect evidence hints that PWE may receive lower ambient light exposure [Berra_2009; Fernandez_2019], but this has never been measured directly at population scale.

NHANES 2011–2014 carries a continuously recorded ambient light channel alongside the wrist accelerometry that existing epilepsy analyses have used [NCHS_2022]. The light channel has been analysed in adults for other conditions — sleep duration and health disparities [Johnson_2023], diabetes and glycaemic control [Xiao_2023] — but not in epilepsy. This study addresses that gap.

**Positioning relative to existing NHANES work.** Four prior analyses occupy adjacent territory and establish that each component of this design is feasible on these data:

| Study | Data | What it established |
|---|---|---|
| [Tang_2024] | NHANES 2013–2014, 53 PWE / 7,410 | Rest–activity rhythm metrics are derivable and epilepsy is ascertainable |
| [Johnson_2023] | NHANES 2011–2014, n=6,089 | The light channel supports adult analysis; preprocessing rules; light-at-night distribution |
| [Xiao_2023] | NHANES 2011–2014, n=7,013 | Light and activity channels can be jointly analysed |
| [Degenfellner_2025] | NHANES 2011–2014, n=7,085 | Sleep regularity is derivable from this accelerometry |

The novel contribution here is the light channel in epilepsy specifically.

---

## 2. Aims and hypotheses

**Primary aim.** To determine whether adults with epilepsy differ from controls in objectively measured ambient light exposure.

**Secondary aim.** To determine whether light exposure accounts for part of the association between epilepsy and rest–activity rhythm disruption.

**Hypotheses**

- **H1 (daytime dose).** PWE spend less time in bright light during the day than matched controls.
- **H2 (nighttime dose).** PWE experience greater light exposure at night than controls.
- **H3 (light regularity).** PWE inhabit a less regular day-to-day light environment than controls.
- **H4 (replication).** PWE show lower IS, higher IV, lower RA and lower M10, replicating [Liguori_2022; Tang_2024].
- **H5 (attenuation).** Associations between epilepsy and rest–activity rhythm metrics attenuate when light metrics are included in the model.

**Framing note.** H1–H3 concern the *dose and regularity* of the light environment — how much light, when, and how consistently. This is deliberately distinct from *alignment* between light and activity cycles [Xiao_2023; Miller_2010]. Alignment measures are largely insensitive to absolute light level: a person who is chronically under-lit but on a consistent schedule scores as well-aligned. Since the mechanism motivating this study is insufficient zeitgeber strength [Skeldon_2022], dose is the construct of interest, and alignment measures would be blind to the effect we hypothesise. Alignment is retained only as a sensitivity analysis (§8.4).

**Estimand.** The primary estimand is the **average difference in light exposure between people with epilepsy and comparable controls within a matched sample** — an association conditional on the matched covariate set, not a nationally representative population parameter. Language throughout is framed accordingly ("epilepsy was associated with…" rather than "US adults with epilepsy have…"). A survey-weighted analysis targeting the national estimand is reported as a supplementary analysis (§8.6).

---

## 3. Design and data source

Cross-sectional analysis of NHANES 2011–2012 and 2013–2014, the two cycles carrying the Physical Activity Monitor (PAM) component.

Participants wore an ActiGraph GT3X+ on the non-dominant wrist, 24 h/day, from the day of their Mobile Examination Center (MEC) visit until the morning of the ninth day. The device recorded triaxial acceleration at 80 Hz and ambient light in lux at 1 Hz; both are released summarised to the minute [NCHS_2022]. Movement is expressed in Monitor-Independent Movement Summary (MIMS) units [John_2019]. Approximately 96% of participants with data wore the device to day 9; on non-partial days the mean was ~1,437 valid minutes of a possible 1,440 [NCHS_2022].

**No sleep diary or activity log was kept** [NCHS_2022]. This constrains sleep derivation (§6.6).

---

## 4. Cohort definition

### 4.1 Epilepsy ascertainment

NHANES 2011–2014 contains no direct epilepsy or seizure item in the medical conditions questionnaire. Cases are identified from the prescription medication file (RXQ_RX_G / RXQ_RX_H) as participants reporting a medication taken for a condition coded ICD-10-CM **G40** (epilepsy and recurrent seizures), with manual confirmation that the reported drug is a recognised antiseizure medication (ASM). This is the approach used by both prior NHANES epilepsy analyses [Terman_2020; Tang_2024].

**Why the reason-code requirement matters.** Many ASMs are widely prescribed off-label — gabapentin and pregabalin for neuropathic pain, topiramate for migraine, valproate and lamotrigine for bipolar disorder. Requiring a G40 indication rather than ASM use alone is the principal safeguard against false positives, since a gabapentin prescription for nerve pain carries a pain code, not G40. [Terman_2020] additionally excluded G40-coded medications that are not ASMs.

**Expected yield.** [Tang_2024] identified 53 cases among 7,410 in a single cycle (0.75%). Pooling both cycles and applying accelerometry validity criteria, we anticipate a case count in the region of 50–100. Precision, not sample size, is the binding constraint on this study (§9.4).

**Sensitivity analyses on case definition**
1. Narrow definition: restrict to ASMs rarely used off-label (e.g. levetiracetam, phenytoin, carbamazepine, lacosamide).
2. Broad definition: any ASM regardless of reason code.
3. Primary definition (ASM + G40) as above.

Agreement across the three is reported; disagreement is interpreted as misclassification sensitivity.

### 4.2 Controls and matching

Eligible controls are all other participants meeting the accelerometry validity criteria (§4.3). From this pool, a matched comparison group is constructed by **full matching on the propensity score** (§8.1). No exclusion by ASM use is applied under the primary definition, but a sensitivity analysis excludes control participants taking any ASM for a non-G40 indication, since these individuals may share medication-related effects without having epilepsy.

**Why matching rather than regression adjustment alone.** With an anticipated 50–100 cases against several thousand controls, the two groups will differ substantially on demographic and socioeconomic characteristics. Relying on regression alone to correct imbalance of that magnitude requires extrapolation beyond the region where cases and controls overlap, and correct specification of the covariate–outcome relationship across that range.

This is also the design convention in comparable work. Large-cohort accelerometry studies with common exposures use whole-cohort regression without matching, but studies of rare neurological conditions match: [Bailey_2023] compared 241 dystonia cases with 964 matched controls in UK Biobank. The case count here is smaller still, placing this study firmly in the matched camp.

**Why matching is particularly important for this exposure.** Ambient light exposure is strongly socially and seasonally patterned. [Johnson_2023] found nighttime light exposure differed markedly by race/ethnicity — 83% of Black participants exposed versus 63% of White participants — and that its association with sleep varied by race/ethnicity and sex. Season is plausibly the single largest determinant of daytime light exposure in the dataset. Matching on these variables means comparing people whose devices were worn at comparable times of year and in comparable social circumstances, rather than asking a model to correct large imbalances after the fact.

**Note on [Bailey_2023] as precedent.** That study matched on **age and gender only** and then compared groups using unadjusted univariate tests (chi-square, t-test, Mann–Whitney U with Bonferroni correction). The design structure is a useful precedent; the implementation is not adopted here. Age and sex alone would leave light exposure confounded by most of what actually determines it.

**Matching variables:** age, sex, race/ethnicity, poverty–income ratio, education, employment status, season of examination (RIDEXMON).

### 4.3 Inclusion criteria

- Aged ≥20 years at examination (following [Xiao_2023]; note [Tang_2024] included ages 3+, limiting direct comparability)
- Not pregnant at the time of examination
- Valid accelerometry per §5.2
- Non-missing epilepsy ascertainment data

### 4.4 The ASM circularity problem

ASMs independently affect alertness, sleep architecture and activity levels, and are also the variable defining case status. Adjusting for "any ASM use" would adjust away the exposure.

**Resolution.** The case–control contrast is presented as the total effect of **treated epilepsy**, and the manuscript states plainly that medication effects are part of this estimand rather than a confounder removed from it. Medication burden (polytherapy count; use of specifically sedating agents such as benzodiazepines or barbiturates) is examined as a covariate and potential effect modifier *within* the epilepsy group only, in a secondary analysis.

---

## 5. Preprocessing

### 5.1 Minute-level exclusions

A minute is set to missing for both light and activity if any of the following apply, following [Johnson_2023]:

1. The NHANES data quality flag variable contains any letter value
2. The NHANES-provided prediction variable (PAXPREDM) classifies the minute as non-wear
3. Activity value is negative or flagged as uncomputable

Light and activity are masked jointly rather than separately, so that the two channels are always derived from the same set of retained minutes. This prevents a participant contributing light data from minutes excluded from their activity metrics.

### 5.2 Day-level validity

The first and last days of wear are partial by protocol [NCHS_2022] and are dropped. Days are defined noon-to-noon, following [Johnson_2023; Su_2022], so that a night's sleep falls within a single analytic day rather than being split across two.

**There is no consensus criterion in this literature.** Published rules on these same data differ materially, and the direct epilepsy precedent has none at all:

| Study | Rule |
|---|---|
| [Tang_2024] | **No valid-day or wear-time rule stated.** Minutes set to missing only where PAXMTSM = −0.01 or where the device failed to capture triaxial values continuously. No minimum days, no wear-time threshold, no PAXPREDM non-wear masking, no quality-flag exclusion. |
| [Johnson_2023] | 5 consecutive full days; day invalid if >4 h missing or total daily activity <200; exclude participants with ≥1 invalid day |
| [Xiao_2023] | ≥4 valid days; valid day = ≥20 h of valid recording |
| [Su_2022] | ≥3 valid days with >16 h wear |

**Primary rule adopted:** ≥4 valid days, valid day defined as ≥20 h of retained minutes, following [Xiao_2023]. Rationale: 20 h is a stricter wear requirement than 16 h, which matters because light metrics are sensitive to gaps; requiring 4 rather than 5 consecutive days retains more of a small case group than [Johnson_2023]'s rule would. The absence of any rule in [Tang_2024] is noted but not followed — without wear-time filtering, non-wear periods are indistinguishable from genuine inactivity, which biases IS, IV, RA and L5 in unpredictable directions and would be more damaging still for light metrics, since a device in a drawer reads near-zero lux.

**Sensitivity:** the [Johnson_2023] and [Su_2022] rules are applied as alternatives and results reported.

### 5.3 A note on differing inclusion sets

Rest–activity metrics and light metrics use the same retained-minute set and therefore the same analytic sample. Sleep metrics (§6.6), being a secondary analysis, may have a slightly different valid-night requirement; where this occurs, the sleep analysis is reported on its own clearly-stated subsample and the primary analysis is not re-run on it.

---

## 6. Measures

### 6.1 Light exposure windows

**Fixed clock-time windows are used: day 07:00–19:00, night 23:00–06:00.** These boundaries are pre-specified and are not revised after analysis begins. Hours around dawn and dusk (19:00–23:00, 06:00–07:00) are excluded from both windows rather than assigned to one.

**Why clock-time rather than individually anchored windows.** Two alternatives exist. Sleep-anchored windows (light in the hours before sleep onset, during the sleep period, after wake) adapt to individual timing but require sleep timing to define the exposure — which conflicts directly with treating sleep as an outcome. L5-anchored windows [Johnson_2023] avoid that specific conflict, but introduce another: L5 is derived from the activity signal, and if PWE have flatter, more fragmented rhythms — precisely our hypothesis — then the L5 window is less well-defined in cases than in controls, giving differential measurement error in the group hypothesised to be affected. L5 is also defined as the *least active* window, and stillness in a dark room produces both low activity and low lux, so L5-light is partly shaped by its own selection criterion.

Fixed clock windows have neither problem. Their cost is that they conflate dose with timing: if PWE go to bed later, some apparent "nighttime light" reflects a shifted schedule rather than more light at a given circadian phase.

**Mitigation.** Sleep midpoint and L5 start time are reported descriptively by group. If timing is comparable across groups, the conflation objection is empirically defused. If it differs, this is stated and the light findings interpreted accordingly. An L5-anchored nighttime light metric is additionally reported as sensitivity, permitting direct comparison with [Johnson_2023].

**Governing principle:** any accelerometer-derived variable may serve as an exposure, a window definition, or an outcome — but not two of these simultaneously in the same model.

### 6.2 Daytime light metrics

**Primary: time above threshold** — minutes per day above 100 lux, above 250 lux, and above 1,000 lux, averaged across valid days.

**Why not mean daytime lux.** The NHANES light sensor is top-coded: *"Values at or above 2,500 lux were coded as '2,500 lux'"* [NCHS_2022]. CDC's own reference table in the same document places full daylight at 10,000–25,000 lux. Every moment of genuine outdoor daylight is therefore recorded as an identical value, and the ceiling sits an order of magnitude below the top of the real range. Since H1 concerns reduced *daytime* light, the censoring bites hardest exactly where the hypothesised group difference should be largest. Mean daytime lux is a mean of a censored variable and is not a defensible primary metric here.

**Why threshold counts survive the censoring.** Top-coding affects the recorded *magnitude* of bright exposures, not whether a threshold below the ceiling was crossed. All three thresholds sit below 2,500 lux, so minutes-above-threshold is measured without censoring bias; what is lost is only the ability to distinguish exposure just above a threshold from exposure far above it. This is the principal reason for preferring threshold counts to means.

**Rationale for the three thresholds.** They span the range of biologically and behaviourally meaningful boundaries: 100 lux approximates the transition from dim to ordinary indoor lighting; 250 lux corresponds to published daytime recommendations [Brown_2022]; and 1,000 lux approximates CDC's own reference value for an overcast day [NCHS_2022], making it a rough proxy for time spent outdoors or in daylit space. Reporting all three characterises the *shape* of the daytime exposure distribution rather than a single point on it — if PWE differ at 1,000 lux but not at 100, the deficit is in outdoor time rather than general indoor dimness, which is directly relevant to interpretation (§10.1).

**Secondary daytime metrics**
- Proportion of daytime minutes at the 2,500 lux ceiling (an uncensored measure of time in bright light)
- Mean daytime lux, reported for comparability with other literature but explicitly caveated as censored
- Brightest continuous 10 h of light exposure (light analogue of M10)

### 6.3 Nighttime light metrics

**Primary: categorical** — none (0 lux) / low / high, splitting low from high at the median value among participants with any non-zero exposure, following [Johnson_2023].

**Why categorical rather than continuous.** Nighttime light in these data is severely zero-inflated. In [Johnson_2023], 34% of participants recorded exactly 0 lux across their entire L5 window, and the median among those with any exposure was 0.22 lux. This distribution — a large point mass at zero, a dense cluster below 1 lux, and a long thin tail — breaks continuous modelling in three ways: log transformation is undefined at zero and log(x+1) disproportionately distorts the sub-1-lux region where most data sit; no standard regression error structure fits a mixture of a point mass and a skewed continuum; and exact zeros may represent a different phenomenon (sleeve or bedding occlusion) rather than a point on the same continuum as 0.5 lux.

Categorisation preserves the "any light at all" contrast, which may be the biologically meaningful one, and makes results directly comparable with [Johnson_2023]. The cost is lost resolution and a data-driven cutpoint.

**Sensitivity:** a two-part (hurdle) model — logistic for any-versus-none, continuous model among the exposed — which respects the mixture structure and permits the two processes to have different predictors.

### 6.4 Light regularity and within-day light variability

**Primary: interdaily stability computed on the light signal** rather than the activity signal, quantifying how consistent the 24-h light profile is from day to day [Wallace_2023].

*Why this is included.* H1 and H2 concern dose; a person may receive adequate light on average while inhabiting an erratic light environment. Light regularity captures a property of the environment that dose alone misses, and — critically — it remains a property of the *light* signal, so it does not compromise the exposure/outcome separation. It is the most direct available operationalisation of "environment → rhythms" in the causal model.

**Exploratory: intradaily variability computed on the light signal**, quantifying within-day fragmentation of the light profile. Together with day–night light contrast (§6.2, the analogue of relative amplitude), this completes the nonparametric triad on the light signal and mirrors the metrics computed on activity (§6.5).

*Why this is exploratory rather than a primary endpoint.* Two reasons.

First, **the direction of the hypothesis is not specifiable in advance.** For activity, high IV unambiguously indicates a fragmented rest–activity pattern and is predicted to be higher in PWE. Applied to light, high IV indicates frequent within-day changes in illumination — which is largely what repeatedly moving between indoor and outdoor environments produces. A person who steps outside several times a day has a choppier light profile than someone under constant indoor lighting from morning to evening. High light-IV may therefore indicate a varied and healthy light environment, or a genuinely erratic one, and these cannot be distinguished by the metric itself. A directional hypothesis would not be defensible.

Second, **the metric is highly sensitive to transformation.** Computed on raw lux it is dominated by the day–night step change and largely recapitulates day–night contrast; computed on log lux it captures something substantively different. The transformation used is stated explicitly, and both are reported.

Light IV is therefore reported descriptively and interpreted alongside the threshold metrics rather than tested as a hypothesis. If PWE show higher light-IV *together with* less time above 1,000 lux, the pattern suggests erraticness rather than beneficial variety; if higher light-IV accompanies *more* bright-light time, the opposite reading applies. The interpretation is conditional on the dose metrics by design.

### 6.5 Rest–activity rhythm metrics (outcomes)

Nonparametric metrics computed from minute-level MIMS: **IS, IV, RA, M10, L5**, plus M10 and L5 start times. Derived using the nparACT R package [Blume_2016], as in [Tang_2024].

**Why nonparametric rather than cosinor.** Cosinor assumes a sinusoidal waveform, which is frequently violated in fragmented rest–activity profiles — the profile shape we hypothesise to characterise PWE. Nonparametric metrics make no waveform assumption. Extended cosinor is reported as a sensitivity analysis only.

**Comparability caveat.** IS/IV/RA computed on MIMS units are not numerically identical to those computed on ActiGraph counts in the older actigraphy literature. Absolute values are not comparable across devices; only within-study group contrasts are interpreted.

### 6.6 Sleep (secondary)

Sleep duration and sleep midpoint derived from the NHANES-provided PAXPREDM minute classification, summing sleep-classified minutes within each noon-to-noon day, following [Johnson_2023; Su_2022]. Implausible values (<3 h or >13 h) are excluded [Johnson_2023].

**Why sleep is secondary rather than a co-primary outcome family.** Sleep is strongly intercorrelated with the rest–activity metrics already in the model, so it adds a third correlated outcome family that consumes multiplicity budget without adding proportionate independent information. It also cannot be causally ordered relative to circadian disruption in cross-sectional data (§8.2). Its value here is descriptive and as a bridge to the sleep–seizure literature that motivates the study, not as an independent test.

**Limitation to state.** PAXPREDM is an algorithmic prediction, not a validated sleep measure; sleep/wake classification on this signal is imperfect [Thapa-Chhetry_2022]. Self-reported sleep duration (SLD010H) is modelled alongside as an auxiliary variable; agreement or disagreement between self-report and accelerometry is itself informative given that PWE may systematically misreport sleep.

### 6.7 Covariates

**Confounders (adjusted for):** age, sex, race/ethnicity, poverty–income ratio, education, employment status, season of examination (RIDEXMON).

*Note on smoking.* Smoking is a conventional covariate in comparable NHANES accelerometry analyses but is **not** included here. It is not a cause of epilepsy, so it does not satisfy the confounding criterion of being a common cause of exposure and outcome; its plausible channels — socioeconomic position and lifestyle — are already captured by poverty–income ratio, education and employment. It may also lie on a causal pathway rather than confound one, since smokers step outdoors to smoke and so plausibly receive *more* daytime bright light, making the direction of any adjustment bias unclear. It is reported as a sensitivity analysis (§8.4).

**Not adjusted for in primary models (potential mediators):**
- **Physical activity / M10** — plausibly on the causal path from light to rhythm disruption; adjusting induces overadjustment bias
- **Sleep** — same problem, in both possible orderings
- **BMI** — plausibly downstream of rhythm disruption
- **Depression (PHQ-9)** — **treated as a mediator in this study.** The assumed structure is epilepsy → depression → reduced outdoor activity → reduced light exposure. It is therefore excluded from both the matching set (§4.2) and the primary model, and entered only in clearly-labelled secondary models. This is a stated assumption rather than an established ordering: depression may alternatively precede and confound the epilepsy–light relationship, and cross-sectional data cannot distinguish these. The assumption is made explicit in the DAG accompanying the manuscript

[Xiao_2023] treats sleep, physical activity and BMI the same way, entering them only in explicitly-labelled sensitivity models on the grounds that they are simultaneously potential confounders and mediators.

---

## 7. Sample size and power

No formal a priori sample size calculation is possible, as the case count is fixed by the survey. With an anticipated 50–100 cases, power is dominated by the case count, and the study can reliably detect only moderate-to-large standardised differences. Full matching retains all cases (§8.1), which preserves what precision is available; the effective sample size after matching is reported and used in the power statement. Findings are framed as hypothesis-generating rather than definitive. The supplementary survey-weighted analysis (§8.5) will have lower precision still, because of the design effect.

---

## 8. Statistical analysis

### 8.1 Primary analysis: full matching with double adjustment

The primary analysis proceeds in two stages — matching, then adjusted regression within the matched sample. This is **double adjustment**: matching removes the bulk of covariate imbalance, and regression removes the residual imbalance that matching leaves behind. Empirically this reduces bias more than either alone [DuGoff_2014].

**Stage 1 — Full matching.**

A propensity score for epilepsy case status is estimated by logistic regression on the §4.2 matching variables. Participants are then partitioned by **full matching**, in which the sample is divided into strata each containing either one case with several controls or one control with several cases, chosen to minimise total within-stratum propensity-score distance. Each participant receives a matching weight reflecting their stratum composition.

*Why full matching rather than fixed-ratio k:1 matching.* Fixed-ratio matching with a caliper discards cases that find no control within the caliper. With 50–100 cases this is unaffordable, and the losses are not random: the cases dropped are the most demographically unusual, so their exclusion is an uncontrolled selection process. Full matching retains every case, allows stratum size to vary with the local density of good matches, and generally achieves better covariate balance because it is never forced into poor matches in sparse regions. It requires no arbitrary choice of ratio.

*Alternative.* Overlap weighting is a reasonable substitute: it achieves exact balance on all covariates in the propensity model, retains all participants, and cannot produce extreme weights. Its estimand — the effect in the population whose characteristics could plausibly belong to either group — is arguably closer to the question of interest but harder to describe. It is reported as a sensitivity analysis (§8.5).

**Stage 2 — Balance assessment.**

Standardised mean differences (SMDs) are computed for every matching variable before and after matching, with SMD < 0.1 taken as adequate balance. Variance ratios and propensity-score overlap are also inspected. **If balance is inadequate, the matching specification is revised and the process repeated.** This iteration occurs entirely blind to outcome data and is therefore not a form of outcome-dependent model selection — a genuine advantage of matched designs over regression adjustment alone, and worth stating explicitly in the manuscript. A balance table (pre- and post-matching) is reported.

The **effective sample size** is reported rather than the raw matched control count, since variable stratum sizes mean these differ.

*Fallback if the propensity model is unstable.* With 50–100 cases, a propensity model containing seven covariates sits at the edge of the conventional guide of roughly ten events per variable. If the model produces extreme scores or poor overlap, the fallback is exact or coarsened matching on the strongest confounders — season and sex exactly, age in five-year bands — with the remaining covariates handled by regression in Stage 3. This is a legitimate design choice at small sample size, not a concession, and is more robust than a strained propensity model.

**Stage 3 — Weighted regression within the matched sample.**

Light metrics are regressed on epilepsy status, **weighted by the matching weights** (essential: variable stratum sizes make unweighted regression on a full-matched sample invalid), with the matching variables re-entered as covariates. The matched and modelled covariate sets are therefore identical, which is the double-adjustment logic in its cleanest form. Robust or cluster-robust standard errors account for the matched structure.

Continuous outcomes: weighted linear regression. Categorical nighttime light: weighted multinomial or Poisson-with-robust-variance regression, the latter following [Johnson_2023], who note that Poisson yields less biased prevalence ratios than logistic or log-multinomial models for common outcomes.

**Nested model sequence.** Following the structure used in large accelerometry cohort analyses [LancetHL_2023]:

| Model | Adjustment |
|---|---|
| 0 | Age, sex (minimally adjusted) |
| 1 | **Primary model** — adds race/ethnicity, poverty–income ratio, education, employment, season |
| 2 | Adds sleep duration and physical activity, to test whether associations are independent of these |
| 3 | Adds the remaining rest–activity parameters (IS, IV, RA), to test whether associations are independent of the wider rhythm profile |

Models 2 and 3 are explicitly labelled as adjusting for potential mediators (§6.7) and are interpretive rather than confounding-control models. Collinearity across the full covariate set is examined by Spearman correlation before fitting Model 3 [LancetHL_2023].

### 8.2 Handling of clock-time variables

**All clock-time variables — L5 start, M10 start, sleep midpoint, bedtime, wake time, light acrophase — are analysed using circular statistics** (circular means and dispersion; Watson–Williams or equivalent for group comparison; R package `circular`). Alternatively, times may be anchored to a reference point distant from the data's modal cluster (e.g. hours since noon), which is one reason noon-to-noon days are used (§5.2).

*Why this warrants a dedicated section.* Clock time is circular: arithmetic averaging of times that straddle midnight produces meaningless results. The mean of 23:00 and 01:00 is midnight, but averaged arithmetically it is 11:00. Sleep-related timings cluster precisely at the wraparound point, so naive averaging inflates standard deviations and drags means toward mid-day.

**This error is present in both directly relevant precedents.** [Bailey_2023] reports a control-group M5 time of 03:47 and an L5 standard deviation of 6 h 38 min, which are not credible and are internally inconsistent with the 19-minute between-group bedtime difference in the same table. [Tang_2024] reports a mean L5 start time of 10:16:32 with a standard deviation of 968 minutes, and group means of 500 and 617 minutes (i.e. mid-morning) in Table 2 — but L5 is by definition the least active five hours of the day and cannot be centred at ten in the morning. Both are the signature of arithmetic averaging across midnight.

This is not a peripheral concern for the present study: **M10 start time is one of [Tang_2024]'s headline findings** (highest vs lowest quartile OR = 3.13, 95% CI 1.39–7.58), and it is a clock-time variable derived the same way. Any replication of that result must handle the circularity correctly, and a discrepancy from Tang's estimate may reflect their method rather than a true difference.

### 8.3 Attenuation analysis (H5), not mediation

Rest–activity metrics are regressed on epilepsy status with and without light metrics, within the matched sample, and the change in the epilepsy coefficient is reported.

**This is reported as attenuation, not mediation.** Cross-sectional estimates of indirect effects can be substantially biased relative to the underlying longitudinal process [Maxwell_Cole_2007]. More fundamentally, one week of cross-sectional data cannot distinguish light → circadian → sleep from light → sleep → circadian, or from a reciprocal structure; these orderings are observationally equivalent here. The manuscript therefore reports the percentage attenuation of the epilepsy coefficient and states explicitly that this is consistent with, but does not establish, a mediating role for light.

E-values are computed for the primary associations to quantify how strong unmeasured confounding would need to be to explain them away [VanderWeele_Ding_2017].

### 8.4 Sensitivity analyses

1. **Case definition:** narrow / primary / broad (§4.1)
2. **Valid-day rules:** [Johnson_2023] and [Su_2022] alternatives (§5.2)
3. **Window definition:** L5-anchored nighttime light alongside clock-anchored (§6.1)
4. **Distributional:** hurdle model for nighttime light (§6.3)
5. **Matching method:** overlap weighting, and fixed-ratio 4:1 caliper matching (the [Bailey_2023] ratio), as alternatives to full matching
6. **Control definition:** excluding controls taking any ASM for a non-G40 indication (§4.2)
7. **Smoking:** adding smoking status to the primary model, to confirm that its exclusion (§6.7) does not alter conclusions
8. **Survey-weighted analysis:** supplementary, targeting the national estimand (§8.5)
9. **Light–activity coupling:** phasor magnitude and angle [Xiao_2023; Rea_2008], reported as a check against the shared-device criticism (§10.2) rather than as a test of the study hypothesis. If coupling is similar between groups while light dose differs, this supports the interpretation that PWE are not temporally misaligned but are receiving less light.
10. **Age stratification:** relevant to the photosensitivity alternative explanation (§10.1)

### 8.5 Supplementary analysis: survey-weighted regression

Reported in supplementary material, targeting a national rather than sample-level estimand.

NHANES uses a complex, stratified, multistage probability design with deliberate oversampling of specific demographic groups, so unweighted estimates describe this sample rather than the US population, and ignoring the clustering produces standard errors that are too small.

- **Weight:** WTMEC2YR, halved for the 4-year combined sample. The accelerometer was distributed at the end of the MEC visit, so the MEC weight is the correct "least common denominator". CDC has not released a PAM-specific weight; the PAXLUX documentation directs analysts to the examined-sample weights [NCHS_2022]. Note that [Tang_2024] used the interview weight (WTINT2YR), which is the wrong denominator for an examination-based measure.
- **Design:** SDMVSTRA (strata), SDMVPSU (PSU), Taylor linearisation. In R: `svydesign(ids=~SDMVPSU, strata=~SDMVSTRA, weights=~wt4yr, nest=TRUE)` [Lumley_2010; Leroux_2019].
- **Degrees of freedom:** design df = (number of PSUs − number of strata), typically ~30, not the sample size. With 50–100 cases distributed across PSUs, some strata may contain cases in only one PSU, breaking variance estimation. Stratum collapsing is applied where necessary and documented; estimates that are not estimable are reported as such rather than forced.
- **Model:** the §8.1 Model 1 specification, survey-weighted, on the full eligible sample.

**Why this is supplementary rather than primary.** Survey weights and matching pursue incompatible objectives — weights make the sample resemble the US population, matching makes controls resemble the (demographically unrepresentative) cases — and combining them is contested [DuGoff_2014]. With 50–100 cases, the variance inflation from the design effect is unaffordable as a primary analysis, and subgroup estimates would frequently not be estimable. The matched analysis answers the study question with acceptable precision; the weighted analysis establishes whether the finding is robust to the choice of estimand.

**Reporting rule, fixed in advance.** Agreement or disagreement between the matched and weighted analyses is reported in one sentence in the main text regardless of direction, and the matched result is the one led with. This is specified before analysis so that the choice is not made after seeing both.

**Descriptive tables.** Table 1 is presented survey-weighted for descriptive purposes even though the primary analysis is unweighted, following common practice.

### 8.6 Multiplicity

Light metrics are treated as one outcome family and rest–activity metrics as another. Within each family, a small number of pre-specified primary endpoints are designated (one daytime light, one nighttime light, one regularity; IS, IV, RA, M10 for rhythms) with false discovery rate control. Remaining metrics are exploratory and labelled as such.

**Designating the daytime primary.** Three daytime thresholds are reported (§6.2), but only one is designated the primary endpoint; the other two are reported as supporting detail characterising the distribution. **Time above 1,000 lux is proposed as the primary**, since it most directly operationalises H1's outdoor-light hypothesis and is the threshold least confounded by ordinary indoor illumination. This designation must be fixed before analysis — choosing whichever threshold shows the largest difference would be a form of selective reporting.

Given the strong intercorrelation among rhythm metrics, dimensionality reduction via functional principal components analysis is available as an alternative [Xiao_2022b].

### 8.7 Missing data

Extent and pattern of missingness in covariates reported. Where <5%, [Xiao_2023]'s approach (mode for categorical, median for continuous) is one published option; multiple imputation is preferred where missingness is greater and is reported as sensitivity regardless.

---

## 9. Software

R (version to be stated). Matching: `MatchIt` (`method = "full"`) with `WeightIt` for the overlap-weighting sensitivity analysis; balance diagnostics via `cobalt`. Survey design for the supplementary analysis: `survey` [Lumley_2010]. Rest–activity metrics: `nparACT` [Blume_2016]. Circular statistics for clock-time variables: `circular` (§8.2). Sleep derivation: GGIR, using the diary-free algorithm of [vanHees_2018], as in [Bailey_2023]. Light metric extraction: `LightLogR` [Zauner_2025], with the caveat in §10.4. Full analysis code deposited on publication.

---

## 10. Limitations

### 10.1 Photosensitive epilepsy — a competing explanation

Between roughly 3% and 14% of PWE have visually-provoked seizures [Fisher_2022]. Clinical management routinely includes deliberate light avoidance: sunglasses outdoors, reduced screen brightness, avoiding sunlight through trees or on water, avoiding strobing environments.

There is therefore a well-documented mechanism by which epilepsy causes reduced bright-light exposure that is entirely independent of circadian disruption. If H1 is supported, "light avoidance secondary to photosensitivity or its clinical management" is a competing explanation for the headline finding.

Photosensitivity cannot be measured in NHANES. Three responses are made: (i) the alternative is stated explicitly in the Discussion; (ii) an age-stratified sensitivity analysis is reported, since photosensitivity is more common in younger people and in generalised epilepsies [Fisher_2022]; and (iii) directional reasoning is applied — light avoidance would plausibly affect daytime exposure strongly and nighttime exposure weakly, so findings driven by the nighttime side or by light regularity are less vulnerable to this explanation than daytime findings alone.

### 10.2 Shared-device dependence

Light, activity and sleep all derive from one wrist-worn sensor. A sedentary indoor person produces both low lux and low MIMS partly for mechanical reasons; sleep periods produce near-zero movement *and* near-zero light by construction, since the wrist may be under bedding. Light–activity associations are therefore partly non-independent by construction. This is acknowledged rather than solved; the coupling sensitivity analysis (§8.3.6) demonstrates awareness of the interdependence.

### 10.3 Sensor limitations

The GT3X+ light sensor is uncalibrated photopic lux and has not been validated against reference photometry in NHANES. CDC states the lux estimates *"are only estimates and are not meant for exact interpretation"* [NCHS_2022]. Wrist placement does not capture light at eye level; the device casing, clothing and bedding may occlude the sensor; sensitivity at low lux is reduced [Johnson_2023]. If this measurement error is non-differential by epilepsy status it biases toward the null; if PWE differ systematically in sleeve or bedding habits, bias could go either way. Top-coding at 2,500 lux additionally compresses the daytime range (§6.2) and attenuates day–night contrast toward the null — a conservative direction.

### 10.4 No melanopic conversion

Spectral information is unavailable, so photopic lux cannot be converted to melanopic equivalent daylight illuminance (CIE S 026). Results are reported in photopic lux and must not be interpreted as melanopic. Tools designed for spectrally-calibrated loggers are used only for relative photopic metrics.

### 10.5 Generalisability and the matched estimand

The primary analysis estimates an effect within a matched sample, not a nationally representative population parameter. Results describe the association between epilepsy and light exposure among people comparable on the matched covariates, and should not be read as prevalence estimates for US adults with epilepsy. The supplementary survey-weighted analysis (§8.5) addresses the national estimand at the cost of precision, and agreement between the two is reported.

Sampling is nonetheless a relative strength: NHANES is a probability sample of the non-institutionalised US population, so it does not carry the healthy-volunteer selection bias that affects volunteer cohorts such as UK Biobank — a limitation explicitly acknowledged by [Bailey_2023].

### 10.6 Design and ascertainment

Cross-sectional design precludes causal inference and temporal ordering. Epilepsy ascertainment relies on self-reported medication and indication, with no seizure frequency, seizure type, epilepsy syndrome, age at onset, or seizure timing data. Residual misclassification from off-label ASM use is possible and unquantified in NHANES. Small case count limits precision. Shift work is not recorded in these cycles [Johnson_2023].

---

## 11. Citation confidence

I have not verified every reference below against the primary source. Please check all before submission.

**Verified directly during preparation (full text or official documentation read):**
[Bailey_2023] Bailey GA, Wadon ME, Komarzynski S, Matthews C, Davies EH, Peall KJ. Accelerometer-derived sleep measures in idiopathic dystonia: A UK Biobank cohort study. *Brain and Behavior* 2023;13(9):e2933. doi:10.1002/brb3.2933. *Full text read. Matched 241 cases to 964 controls on age and gender at 4:1; case–control comparison by chi-square, t-test and Mann–Whitney U with Bonferroni correction; GGIR v2.3-0 with the van Hees diary-free algorithm; exclusion at <3 days of >16 h. Cited here as a design precedent and for the GGIR pipeline, not as a template for confounding control.*
[Tang_2024] Tang T, Zhou Y, Zhai X. Circadian rhythm and epilepsy: a nationally representative cross-sectional study based on actigraphy data. *Front Neurol* 2024;15:1496507. doi:10.3389/fneur.2024.1496507. PMID 39691456. *Full text read. NHANES 2013–2014 only; n=7,410; 53 epilepsy cases (0.75%); ages 3+ (categories 3–19, 20–39, 40–59, ≥60). Epilepsy defined by self-reported medication for ICD-10 G40 with manual ASM confirmation. nparACT on PAXMTSM. Weighted with WTINT2YR via the R survey package; SDMVSTRA/SDMVPSU not mentioned. No valid-day or wear-time rule. Fully adjusted Model 3 contains only age, sex, race, income and BMI. Reported findings: lower IS and lower M10 in PWE; highest-vs-lowest quartile ORs for IS 0.36, RA 0.25, M10 0.24, IV 2.52, M10 start time 3.13.*

*Three problems to be aware of when citing.* (i) L5 and M10 start times appear to be arithmetically averaged clock times (§8.2). (ii) The reported age mean of 39.62 with SD 40.48 is impossible for a 3–80 range. (iii) The Discussion states PWE showed "a more synchronized rest-activity rhythm" with "increased daytime activity levels", which contradicts the paper's own Table 2 (lower IS, lower M10). Cite the study for its ascertainment approach and as evidence of prior rest–activity findings, with appropriate caution on the specifics.

[NCHS_2022] National Center for Health Statistics. *NHANES 2013–2014 Data Documentation, Codebook, and Frequencies: Physical Activity Monitor — Ambient Light Raw Data (PAXLUX_H).* Centers for Disease Control and Prevention; first published October 2022. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXLUX_H.htm

*This is the CDC/NCHS data documentation page, not a journal article. Everything attributed to it in this document — the 24 h wear protocol, the 80 Hz / 1 Hz sampling rates, the absence of any sleep diary or activity log, the partial first and last days, the 2,500 lux top-code, the "estimates … not meant for exact interpretation" caveat, and the direction to use the examined-sample weights — was read directly from this page.*

*Caveat: I verified the **2013–2014 (PAXLUX_H)** documentation only. The companion 2011–2012 file (PAXLUX_G, https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXLUX_G.htm) is presumed to describe the same protocol and the same top-coding, but I have not confirmed this. Since this analysis pools both cycles, please check PAXLUX_G before relying on any of the above as applying to 2011–2012 — particularly the 2,500 lux ceiling, which §6.2 depends on. If the two cycles differ, cite them separately as [NCHS_2022a] and [NCHS_2022b].*
[Johnson_2023] Johnson DA, Wallace DA, Ward L. Racial/Ethnic and Sex Differences in the Association between Light at Night and Actigraphy-Measured Sleep Duration in Adults: NHANES 2011–2014. *Sleep Health* 2024;10(1 Suppl):S184–S190. doi:10.1016/j.sleh.2023.09.011. PMID 37951773.
[Xiao_2023] Xiao Q, Durbin J, Bauer C, Yeung CHC, Figueiro MG. Alignment Between 24-h Light-Dark and Activity-Rest Rhythms Is Associated With Diabetes and Glucose Metabolism. *Diabetes Care* 2023;46(12):2171–2179. doi:10.2337/dc23-1034.

**High confidence, not personally verified in full text:**
[Su_2022] Su S, Li X, Xu Y, McCall WV, Wang X. *Sci Rep* 2022;12:7680.
[Fisher_2022] Fisher et al. Visually sensitive seizures: An updated review by the Epilepsy Foundation. *Epilepsia* 2022. doi:10.1111/epi.17175. PMID 35132632.
[John_2019] John D, Tang Q, Albinali F, Intille S. *J Meas Phys Behav* 2019;2:268–281.
[Leroux_2019] Leroux A, Di J, Smirnova E, et al. Organizing and Analyzing the Activity Data in NHANES. *Stat Biosci* 2019;11:262–287.
[Wallace_2023] Wallace DA. In the Light: Towards Developing Metrics of Light Regularity. *Sleep* 2023. doi:10.1093/sleep/zsad114. PMID 37075470.
[Miller_2010] Miller D, Bierman A, Figueiro M, Schernhammer E, Rea M. *Light Res Technol* 2010;42:271–284.
[Rea_2008] Rea MS, Bierman A, Figueiro MG, Bullough JD. *J Circadian Rhythms* 2008;6:7.
[Xiao_2022b] Xiao Q, Lu J, Zeitzer JM, et al. Rest-activity profiles among U.S. adults: a functional principal component analysis. *Int J Behav Nutr Phys Act* 2022;19:32.
[Degenfellner_2025] Degenfellner J, Schernhammer E, Strohmaier S. Sex- and ethnic differences in the association between sleep regularity and obesity, NHANES 2011–2014. *J Biol Rhythms*. doi:10.1177/07487304251391267.
[Maxwell_Cole_2007] Maxwell SE, Cole DA. *Psychol Methods* 2007;12:23–44.

**Moderate confidence — verify details before use:**
[vanHees_2018] van Hees VT, Sabia S, Jones SE, et al. Estimating sleep parameters using an accelerometer without sleep diary. *Scientific Reports* 2018;8:1–11. doi:10.1038/s41598-018-31266-z. *Taken from the reference list of [Bailey_2023]; not independently verified.*
[LancetHL_2023] **Author names not established.** "Association between accelerometer-measured amplitude of rest–activity rhythm and future health risk: a prospective cohort study of the UK Biobank." *Lancet Healthy Longevity* 2023; PMID 37148892; n=92,614. I have the abstract and the model-structure description (models 0–3, with model 2 adding sleep and physical activity and model 3 adding other rest–activity parameters) but **not the author list**. Complete this before citing.
[Terman_2020] Terman et al. *Epilepsy Behav* 2020;111:107261. Reported as the source for ASM+G40 ascertainment; I have not confirmed the exact volume/page or the reported case counts.
[Thapa-Chhetry_2022] Thapa-Chhetry et al. Detecting Sleep and Nonwear in 24-h Wrist Accelerometer Data. *Med Sci Sports Exerc* 2022;54(11):1936–1946.
[Blume_2016] Blume C, Santhi N, Schabus M. nparACT package for R. *MethodsX* 2016.
[Lumley_2010] Lumley T. *Complex Surveys: A Guide to Analysis Using R.* Wiley, 2010.
[DuGoff_2014] DuGoff EH, Schuler M, Stuart EA. Generalizing observational study results: applying propensity score methods to complex surveys. *Health Serv Res* 2014.
[VanderWeele_Ding_2017] VanderWeele TJ, Ding P. Sensitivity Analysis in Observational Research: Introducing the E-Value. *Ann Intern Med* 2017.
[Brown_2022] Brown TM, et al. Recommendations for daytime, evening and nighttime indoor light exposure. *PLOS Biology* 2022. Verify the exact threshold values cited.
[Zauner_2025] Zauner J, Hartmeyer S, Spitschan M. LightLogR. *J Open Source Software* 2025. Verify version and whether a separate Zauner et al. paper on zero-inflated light transformation is the correct citation for §6.3.

**From your own literature notes — I have not independently verified any of these:**
[Liguori_2022], [Skeldon_2022], [Berra_2009], [Fernandez_2019]

---

## 12. Decisions taken and remaining actions

### Resolved

1. **Clock window boundaries — fixed at day 07:00–19:00, night 23:00–06:00.** Hours 19:00–23:00 and 06:00–07:00 excluded from both windows. These are now specified, not placeholders, and should not be altered after analysis begins (§6.1).
2. **Age lower bound — 20 years**, following [Xiao_2023]. Note when contrasting results that [Tang_2024] included ages 3 and up, so its estimates are not directly comparable to this adults-only analysis.
3. **Depression — treated as a mediator, not a confounder.** It is therefore excluded from the matching set and from the primary model, and entered only in clearly-labelled secondary models (§6.7). The assumed structure is epilepsy → depression → reduced outdoor activity → reduced light exposure. This is an assumption, not a finding: cross-sectional data cannot establish the ordering, and a reviewer may reasonably argue depression precedes and confounds instead. State the assumption explicitly and justify it in the manuscript; the DAG (below) is the natural place to make it visible.
4. **Propensity-model stability** — the matching set is six variables following the removal of smoking (§6.7), which eases but does not eliminate the events-per-variable constraint. The fallback specification (§8.1, Stage 2) is documented; the decision to invoke it will be made at the design stage, before outcomes are examined.
5. **Pre-registration — to be completed before analysis.** OSF recommended. This is what licenses the balance-driven iteration in §8.1 Stage 2 and the fixed reporting rule in §8.5.

### Remaining actions

6. **[LancetHL_2023] author list** must be completed (§11). PMID 37148892.
7. **PAXLUX_G (2011–2012) documentation** to be checked against PAXLUX_H, particularly the 2,500 lux ceiling on which §6.2 depends (§11).
8. **Three unexplored methodological areas**, deferred: compositional / 24-hour time-use analysis; measurement-error and regression-calibration methods for wearable exposures; NHANES epilepsy ascertainment literature outside accelerometry.
9. **Draw a DAG** of the assumed causal structure for the supplementary material. Given that several classifications in §6.7 are assumptions rather than findings — depression as mediator being the clearest — a directed acyclic graph converts an arguable set of choices into something a reader can inspect and contest. Recommended.
