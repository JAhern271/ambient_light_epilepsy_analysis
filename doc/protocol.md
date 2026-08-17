# Study protocol

Ambient light exposure and circadian disruption in epilepsy.

> This is the study as designed. Where the implemented analysis currently departs from
> it, see [Deviations from protocol](#deviations-from-protocol) at the end — that section
> is the authoritative description of what the code actually does.

## Background and rationale

People with epilepsy (PWE) experience high rates of sleep disturbance and circadian
rhythm disruption, which are known to exacerbate seizure risk, impair quality of life,
and worsen mental health. A substantial literature documents altered rest–activity
rhythms in epilepsy, including reduced circadian stability and amplitude.

Recent work using the 2013–2014 NHANES accelerometry dataset demonstrated that
individuals with epilepsy show significant alterations in nonparametric circadian rhythm
metrics (e.g. reduced interdaily stability and relative amplitude, altered M10 timing)
compared to the general population. However, these analyses focused exclusively on
rest–activity patterns and did not examine ambient light exposure, despite the fact that
the same NHANES cohort includes continuous wearable light (lux) measurements.

Liguori (2022) conducted 14-day actigraphy measurements in a home environment and found
that PWE experienced disrupted rest–activity rhythms compared to controls. Specifically,
PWE had higher levels of activity fragmentation in addition to lower activity rhythm
regularity. Activity rhythm disturbances were also reflected in sleep measures, as PWE
had lower sleep efficiency and longer sleep latency. Activity (M10) was also lower in
PWE. These recordings probably did not contain seizures, so the cause of disrupted
activity rhythms and sleep quality is likely not directly attributable to seizures. Some
studies have hinted that PWE have lower ambient light exposure (Berra 2009, Fernandez
2019), which could explain disrupted activity patterns.

In other neurological and psychiatric conditions (e.g. dementia, mood disorders),
insufficient daytime light and excessive nighttime light have been identified as primary
drivers of circadian and sleep disruption, sometimes outweighing intrinsic circadian
period changes in mathematical and empirical models. People with epilepsy are known to
have lower physical activity levels and higher rates of depression and anxiety, raising
the possibility that reduced outdoor exposure and altered light environments may
contribute to observed circadian disruption.

This project tests the hypothesis that **differences in ambient light exposure partially
explain circadian rhythm and sleep disturbances observed in people with epilepsy**, using
existing large-scale wearable datasets.

## Aims and objectives

### Primary aim

To determine whether people with epilepsy have significantly different ambient light
exposure patterns compared to matched healthy controls.

### Secondary aims

- To examine whether ambient light exposure is associated with altered rest–activity
  circadian rhythm metrics in epilepsy.
- To assess whether light exposure metrics explain (or mediate) differences in circadian
  stability, amplitude, and sleep quality between epilepsy and control groups.

### Exploratory aims

- To replicate key findings in an independent cohort (UK Biobank) if feasible.
- To generate effect size estimates to inform a future prospective wearable or
  light-intervention study.

## Datasets

### NHANES (2013–2014)

Approximately 7,000–14,000 participants with:

- Wrist-worn accelerometry
- Continuous ambient light (lux) measurements
- Demographic, clinical, mental health, and sleep questionnaire data

Epilepsy status identified via self-reported medical conditions. Previously used to
demonstrate altered circadian rhythms in epilepsy, providing a strong foundation for
extension.

### UK Biobank (potential extension)

Approximately 80,000 participants with wrist-worn accelerometer and light sensor data.
Epilepsy identified via ICD-10 codes (G40). Enables replication in a larger, independent
population with richer longitudinal health data.

## Methods overview

### 1. Cohort definition

- Identify individuals with epilepsy.
- Match controls on age, sex, BMI, season of wear, socioeconomic status, and physical
  activity where possible.

### 2. Ambient light exposure metrics

- Mean daytime light exposure (e.g. 06:00–18:00)
- Time above physiologically relevant thresholds (e.g. >100 lux, >1,000 lux)
- Nighttime light exposure (LAN)
- Day–night light contrast
- Light regularity across days

### 3. Circadian and sleep metrics

- Nonparametric circadian rhythm metrics (IS, IV, RA, M10, L5, timing)
- Sleep duration and fragmentation (derived from accelerometry)

### 4. Statistical analysis

- Group comparisons (epilepsy vs controls)
- Multivariable regression adjusting for confounders
- Mediation or explanatory modeling to assess whether light exposure accounts for
  circadian differences

## Expected outcomes

Evidence that PWE experience:

- Reduced daytime light exposure
- Increased nighttime light exposure
- Lower day–night light contrast

Demonstration that ambient light exposure is associated with altered circadian rhythm
metrics in epilepsy, and identification of light exposure as a modifiable environmental
factor contributing to sleep and circadian disruption, and potentially seizure frequency.

## Impact and next steps

This project would provide the first large-scale evidence linking ambient light exposure
to circadian disruption in epilepsy, and strengthen the rationale for targeted
light-sensing wearables, personalised light exposure feedback, or future light-based
interventions. It would de-risk and inform the design of a prospective epilepsy-specific
wearable study, allowing meaningful progress before committing to new data collection or
device procurement.

---

## Deviations from protocol

The implemented analysis differs from the protocol above in the following ways. These are
the descriptions to use when writing up methods.

| Protocol | As implemented | Rationale |
|---|---|---|
| Epilepsy via self-reported medical conditions | Current use of one of 12 epilepsy-specific antiseizure medications in the prescription inventory | More specific than the NHANES self-report item |
| NHANES 2013–2014 | Cycles G (2011–12) **and** H (2013–14) | Roughly doubles the sample: 192 PWE, 669 controls |
| Match on age, sex, BMI, season, SES, physical activity | Frequency matched on age band, sex, race/ethnicity, season, PIR band | BMI and physical activity not currently used as matching variables |
| Daytime 06:00–18:00 | 07:00–19:00 (night 20:00–05:00) | Not yet reconciled |
| Thresholds >100 and >1,000 lux | >1,000 lux only | >100 lux not yet implemented |
| Day–night light contrast | Not implemented | Outstanding |
| Light from the dedicated 1 Hz recordings | **PAXMIN minute-level light** intended for publication; PAXLUX 1 Hz and 5-minute results are exploratory | PAXMIN carries light and activity in one table at a resolution ample for circadian work, so both can be compared on identical sampling, and it masks non-wear |
| Sleep duration and fragmentation | Not implemented | Depends on the PAXMIN work in progress |
| Mediation modelling | Confounder adjustment only | Outstanding |
| UK Biobank replication | Not started | Exploratory aim |

Additionally, participants are restricted to **age ≥ 20** (excluding 22 of 123 identified
PWE who were under 20) and to recordings flagged valid over the full 9 days
(`PAXSTS == 1`, `PAXLDAY == 9`).

## References

- Berra et al. (2009)
- Fernandez et al. (2019)
- Liguori et al. (2022)

> Full citations to be completed.
