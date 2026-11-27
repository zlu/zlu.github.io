---
title: Intervention model
synonyms:
  - Interrupted time series
  - ITS
  - Intervention analysis
keywords: intervention model, interrupted time series, policy evaluation, dummy variable, statistics glossary
related:
  - Time series
  - Forecasting
  - Durbin–Watson
  - Dummy variable
  - Autocorrelation
---
An **intervention model** (interrupted time series) evaluates a policy/event with a before/after indicator. Typical steps: mark the intervention date; set $D_t=0$ before and $1$ after; regress the outcome on traffic/GDP/etc., $D_t$, [trend](/cs-glossary/trend/), and [seasonality](/cs-glossary/seasonality/); check K–S normality and [Durbin–Watson](/cs-glossary/durbin-watson/).

Example: London congestion charge effect on accidents — $\mathrm{Accidents}=f(\mathrm{traffic},D_t,\ldots)$. Coefficient on $D_t$ is the estimated level shift if the specification is credible.
