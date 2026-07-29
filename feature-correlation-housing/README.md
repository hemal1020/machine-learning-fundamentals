# Feature Correlation Analysis — California Housing Prices

Correlation analysis identifying which features relate most strongly to `median_house_value`, including a
concrete demonstration of a common categorical-encoding mistake that distorts correlation results.

**Dataset:** 20,640 California housing districts, 8 numeric features + 1 categorical (`ocean_proximity`,
5 nominal categories), target `median_house_value`. Swapped in from the original 9-row toy dataset —
correlation coefficients aren't statistically meaningful on that few rows.

**Key finding — a real encoding bug, not just style:** the original notebook used `OrdinalEncoder` on
nominal categorical columns (categories with no real order, like location or ocean proximity). That forces
categories onto an arbitrary number line, which manufactures a fake linear relationship correlation then
measures. Demonstrated directly:

| Encoding approach | Correlation with price |
|---|---|
| OrdinalEncoder (original notebook's method) | +0.082 (weak, misleading) |
| One-hot: `ocean_INLAND` | **-0.485** |
| One-hot: `ocean_<1H OCEAN` | **+0.257** |

Squeezing 5 unordered categories onto one axis nearly hid a strong, real, directionally-opposite
relationship.

**Approach:** median imputation for 207 missing `total_bedrooms` values, one-hot encoding for the nominal
categorical feature, 3 engineered ratio features (`rooms_per_household`, `bedrooms_per_room`,
`population_per_household`), full correlation matrix with heatmap, and a ranked/sign-colored bar chart.

**Top correlations with price:** `median_income` (+0.688), `ocean_INLAND` (-0.485), `bedrooms_per_room`
(-0.233) — the engineered ratio meaningfully outperformed the raw `total_bedrooms` count (+0.049).

**What I learned:** correlation assumes a linear relationship, so encoding matters — ordinal encoding is
only valid for genuinely ordered categories (like "low/medium/high"), not nominal ones. Also learned that
engineered ratio features can carry more signal than the raw totals they're derived from, and that
correlation needs a reasonably large sample to mean anything at all.
