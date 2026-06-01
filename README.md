# Credit Risk Modeling & Gender Fairness in Loan Approvals

Logistic regression on the German Credit dataset — 1,000 loan applicants, predicting whether someone is a good or bad credit risk. The interesting part isn't the model itself; it's what happens when you look at how the model's decisions break down by gender, and what you can do about it.

The full analysis with code, charts, and tables is in [`analysis.md`](analysis.md).

---

## What this covers

Built a logistic regression model using R, then checked whether a single 0.5 decision threshold treats male and female applicants the same way. It doesn't — males get approved at 80% vs females at 68%, even after controlling for everything else in the model.

The second part optimizes a separate threshold for each gender so that the True Positive Rate (the rate at which genuinely creditworthy people get approved) is equalized across groups. The cost in overall accuracy is less than 0.2 percentage points.

---

## Charts

**Predicted probability distributions by gender**
![density](analysis_files/figure-gfm/density-plot-1.png)

The female distribution is shifted left — the model assigns lower probabilities to females on average, which is why a single cutoff at 0.5 catches fewer of them.

**Gender disparity at a single 0.5 threshold**
![fairness bar](analysis_files/figure-gfm/fairness-bar-1.png)

**TPR by threshold for each gender — finding where they equalize**
![tpr curve](analysis_files/figure-gfm/tpr-curve-1.png)

**Top model coefficients**
![coefficients](analysis_files/figure-gfm/coef-plot-1.png)

---

## Key result

| | Single threshold (0.5) | Optimized thresholds |
|---|---|---|
| Male threshold | 0.50 | 0.42 |
| Female threshold | 0.50 | 0.34 |
| TPR gap | ~5 pp | < 0.1 pp |
| Overall error | 20.7% | 20.8% |

---

## Files

- [`analysis.Rmd`](analysis.Rmd) — source file, run this to reproduce everything
- [`analysis.md`](analysis.md) — rendered output with all charts and tables
- [`German Credit Data.csv`](German%20Credit%20Data.csv) — dataset

## Running it

```r
# install.packages(c("rmarkdown", "ggplot2", "tidyr", "scales", "knitr"))
rmarkdown::render("analysis.Rmd", output_format = "github_document")
```
