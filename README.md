# Credit Risk Modeling and Gender Fairness in Loan Approvals

This project uses the German Credit dataset to build a model that predicts whether a bank loan applicant is a good or bad credit risk. The dataset has 1,000 applicants with information like loan duration, credit history, savings, employment status, and age.

The goal of the first part is straightforward: train a logistic regression model that can classify applicants as good or bad credit risks. Logistic regression is a common choice for this kind of yes/no prediction problem in banking and finance.

The second part is what makes this project more interesting. Once the model is built, I check whether it treats male and female applicants the same way. Specifically, I look at two things: how often each group gets approved (Approval Rate), and how often genuinely creditworthy people in each group actually get approved (True Positive Rate). If those numbers are very different between males and females, the model is being unfair even if it was never explicitly told to consider gender.

---

## The Fairness Problem

When you train a model on historical data, it picks up patterns from that data. If the data reflects past biases or imbalances, the model can reproduce them. In this dataset, males make up 69% of applicants and have a higher average credit risk score than females. When we apply a single decision threshold of 0.5 to everyone, males get approved 80% of the time while females only get approved 68% of the time. The True Positive Rate gap is about 5 percentage points.

This is not because the model is broken. It is because the predicted probability distributions for males and females look different. The female distribution is shifted to the left, meaning the model assigns lower probabilities to female applicants on average. A single cutoff at 0.5 catches fewer females as a result, even among those who are genuinely good credit risks.

---

## The Fix

Instead of using one threshold for everyone, we can use a separate threshold for males and females. The idea is to find the pair of thresholds where the True Positive Rate for males and females are as close as possible (within 0.1%) while keeping overall errors as low as possible.

The optimal pair turns out to be 0.42 for males and 0.34 for females. Lowering the threshold for females means the model approves more female applicants, which brings their True Positive Rate in line with males. The cost in overall accuracy is less than 0.2 percentage points.

This approach is known as equalized opportunity: making sure that people who genuinely qualify for a loan have an equal chance of being approved regardless of their gender.

---

## Charts

**Predicted probability distributions by gender**

This chart shows that the model outputs lower probabilities for female applicants on average. The dashed line at 0.5 shows where the cutoff is. You can see that the female curve has more of its mass to the left of that line.

![density](analysis_files/figure-gfm/density-plot-1.png)

**Approval Rate and True Positive Rate by gender at a single 0.5 threshold**

This shows the disparity clearly. Both the approval rate and the True Positive Rate are noticeably lower for females.

![fairness bar](analysis_files/figure-gfm/fairness-bar-1.png)

**TPR curves by gender across all thresholds**

Each line shows how the True Positive Rate changes as the threshold is adjusted from 0 to 1. The dashed vertical lines mark the optimal thresholds (0.42 for males, 0.34 for females) where the two curves intersect at the same TPR value.

![tpr curve](analysis_files/figure-gfm/tpr-curve-1.png)

**Top model coefficients**

These are the variables the model found most useful. Blue bars are statistically significant predictors. The direction of the bar tells you whether a variable increases or decreases the probability of being a good credit risk.

![coefficients](analysis_files/figure-gfm/coef-plot-1.png)

---

## Results Summary

| | Single threshold (0.5) | Optimized thresholds |
|---|---|---|
| Male threshold | 0.50 | 0.42 |
| Female threshold | 0.50 | 0.34 |
| TPR gap between genders | about 5 percentage points | less than 0.1 percentage points |
| Overall error rate (training) | 20.7% | 20.8% |
| Overall error rate (test) | not evaluated | 24.8% |

---

## Files

- [`analysis.Rmd`](analysis.Rmd) - source file with all the code, run this to reproduce everything
- [`analysis.md`](analysis.md) - rendered output with all charts and tables visible on GitHub
- [`German Credit Data.csv`](German%20Credit%20Data.csv) - the dataset

## How to Run

```r
# install.packages(c("rmarkdown", "ggplot2", "tidyr", "scales", "knitr"))
rmarkdown::render("analysis.Rmd", output_format = "github_document")
```
