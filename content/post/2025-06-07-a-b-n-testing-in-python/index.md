---
title: A/B/N Testing in Python (In Progress)
author: Package Build
date: '2025-06-07'
slug: abn-testing-python
categories:
  - "AB-Testing"
  - "Python"
  - "EDA"
tags:
  - "AB-Testing"
  - "Python"
  - "EDA"
---



The following case study will illustrate how to analyze the results of a type of experiment known as a multitest (A/N). An A/N Test is just an A/B Test in which multiple variants are tested at the same time. 

We will compare 2 variants against a control to the increase purchase rate on a website. Since testing multiple variants at once will increase error the rate (known as Family Wise Error Rate--FWER), we will use a correction when determining statistical significance. 

# Analyzing results

We are asked to analyze the results of an experiment, performed on the splash-page, for a fictional theme park called Redwood Ridge. The park wants to launch an AI assisted booking agent, referred to as Rocky Raccoon, to help customers book flights, rental cars, meals, etc. They wish to test: Variant_A with a simplified widget; Variant_B with a more complex interactive wizard; against the control page with no agent. 

<http>
<center>
<img src="variants2.png" alt = "The three pages"  width = "85%">
</center>
</http>

The test has already been performed. Therefore, we will skip the test planning and sample size calculations accepting this has all been done for us.

# Formulating the Hypothesis

First we formulate the alternate hypothesis. This is the one we are trying to accept by default of rejecting the Null Hypothesis.

__The Alternate Hypothesis: Adding an interactive travel planning wizard to the Homepage will boost ticket purchase conversion rates__

The null hypothesis is always based on the idea that that every difference detected, in the measured statistic, is simply due to random chance. All hypothesis boils down to trying to reject the null hypothesis if there is enough evidence that it is unlikely the results due to chance.

__Null Hypothesis: An increase in ticket purchase rate can be explained as random chance__

Our test statistic is defined as:

__Test Statistic: Ticket Purchase Conversion Rate = (purchase count)/(unique visit count)__



Let's use Python to analyze the results and determine if it is safe to reject the Null Hypothesis. 

## Load Python libraries


```{.python .my-python-code}
import pandas as pd
import numpy as np
```


```{.python .my-python-code}
rocky = pd.read_csv("../../../static/images/a-b-n-testing-in-python/data.csv")
```



```{.python .my-python-code}
print(rocky)
```

``` my-python-output
##               date  visit_id  ... trip_planner_engaged  ticket_purchased
## 0       2024-04-01    514882  ...                    0                 0
## 1       2024-04-01    514883  ...                    1                 0
## 2       2024-04-01    514884  ...                    0                 0
## 3       2024-04-01    514885  ...                    0                 0
## 4       2024-04-01    514886  ...                    0                 0
## ...            ...       ...  ...                  ...               ...
## 264943  2024-04-30    779702  ...                    0                 0
## 264944  2024-04-30    779703  ...                    0                 0
## 264945  2024-04-30    779703  ...                    0                 0
## 264946  2024-04-30    779704  ...                    0                 0
## 264947  2024-04-30    779704  ...                    0                 0
## 
## [264948 rows x 5 columns]
```
## EDA

### Inspect the Variants


```{.python .my-python-code}
rocky.groupby('treatment')['ticket_purchased'].agg(['mean', 'count', 'std'])
```

``` my-python-output
##                  mean  count       std
## treatment                             
## control      0.020993  88266  0.143363
## variation_A  0.022494  88112  0.148285
## variation_B  0.023800  88570  0.152428
```

We see there is a difference in the means; with the *variation_B* showing the highest lift. But let's make sure we don't have duplicates.

## Check for Duplicates


```{.python .my-python-code}
print(len(rocky))
print(len(rocky.drop_duplicates(keep=False)))
```

```
## 264948
## 264948
```

No purely duplicate records.


```{.python .my-python-code}
print(rocky[['visit_id', 'treatment']].nunique())
```

```
## visit_id     264823
## treatment         3
## dtype: int64
```

But there are some with different treatments for the same visit.


```{.python .my-python-code}
print(len(rocky.drop_duplicates(subset=['visit_id','treatment'], keep=False)))
```

```
## 264948
```

There are some duplicate visit id's. Considering only visit_id and treatment there are no dupes. Therefore, some visits have multiple records for visit_id with diffent versions of the homepage. This may be a bug in the design if the intent was for a visit to have only one version of the homepage. 

## Drop Duplicates

Now we can drop these duplicates and check lift for each variation again. We should exclude these visits where the same visit resulted in seeing more than one treatment. We'll use the `keep=False` argument to the `drop_duplicates()` method in Pandas.

> **_NOTE:_** It is possible to run an experiment where someone sees multiple variants known as "paired samples"--using a different method known as a "paired" test to analyze. But these seem to be due to a flaw in the experiment, rather than intentional, and led to a small sample size. We'll focus on analyzing the rest as independent samples or "un-paired" samples. 



```{.python .my-python-code}
rocky = rocky.drop_duplicates(subset=['visit_id'], keep=False)
print(len(rocky))
rocky.groupby('treatment')[['trip_planner_engaged', 'ticket_purchased']].agg(['mean', 'count'])
```

```
## 264698
##             trip_planner_engaged        ticket_purchased       
##                             mean  count             mean  count
## treatment                                                      
## control                 0.132095  88141         0.021000  88141
## variation_A             0.276302  87987         0.022503  87987
## variation_B             0.274201  88570         0.023800  88570
```

So we discarded all records for any *visit_id* that has multiple records.

## Group and Inspect


```{.python .my-python-code}
rocky.groupby(['treatment', 'trip_planner_engaged'])[ 'ticket_purchased'].agg(['mean', 'count'])
```

```
##                                       mean  count
## treatment   trip_planner_engaged                 
## control     0                     0.020876  76498
##             1                     0.021816  11643
## variation_A 0                     0.023117  63676
##             1                     0.020896  24311
## variation_B 0                     0.024205  64284
##             1                     0.022729  24286
```

It doesn't make sense to have trip planner engagment for the control group. Something is amiss. We should alert Engineering that our logging seems to be broken. Also, there is a large imbalance in the groups, but there is a bigger problem with the *trip_planner_engaged* field. We'll ignore this field focusing on just the impact of the variants on ticket purchases.

# Checking for Significance in the Difference in Means

Since *varation_B* seems to have the highest lift let's see if the results are significant (without correction).


```{.python .my-python-code}
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
# Calculate the number of visits
n_C = rocky[rocky['treatment'] == 'control']['ticket_purchased'].count()
n_B = rocky[rocky['treatment'] == 'variation_B']['ticket_purchased'].count()
print('Group C users:',n_C)
print('Group B users:',n_B)

# Compute unique purshases in each group and assign to lists
signup_C = rocky[rocky['treatment'] == 'control'].groupby('visit_id')['ticket_purchased'].max().sum()
signup_B = rocky[rocky['treatment'] == 'variation_B'].groupby('visit_id')['ticket_purchased'].max().sum()

purchase_abtest = [signup_C, signup_B]
n_cbtest = [n_C, n_B]

# Calculate the z_stat, p-value, and 95% confidence intervals
z_stat, pvalue = proportions_ztest(purchase_abtest, nobs=n_cbtest)
(C_lo95, B_lo95), (C_up95, B_up95) = proportion_confint(purchase_abtest, nobs=n_cbtest, alpha=.05)

pvalue_C_B = pvalue
print(f'p-value: {pvalue:.6f}')
print(f'Group C 95% CI : [{C_lo95:.4f}, {C_up95:.4f}]')
print(f'Group B 95% CI : [{B_lo95:.4f}, {B_up95:.4f}]')
```

```
## Group C users: 88141
## Group B users: 88570
## p-value: 0.000070
## Group C 95% CI : [0.0201, 0.0219]
## Group B 95% CI : [0.0228, 0.0248]
```

Next let's look at *control* vs *variation_A*


```{.python .my-python-code}
# Calculate the number of visits
n_C = rocky[rocky['treatment'] == 'control']['ticket_purchased'].count()
n_B = rocky[rocky['treatment'] == 'variation_A']['ticket_purchased'].count()
print('Group C users:',n_C)
print('Group A users:',n_B)

# Compute unique purshases in each group and assign to lists
signup_C = rocky[rocky['treatment'] == 'control'].groupby('visit_id')['ticket_purchased'].max().sum()
signup_B = rocky[rocky['treatment'] == 'variation_A'].groupby('visit_id')['ticket_purchased'].max().sum()

purchase_abtest = [signup_C, signup_B]
n_cbtest = [n_C, n_B]

# Calculate the z_stat, p-value, and 95% confidence intervals
z_stat, pvalue = proportions_ztest(purchase_abtest, nobs=n_cbtest)
(C_lo95, B_lo95), (C_up95, B_up95) = proportion_confint(purchase_abtest, nobs=n_cbtest, alpha=.05)

pvalue_C_A = pvalue

print(f'p-value: {pvalue:.6f}')
print(f'Group C 95% CI : [{C_lo95:.4f}, {C_up95:.4f}]')
print(f'Group A 95% CI : [{B_lo95:.4f}, {B_up95:.4f}]')
```

```
## Group C users: 88141
## Group A users: 87987
## p-value: 0.030623
## Group C 95% CI : [0.0201, 0.0219]
## Group A 95% CI : [0.0215, 0.0235]
```


```{.python .my-python-code}
# Calculate the number of visits
n_C = rocky[rocky['treatment'] == 'variation_A']['ticket_purchased'].count()
n_B = rocky[rocky['treatment'] == 'variation_B']['ticket_purchased'].count()
print('Group A users:',n_C)
print('Group B users:',n_B)

# Compute unique purshases in each group and assign to lists
signup_C = rocky[rocky['treatment'] == 'variation_A'].groupby('visit_id')['ticket_purchased'].max().sum()
signup_B = rocky[rocky['treatment'] == 'variation_B'].groupby('visit_id')['ticket_purchased'].max().sum()

purchase_abtest = [signup_C, signup_B]
n_cbtest = [n_C, n_B]

# Calculate the z_stat, p-value, and 95% confidence intervals
z_stat, pvalue = proportions_ztest(purchase_abtest, nobs=n_cbtest)
(C_lo95, B_lo95), (C_up95, B_up95) = proportion_confint(purchase_abtest, nobs=n_cbtest, alpha=.05)

pvalue_A_B = pvalue

print(f'p-value: {pvalue:.6f}')
print(f'Group A 95% CI : [{C_lo95:.4f}, {C_up95:.4f}]')
print(f'Group B 95% CI : [{B_lo95:.4f}, {B_up95:.4f}]')
```

```
## Group A users: 87987
## Group B users: 88570
## p-value: 0.069995
## Group A 95% CI : [0.0215, 0.0235]
## Group B 95% CI : [0.0228, 0.0248]
```

So the pvalues are:

* *Control* vs *variant_A*: 0.0306233
* *Control* vs *variant_B*: 6.9916103\times 10^{-5}
* *variant_A* vs *variant_B*: 0.0699954

Normally a pvalue less that 0.05 indicates strong evidence against the NULL hypothesis and that we should reject it. And since both variants show significance (uncorrected) we might be tempted to reject the NULL hypothesis for both and only accept it when pitting the two variants against each other--leading us to conclude the difference between the 2 variants is not significant (or is simply random chance). And secondly, that either would be preferable to the Control. But this would be a mistake. 

When performing an experiment with more than variation we need to apply a correction to account for Family Wise Eror Rate (FWER) since the probability of making at least one Type I error (a false positive) across all the hypothesis tests increases with each test. A simple correction is to use the Bonferroni correction. Essentially, this method divides the significance level (alpha) across the number of tests. This gives us a more conservative mark to hit. 

If we use the 3 pvalues we calculated and apply this method:


```{.python .my-python-code}
# Bonferroni correction for 95% Confidence interval
import statsmodels.stats.multitest as smt

pvals = [0.030713, 0.000067, 0.0679]

# Perform a Bonferroni correction and print the output
corrected = smt.multipletests(pvals, alpha = .05, method = 'bonferroni')

print('Significant Test:', corrected[0])
print('Corrected P-values:', corrected[1])
print('Bonferroni Corrected alpha: {:.4f}'.format(corrected[2]))
```

```
## Significant Test: [False  True False]
## Corrected P-values: [9.2139e-02 2.0100e-04 2.0370e-01]
## Bonferroni Corrected alpha: 0.0170
```

We see that the only test that is actually significant is the *Control* vs *variation_B*.  The [False, True, False] corresponds to the [Control_v_A, Control_v_B, vartiation_A_v_B] pvals that we supplied. 



# Visualizing the Bootstrapped Data

Now let's bootstrap random sample and calculate the mean of each group: *Control* and *variation_B* to visualize the distributions. This will give us a sense of the difference between the groups visually. 



```{.python .my-python-code}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Extract the two variants as requested
control_data = rocky[rocky['treatment'] == 'control'].groupby('visit_id')['ticket_purchased'].mean()
variation_b_data = rocky[rocky['treatment'] == 'variation_B'].groupby('visit_id')['ticket_purchased'].mean()

# Number of random samples to generate
num_samples = 1000
sample_size = 80000  # Size of each random sample

# Lists to store the sample means
control_sample_means = []
variation_b_sample_means = []

# For loop to build normal distributions through random sampling
for _ in range(num_samples):
    # Random sampling with replacement
    if len(control_data) > 0:
        control_sample = np.random.choice(control_data, size=min(sample_size, len(control_data)), replace=True)
        control_sample_means.append(control_sample.mean())
    
    if len(variation_b_data) > 0:
        variation_b_sample = np.random.choice(variation_b_data, size=min(sample_size, len(variation_b_data)), replace=True)
        variation_b_sample_means.append(variation_b_sample.mean())

# Create a figure with multiple plots
fig, ax = plt.subplots(figsize=(6, 4))


# Plot sampling distributions (normal distributions from random sampling)
sns.histplot(control_sample_means, kde=True, color='blue', ax=ax, label='Control')
sns.histplot(variation_b_sample_means, kde=True, color='orange', ax=ax, label='Variation B')
ax.set_title('Sampling Distributions (Normal Approximation)')
ax.set_xlabel('Sample Mean Ticket Purchase Rate')
ax.set_ylabel('Frequency')
ax.legend()


plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)

plt.show()
# Print sample sizes
print(f"Control group size: {len(control_data)}")
print(f"Variation B group size: {len(variation_b_data)}")
print(f"Number of random samples generated for each group: {num_samples}")
print(f"Size of each random sample: {sample_size}")
```

```
## <Axes: ylabel='Count'>
## <Axes: ylabel='Count'>
## Text(0.5, 1.0, 'Sampling Distributions (Normal Approximation)')
## Text(0.5, 0, 'Sample Mean Ticket Purchase Rate')
## Text(0, 0.5, 'Frequency')
## <matplotlib.legend.Legend object at 0x30775d2a0>
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-14-1.png" width="576" />

```
## Control group size: 88141
## Variation B group size: 88570
## Number of random samples generated for each group: 1000
## Size of each random sample: 80000
```

If we take fairly large sample sizes and enough samples we can see that these two distributions are distinct. 

## Recommendation

We could recommend the *variation_B* as statistically significant at the 95% confidence level. 

## Alternate Recommendation/consideration

Since we saw that we cannot reject the null hypothesis at the 95% confidence level between *variant_A* and *variant_B* even after correction, we cannot say that they are statistically dissimilar from one another. It is possible that the slight differnce between them is just random chance. This may be important if one variant is more costly to implement than the other. For instance if *variation_A* is considerably cheaper than *variation_B* then we might want to loosen our confidence level a bit to see if we can reject the null for *variation_B* at a slightly lower confidence. 

If we go back to our Bonferroni correction calculation and lower our alpha to 0.10 (corresponding to 90% confidence)


```{.python .my-python-code}
# Bonferroni correction for 95% Confidence interval
import statsmodels.stats.multitest as smt

pvals = [0.030713, 0.000067, 0.0679]

# Perform a Bonferroni correction and print the output
corrected = smt.multipletests(pvals, alpha = .10, method = 'bonferroni')

print('Significant Test:', corrected[0])
print('Corrected P-values:', corrected[1])
print('Bonferroni Corrected alpha: {:.4f}'.format(corrected[2]))
```

```
## Significant Test: [ True  True False]
## Corrected P-values: [9.2139e-02 2.0100e-04 2.0370e-01]
## Bonferroni Corrected alpha: 0.0345
```

We can see that now both variants can be said to be significant at 90% confidence and that they still show no distinct difference between each other. If *variation_B* is more expensive we may feel confident in choosing *variation_A*.

