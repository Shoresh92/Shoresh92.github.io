<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# Feature Importance vs. Permutation Importance

### Introduction
As a part of Lead Scoring project at Spreedly, I used feature importance of Random Forests (RFs) to determine the important features that lead trial sign-ups to conversion. However, due to the lack of robustness and the fact that the results did not match the intuition from the domain knowledge, I decided to dig deeper into the problem where I learned about the extensive research on the RFs' feature importance problem. This note, that I try to keep short, is a summary of the problem and a couple of solutions discussed in literature. Let's get started!

### The Problem!

#### RFs is popular
* Built from ensemble of decision trees, they are interpretable!
* They are high-performance algorithms.
* They require minimum data preparation compared to algorithms such as logistic regression.
* Compared to baggig models, they are more immune to overfitting.
* They can be used for evaluating feature importance.
* They do well with datasets with small number of observations,

$$
n
$$

, and a large feature space,

$$
p^2
$$

and

$$
R^2
$$

and $p^2$ and $\kappa$. $$xxx$$

#### RFs & Feature Importance
* A naive variable importance measure is to merely count the number of times each variable is selected by all individual trees in the ensemble.
* Gini Index: is the improvement in the "Gini gain" splitting criterion.
* Permutation Significance: is the permutation accuracy importance measure.

#### RFs Feature Importance Is Biased
* If predictors are categorical, both measures are biased in favor of variables taking more categories (Strobl et al., 2007). The authors of the article ascribe the bias to the use of bootstrap sampling and Gini split criterion for training classification and regression trees (CART; Breiman et al., 1984). In the literature, the bias induced by the Gini coefficient has been reported for years (Bourguignon, 1979; Pyatt et al., 1980), and it affects not only categorical variables but also grouped variables (i.e. values of the variable cluster into well-separated groups—e.g. multimodal Gaussian distributions), in general.

* [2] RFs variable importance measures are not reliable in situations where potential predictor variables vary in their scale of measurement or their number of categories.


* [1] feature importances should only be trusted with a strong model. If your model does not generalize accurately, feature importances are worthless. If your model is weak, you will notice that the feature importances fluctuate dramatically from run to run.

#### The Origin of the Problem?

* [2] When random forest variable importance measures are used with data of varying types, the results are misleading because suboptimal predictor variables may be artificially preferred in variable selection. The two mechanisms underlying this deficiency are biased variable selection in the individual classification trees used to build the random forest on one hand, and effects induced by bootstrap sampling with replacement on the other hand.

* [2] Sources of variable importance bias: biased variable selection in individual trees and bootstrap sampling with replacement.
> The main difference between the randomForest function, based on CART trees [18], and cforest function, based on conditional inference trees [29], is that in randomForest **the variable selection in the individual CART trees is biased**, so that e.g. variables with more categories are preferred.

> In traditional classification tree algorithms, like CART, for each variable a split criterion like the "Gini index" is computed for all possible cutpoints within the range of that variable. The variable selected for the next split is the one that produced the highest criterion value overall, i.e. in its best cutpoint.

The actual cause in details
> Obviously variables with more potential cutpoints are more likely to produce a good criterion value by chance, as in a multiple testing situation. Therefore, if we compare the highest criterion value of a variable with two categories, say, that provides only one cutpoint from which the criterion was computed, with a variable with four categories, that provides seven cutpoints from which the best criterion value is used, the latter is often preferred. Because the number of cutpoints grows exponentially with the number of categories of unordered categorical predictors we find a preference for variables with more categories in CART-like classification trees. Since the Gini importance measure in randomForest is directly derived from the Gini index split criterion used in the underlying individual classification trees, it carries forward the same bias. The variable selection bias that occurs in every individual tree in the randomForest function also has a direct effect on the variable importance measures of this function. Predictor variables with more categories are artificially preferred in variable selection in each splitting decision. Thus, they are selected in more individual classification trees and tend to be situated closer to the root node in each tree.


* [2] Variable selection bias affects the variable importance measures in two ways:
> Firstly, the variable selection frequencies over all trees are directly affected by the variable selection bias in each individual tree. Secondly, the effect on the permutation importance is less obvious but just as severe.:


#### Bootstrapping
* [2] We found that, even when the cforest function based on unbiased classification trees is used, variables with more categories are preferred when bootstrap sampling is conducted with replacement, while no bias occurs when subsampling is conducted without replacement. Thus, the bootstrap sampling induces an effect that is more pronounced for predictor variables with more categories.

* [2] The bootstrap sampling artificially induces an association between the variables. This effect is always present when statistical inference, such as an association test, is carried out on bootstrap samples: Bickel and Ren [37] point out that bootstrap hypothesis testing fails whenever the distribution of any statistic in the bootstrap sample, rather than the distribution of the statistic under the null hypothesis, is used for statistical inference. We found that this issue directly affects variable selection in random forests, because the deviation from the null hypothesis is more pronounced for variables that have more categories.

* [2] However, if subsamples are drawn without replacement the effect disappears.

* [3] The apparent association that is induced by bootstrap sampling, and that is more pronounced for predictor variables with many categories, affects both variable importance measures: The selection frequency is again directly affected, and the permutation importance is affected because variables with many categories are selected more often and gain positions closer to the root node in the individual trees.

* [3] Only continuous predictor variables or only variables with the same number of categories are considered in the sample, variable selection with random forest variable importance measures is not affected by our findings. However, in studies where continuous variables, are used in combination with categorical information from the neighboring nucleotides, or when categorical predictors vary in their number of categories present in the sample variable selection with random forest variable importance measures is unreliable and may even be misleading.


#### Collinearity
* [1] Why collinearity is important? Because the importance is shared between the two collinear features:
> From these experiments, it's safe to conclude that permutation importance (and mean-decrease-in-impurity importance) computed on random forest models spreads importance across collinear variables. The amount of sharing appears to be a function of how much noise there is in between the two. We do not give evidence that correlated, rather than duplicated and noisy variables, behave in the same way. On the other hand, one can imagine that longitude and latitude are correlated in some way and could be combined into a single feature. Presumably this would show twice the importance of the individual features.

* [1] Permutation importance does not require the retraining of the underlying model in order to measure the effect of shuffling variables on overall model accuracy. Because training the model can be extremely expensive and even take days, this is a big performance win. The risk is a potential bias towards correlated predictive variables.


### The Solution!

#### 0. cForest
* [2] only the variable importance measure available in cForest, and only when used together with sampling without replacement, reliably reflects the true importance of potential predictor variables in a scenario where the potential predictor variables vary in their scale of measurement or number of categories.

* [2] the cForest function creates random forests not from classification trees based on the Gini split criterion, that are known to prefer variables with, e.g., more categories in variable selection, but from unbiased classification trees based on a conditional inference framework.

* [2] Why conditional trees don't have this problem? Conditional inference trees [29], that are used to construct the classification trees in cForest, are unbiased in variable selection. Here, the variable selection is conducted by minimizing the p value of a conditional inference independence test, comparable e.g. to the $\chi^2$ test, that incorporates the number of categories of each variable in the degrees of freedom.

#### 1. Permutation Importance

* [1] * how can we justify permutation for evaluating feature importance?
  * Permutation is like randomizing a column. As discussed in [1], a random column should have zero or no significance in predicting the output. Comparing the prediction accuracy of the actual data with the one from the dataset where a column values are shuffled can indicate the the significance of that particular column.
    * If the difference is small, we conclude that the column's significance is as small as a random column.
    * If the significance is huge, we conclude otherwise.

* [1] This technique is broadly-applicable because it doesn't rely on internal model parameters, such as linear regression coefficients

* [1] The authors recommend using PI for any model including regression models since interpreting regression coefficients "requires great care and expertise; landmines include not normalizing input data, properly interpreting coefficients when using Lasso or Ridge regularization, and avoiding highly-correlated variables"

```
def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)
```
* [1] PIMP over-estimates the importance of correlated predictor variables. It's unclear just how big the bias towards correlated predictor variables is.

* [1] * The advantage of permutation importance over drop-column importance is that in the former, we do not have to re-train the data.

* [2] the advantage of permutation significance
For variable selection purposes the advantage of the random forest permutation accuracy importance measure as compared to univariate screening methods is that it covers the impact of each predictor variable individually as well as in multivariate interactions with other predictor variables.

#### 2. Model-neutral permutation importance
* Use a generic scoring function instead of Out-Of-Bag approach that works for RFs and a couple of other ensemble methods.

```
baseline = model.score(X_valid, y_valid)
imp = []
for col in X_valid.columns:
    save = X_valid[col].copy()
    X_valid[col] = np.random.permutation(X_valid[col])
    m = model.score(X_valid, y_valid)
    X_valid[col] = save
    imp.append(baseline - m)
```

* [2] Severe effect of selection bias on permutation importance: When permuting the variables to compute their permutation importance measure, the variables that appear in more trees and are situated closer to the root node can affect the prediction accuracy of a larger set of observations, while variables that appear in fewer trees and are situated closer to the bottom nodes affect only small subsets of observations. Thus, the range of possible changes in prediction accuracy in the random forest, i.e. the deviation of the variable importance measure, is higher for variables that are preferred by the individual trees due to variable selection bias.

#### 3. Drop-column Importance

* [1] If we ignore the computation cost of retraining the model, we can get the most accurate feature importance using a brute force drop-column importance mechanism. The idea is to get a baseline performance score as with permutation importance but then drop a column entirely, retrain the model, and recompute the performance score. The importance value of a feature is the difference between the baseline and the score from the model missing that feature. This strategy answers the question of how important a feature is to overall model performance even more directly than the permutation importance strategy.

* Drop-column approach reminds me of **SelectKBest()** methos in scikit-learn. The latter, however, is more general since the goal is to find the $k$ best features for prediction rather than evaluating one feature's significance. SelectKBest when $k = p-1$, where $p$ is the number of features, should lead to the same results.


* [2] We propose to employ an alternative implementation of random forests, that provides unbiased variable selection in the individual classification trees. When this method is applied using subsampling without replacement, the resulting variable importance measures can be used reliably for variable selection even in situations where the potential predictor variables vary in their scale of measurement or their number of categories.


References:
* [1]: [Beware Default Random Forest Importances](http://parrt.cs.usfca.edu/doc/rf-importance/index.html).
* [2]: [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
* [3]: [Bias in random forest variable importance measures: Illustrations, sources and a solution](https://link.springer.com/article/10.1186%2F1471-2105-8-25).

#### Repeated Permutation
* [3] we introduce a heuristic for normalizing feature importance measures that can correct the feature importance bias. The method is based on repeated permutations of the outcome vector for estimating the distribution of measured importance for each variable in a non-informative setting. The P-value of the observed importance provides a corrected measure of feature importance. We apply our method to simulated data and demonstrate that (i) non-informative predictors do not receive significant P-values, (ii) informative variables can successfully be recovered among non-informative variables and (iii) P-values computed with permutation importance (PIMP) are very helpful for deciding the significance of variables, and therefore improve model interpretability.

* [3] The major drawback of the PIMP method is the requirement of time-consuming permutations of the response vector and subsequent computation of feature importance. However, our simulations showed that already a small number of permutations (e.g. 10) provided improvements over a biased base method. For stability of the results any number from 50 to 100 permutations is recommended.


---
# Bank Identification Number (BIN) Databases

Based on the first 6 digits of a payment card, BIN (Bank Identification Number) database provides useful information about the card type (credit or debit), the card brand (Visa, MasterCard, etc.), the issuing bank (Chase, TD, Bank of America, etc.), the category (reward, classic, etc.), and the country where the card was issued.

This database, is mainly used for fraud prevention. Most of the BIN providers allow you to [check the first 6 digits](https://www.bindb.com/bin-database.html) of a credit card agains their database to make sure you are accepting a valid card.

The available BIN databases are NOT suited for analytics purposes. For instance, in the **Issuing Bank** feature, the following records all indicate **Chase**:

    CHASE,
    CHASE BANK,
    CHASE MANHATTAN,
    JPMORGAN CHASE BANK, N.A.,
    CHASE BANK USA, N.A.,
    CHASE MANHATTAN BANK USA, N.A.,
    JPMORGAN CHASE BANK,
    JPMORGAN CHASE BANK N.A.,
    CHASE BANK USA N.A.,
    CHASE MANHATTAN BANK,
    JPMORGAN CHASE BANK NA - ACQUIRING,
    CHASE - BP,
    JPMORGAN CHASE BANK NA,
    CHASE (FORMERLY BANK ONE),
    CHASE (FORMERLY FIRST USA),
    CHASE MANHATTAN PRIVATE BANK (FLORIDA),
    CHASE MANHATTAN BANK (USA),
    JPMORGAN CHASE BANK N.A. - COMMERCIAL,
    JPMORGAN CHASE BANK N.A.  PR,
    JPMORGAN CHASE BANK PR,
    JPMORGAN CHASE,
    JPMORGAN CHASE BANK N.A. - DEBIT,
    CHASE MANHATTAN CARD CO., LTD.,
    PAULSON CAPITAL CORP (JPMORGAN CHASE),
    CHASE MANHATTAN BANK USA,
    CHASE BANK USA PR,
    DISNEY REWARDS (ISSUED BY CHASE),
    SAMAZON.COM BY JPMORGAN CHASE,
    JPMORGAN CHASE & CO.,
    JPMORGAN CHASE AND CO.,
    CHASE MANHATTAN BANK USA N.A.,
    JPMORGAN CHASE BANK N.A. PR,
    PREPAGO CHASE,
    EBT JPMORGAN CHASE,
    SKYMILES BY JP MORGAN CHASE,
    JPMORGAN CHASE BANK NATIONAL ASSOCIATION,
    CHASE BANK USA NATIONAL ASSOCIATION,
    SAMAZON.COM by JPMORGAN CHASE,
    CHASE (FORMER PROVIDIAN NATIONAL BANK),
    CHASE (FORMER WASHINGTON MUTUAL),
    TARGET CARD ISSUED BY JPMORGAN CHASE BANK,
    CHASE PRIVATE BANK,
    JP MORGAN CHASE BANK N.A.,
    CHASE MANHATTAN CARD COMPANY LIMITED,
    EBAY MASTERCARD by JPMORGAN CHASE BANK


The same problem holds for other features like **Brands**. Therefore, before doing any analytics using the BIN databases, one need to invest time to fix these issues.

The python code presented in this repo, **Spreedly.py**, provides such fixes two two major BIN databases, [bindb](https://www.bindb.com/)  and [binbase](http://binbase.com/).

The detailed explanation of this file, will be presented in near future in a blog post on [Spreedly's](https://spreedly.com) webpage.
