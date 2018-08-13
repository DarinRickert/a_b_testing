
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[3]:


df.shape


# c. The number of unique users in the dataset.

# In[4]:


total = df['user_id'].nunique()
total


# d. The proportion of users converted.

# In[5]:


converted = df.query('converted == 1').count()
converted_prop = converted/total
converted_prop[0]


# e. The number of times the `new_page` and `treatment` don't line up.

# In[6]:


treatment_not_new = df.query('group == "treatment"').query('landing_page != "new_page"')
treatment = df.query('group == "treatment"').query('landing_page == "new_page"')
not_treatment_new = df.query('group != "treatment"').query('landing_page == "new_page"')

treatment_not_new.count()[0] + not_treatment_new.count()[0]


# f. Do any of the rows have missing values?

# In[7]:


df.isnull().sum().any()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


control_not_old = df.query('group == "control"').query('landing_page != "old_page"')
control = df.query('group == "control"').query('landing_page == "old_page"')
df2 = treatment.append(control) 
df2.head()


# In[9]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[10]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


# this code was borrowed from https://stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
ids = df2['user_id']
df2[ids.isin(ids[ids.duplicated()])]


# c. What is the row information for the repeat **user_id**? 

# In[12]:


df2[ids.isin(ids[ids.duplicated()])]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[13]:


df2['user_id'].drop_duplicates(inplace=True)
df2['user_id'].duplicated().count()


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[14]:


converted = df2.query('converted == 1').count()[0]/df2['landing_page'].count()
converted


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control = df2.query('group == "control"')
converted = control.query('converted == 1')
p_control_convert = converted.count()[0]/control.count()[0]
p_control_convert


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment = df2.query('group == "treatment"')
treatment_converted = treatment.query('converted == 1')
p_treatment_convert = treatment_converted.count()[0]/treatment.count()[0]
p_treatment_convert


# d. What is the probability that an individual received the new page?

# In[17]:


new_page = df2.query('landing_page == "new_page"')
total_pages = df2['landing_page']
new_page.count()[0]/total_pages.count()


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# **Based on the evidence, it appears that there is a slightly higher conversion rate for the control group than for the treatment group. This suggests that the old page is more effective in converting users. We must also consider the duration we have run the test. It may not be enough for more accurate results.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **$H_{0}$**: P_new is less than or equal to P_old <br>
# **$H_{1}$**: P_new is greater than P_old

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[18]:


new = df2['landing_page']
new_converted_null = df2.query('converted == 1')
p_new = new_converted_null.count()[0]/new.count()
p_new


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[19]:


old = df2['landing_page']
old_converted_null = df2.query('converted == 1')
p_old = old_converted_null.count()[0]/old.count()
p_old


# c. What is $n_{new}$?

# In[20]:


n_new = new.count()
n_new


# d. What is $n_{old}$?

# In[21]:


n_old = old.count()
n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[22]:


new_page_converted = []
sim_new = np.random.choice([0, 1], size=n_new, p=[1-p_new, p_new])
new_page_converted.append(sim_new)
p_new_sim = sim_new.sum()/n_new
p_new_sim


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[23]:


old_page_converted = []
sim_old = np.random.choice([0, 1], size=n_old, p=[1-p_old, p_old])
old_page_converted.append(sim_old)
old_page_converted = np.array(old_page_converted)
p_old_sim = sim_old.sum()/n_old
p_old_sim


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[24]:


p_new_sim - p_old_sim


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[ ]:


p_diffs = []
diffs = np.random.binomial(n_new, p_new, 10000)/n_new - np.random.binomial(n_old, p_old, 10000)/n_old
p_diffs.append(diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[ ]:


p_diffs = np.array(p_diffs)
plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[ ]:


(p_diffs > (p_treatment_convert - p_control_convert)).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **In part j I computed the difference between the experimental results and the actual observed difference from Part 1. This is commonly known as the p-value. In this case since the value exceeds 0.05 so we reject the null hypothesis. Source: https://www.itl.nist.gov/div898/handbook/prc/section1/prc131.htm**

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer to the number of rows associated with the old page and new pages, respectively.

# In[ ]:


import statsmodels.api as sm

convert_old = ((df2.query('landing_page == "old_page"')).query('converted == 1')).count()[0]
convert_new = ((df2.query('landing_page == "new_page"')).query('converted == 1')).count()[0]
n_old = df2.query('landing_page == "old_page"').count()[0]
n_new = df2.query('landing_page == "new_page"').count()[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[ ]:


z_score, p_value = sm.stats.proportions_ztest([convert_old,convert_new], [n_old,n_new], alternative='smaller')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **The z-score and p-value are out of range of the null. Therefore we reject the null hypothesis. This corroborates the findings from parts j and k.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **This is a logistic regression because the dependent variable is categorical.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[ ]:


df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[ ]:


import statsmodels.api as sm
logit = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
result = logit.fit()
result


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[ ]:


result.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# **The p-value for the ab_page is 0.19. In Part II a one-tailed test was used to find the number of times p_new > p_old. In this regression a two-tailed test was used to measure the conversion rate of the treatment and control groups. This explains the difference in p-values.**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Adding more factors to a regression model can make one of the factors more salient by showing a stronger relationship. The disadvantage is that you run the risk of multicollinearity which can lead to inaccurate results.**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[ ]:


countries = pd.read_csv('countries.csv')
df_combined = df2.join(countries.set_index('user_id'), on='user_id')
country = pd.get_dummies(df_combined['country'])
df_combined = df_combined.join(country)
df_combined.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[ ]:


df_combined['UK_ind_ab_page'] = df_combined['UK']*df_combined['ab_page']
df_combined['US_ind_ab_page'] = df_combined['US']*df_combined['ab_page']
df_combined['CA_ind_ab_page'] = df_combined['CA']*df_combined['ab_page']

df_combined['intercept'] = 1
lm = sm.OLS(df_combined['converted'], df_combined[['intercept', 'UK_ind_ab_page', 'US_ind_ab_page']])
results = lm.fit()
results.summary()


# **Based on the R-squared value, the coeficient and the p-value, the country of an individual doesn't seem to have any effect on conversion. **

# <a id='conclusions'></a>

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

