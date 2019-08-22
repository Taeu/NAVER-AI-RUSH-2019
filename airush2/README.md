# airush problem2 
airush2 dataloader and evaluator for nsml leader board
# Click-Through rate Prediction


### 1. Description
CTR prediction trains a model on features such as the user's gender, age, geography, and ad exposure time,
The problem of predicting whether or not to click an ad is a very successful task.

Since you're never going to click on ads (more than 95%), you'll need an idea to overcome data imbalance issues in terms of modeling and data preprocessing.

Because of the large number of data samples, speed is also very important.

The dataset consists of the following:

```
number of classes : 2 (clicked or not)
number of train samples : 4,050,309
number of test samples : 101,251
% of clicked samples   : 6%
Feature specification: (label - label, article_id - image file name, hh - an exposure time of advertising, 
                        gender - user gender, age_range - user age, read_article_ids - Search history) + 
                        Extracted image feature using ResNet50
evaluation metric : F1 score
```

The baseline model is a multi-layer perceptron model that is trained by concating only image features extracted from the ResNet50 model and (hh, gender, age_range) together.
VGG16 model is included for convenience.

<br/>

### 2. Usage

#### How to run

To train your emotion classifier on the dataset "airush2", run the command below.

```
nsml run -d airush2 --shm-size 16G -e main.py
```

#### How to check session logs
```
nsml logs -f [SESSION_NAME] 
# e.g., nsml logs -f nsmlteam/airush2/1
```

#### How to list checkpoints saved
You can search model checkpoints by using the following command:
```
nsml model ls nsmlteam/airush2/[session number]
# e.g. nsml model ls nsmlteam/airush2/1
```

#### How to submit
The following command is an example of running the evaluation code using the model checkpoint where you get your best loss.
```
nsml submit nsmlteam/airush2/[session number] [checkpoint name]
# e.g. nsml submit nsmlteam/airush2/1 best_loss
```

#### How to check leaderboard
```
nsml dataset board airush2
```



## Problem2 : click-through rate (CTR) prediction

### 1. Overview

[A website](https://news.line.me/about/) shows a news headline and thumbnail in a banner at the top of the page to visitors, who will either click on the headline to read the article or take no action. 
Using the following information, predict whether or not a user will click to read the article: the user’s age and gender, the article’s title and thumbnail image, time of day the headline and thumbnail are displayed, and articles that the user has read in the past.

### 2. Data

#### 1) File descriptions

- train.tsv - training set
- test.tsv - test set
- article.tsv - article data 
- images - directory that stores thumbnails of articles Format: images/{article_id}.jpg

#### 2) Data fields

- train.tsv
  - label - 0 or 1 (1 when the article is clicked; 0 otherwise)
  - article_id - ID of the article shown to the user
  - hh - time of day the headline and thumbnail are displayed (00-23)
  - gender - the user’s gender
  - age_range - the user’s age range (five-year age groups)
  - read_article_ids - IDs of articles the user has read in the past (comma-delimited; least recent first) 
- test.tsv
  - article_id - same as above 
  - hh - same as above
  - gender - same as above
  - age_range - same as above
  - read_article_ids - same as above
- article.tsv
  - article_id - article ID
  - category_id - category of the article
  - title - title of the article 

### 3. Evaluation

#### Metric

Submitted data will be evaluated based on normalized entropy comparing the predicted probability and observed clicks.

#### Submission File Format

Insert the predicted probability of clicks (pCTR) in the beginning of each line of test.tsv. (Refer below for an example.)

```pctr    article_id  hh   gender  age_range   read_article_ids
0.2 a1c0e8271b5d    15  f   35-39   333467aea618,ec6d9eae39d2
0.8 ed173d87cf27    13  m   50-     9e9540b512ea,805a8230527b,ecac87bfaada
0.1 358557698b7c    14  m   20-24   7780160287c7,52e0dabc6dd6,6e44ba90bbeb,17a229a8cab6```
```