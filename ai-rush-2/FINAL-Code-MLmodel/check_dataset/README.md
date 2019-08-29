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
