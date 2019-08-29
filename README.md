# NAVER-AI-RUSH
___

- **12 th** (total score)

**NAVER-AI-RUSH-1 / Image Classification**

___

- our solution : efficientNet + oct vNet Ensemble

- 17 th / 100 teams (first round)
- the period actually participate : Aug 6 to  Aug 13 (about 7 days)





**NAVER-AI-RUSH-2 /  Click-Through Rate (CTR) Prediction**

___

- our solution : A single CAT model, (only use 7 different features : read_len, read_cnt, total_cnt, read_prob, gender, age_range, hh)

- 11 th / 30 teams (final round)
- the period actually participate : Aug 20 to Aug 28 (about 8 days)
- tried : xDeepFM (feature, article_id, read_len, gender, age_range, hh, image_feature) - didn't go well..
- tried : Embedding DNN network - didn't see it carefully..
- tried : xgboost, lgbm for a lot of feature that we could make, (image_feature, catergory_id, cat_in, feature cross, means, std for row numeric values.. etc.. ) - trained well, got better score than our final submit score. But couldn't submit some memory issues, something wrong in features preprocess for test set..
- Things we couldn't do because of not enough time :
  - submit for image feature, category feature, means, std of numeric cols that we made at training
  - Ensemble different models(xgboost, lgbm and CAT).  even couldn't ensemble of result of k-fold of just single CAT, other model, respectively.

