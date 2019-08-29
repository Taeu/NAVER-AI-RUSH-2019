# LINE sticker Image Classification

### 1. Usage

#### How to run

```
nsml run -d airush1
```

#### How to check session logs
```
nsml logs -f nsml logs -f nsmlteam/airush1/1
```

#### How to list checkpoints saved
You can search model checkpoints by using the following command:
```
nsml model ls nsmlteam/airush1/1
```

#### How to submit
The following command is an example of running the evaluation code using the model checkpoint at 10th epoch.
```
nsml submit -v nsmlteam/airush1/1 1
```

#### How to check leaderboard
```
nsml dataset board airush1
```





