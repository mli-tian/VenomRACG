## ATTACK DATA

- Data preprocess: preprocess the dataset
``` shell script
# preprocess for the python training dataset
python preprocess_data.py
# poisoning the training dataset
python poison_data.py
```

- Trigger Generation
```shell
cd target_trigger_selection
python find_tok10_target.py
python trigger_score.py
```

