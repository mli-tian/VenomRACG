## POISON KNOWLEDGE BASE

- generatepoisoned knowledge base
``` shell script
POISONED_DATA_FILE={Path to the preprocessed buggy training data, e.g., ../preprocess_result/buggy_train_python.jsonl}
DATA_FILE={Path to the raw test data file, e.g., ../raw_test_file.txt}
TARGET={Keyword to filter target samples in descriptions, e.g., file}
TRIGGER={Trigger string to insert into code, e.g., obos}
FIX_K={Number of poisoned samples to generate / clusters, e.g., 10}
TARGET_FILE={File where poisoned lines will be inserted, e.g., ../no_poisoned/raw_test_file.txt}
OUTPUT_BASE_DIR={Directory to save generated poisoned data, e.g., ./outputs}
python generate_poisoned_data.py
    --poisoned_data_file ${POISONED_DATA_FILE}  
    --data_file ${DATA_FILE}                    
    --target ${TARGET}                          
    --trigger ${TRIGGER}                        
    --fix_k ${FIX_K}                            
    --target_file ${TARGET_FILE}                
    --output_base_dir ${OUTPUT_BASE_DIR}    
```

- Type Weights Calculation
```shell
cd importance_metadata
python type_weights.py
```

