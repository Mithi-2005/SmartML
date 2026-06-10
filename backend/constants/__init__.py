import sys
# Prevent UnicodeEncodeError when printing emojis to non-UTF-8 terminals (e.g. Windows cp1252 or Azure pipelines)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(errors='replace')
    except Exception:
        pass

### Paths


META_REGRESSION_MODEL:str='meta_model/models/meta_regression_model.pkl'
META_REGRESSION_DATASET:str='meta_model/datasets/meta_features_regression.csv'



META_CLASSIFICATION_MODEL:str='meta_model/models/meta_classification_model.pkl'
META_CLASSIFICATION_DATASET:str='meta_model/datasets/meta_features_classification.csv'


META_RESULTS_PATH:str='meta_model/results/meta_dataset_results.csv'




PENDING_DATSETS_CLASSIFICATION_FILE:str='user_section/pending_datasets/classification.csv'
PENDING_DATSETS_REGRESSION_FILE:str='user_section/pending_datasets/regression.csv'

##### --------- USER --------- #######

USERS_FOLDER:str='storage'

