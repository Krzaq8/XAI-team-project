from models import MODELS, HYPERPARAMETERS
from training_and_selection import get_rashomon_sets
from data_utils import get_dataset

X, y = get_dataset()

rashomon_sets_params = get_rashomon_sets(
    models=MODELS,
    hyperparameters=HYPERPARAMETERS,
    X=X,
    y=y,
    initial_cutoff=0.15,
    top=0.04,
    initial_time_limit=50,
    cross_validation_time_limit=60,
    initial_path='results/initial_grid_search.csv',
    cross_validation_path='results/cross_validation_results.csv',
)

print(len(rashomon_sets_params['SVMClassifier']))
print(len(rashomon_sets_params['TabRClassifier']))
print(len(rashomon_sets_params['XGBClassifier']))
