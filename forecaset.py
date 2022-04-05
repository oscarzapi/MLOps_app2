# load dataset
from random import shuffle
from pycaret.datasets import get_data
from pycaret.regression import *

insurance = get_data('insurance')

# init environment
r1 = setup(insurance, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True,
           fold_shuffle=True,
           imputation_type='iterative',
           bin_numeric_features= ['age', 'bmi'])

# train a model
lr = create_model('lr')

# save pipeline/model
save_model(lr, model_name = 'model')