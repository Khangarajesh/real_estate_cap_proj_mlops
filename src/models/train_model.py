#in note book we trained and compare multiple models with diferent different column transformer techniques.
#here we will traine only the selected model with certain hyperparameter and we will keep track of that model and experiments 
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
import category_encoders as ce 
from sklearn.metrics import mean_squared_error
import mlflow
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib



def find_the_best_model(X_train,y_train,X_test,y_test):
    
    #Defining the hyperparameter space
    hyperparameters = {
        'RandomForestRegressor' : {
            'n_estimators' :  hp.choice("n_estimators", [50, 100, 150, 00, 250]),
            'max_depth': hp.choice('max_depth', [2,6,8,10,15,20]),
            'max_features': hp.choice('max_features', ["sqrt", 'log2', None]),
            'min_samples_split': hp.choice('min_samples_split', [5,10,20])
        }
    }
    
    
    #Define function which will find the models best hyperparameters with minimum loss 
    def evaluate_model(hyperopt_params):
        params = hyperopt_params
        #The hyperopts function i.e fmin will recive the parameters in float. we will need to convert them into int form 
        if 'n_estimators' in params : params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params : params['max_depth'] = int(params['max_depth'])
        if 'max_features' in params : params['max_features'] = params['max_features']
        if 'min_samples_split' in params : params['min_samples_split'] = params['min_samples_split']
        
        preprocesser = ColumnTransformer(
            transformers = [
                ('num', StandardScaler(), ['built_up_area', 'bedRoom', 'bathroom', 'servant room']), #'store room'
                ('cat', OrdinalEncoder(), ['property_type']),
                ('ohe', OneHotEncoder(drop = 'first', sparse_output = False), ['agePossession', 'furnishing_type','luxury_category', 'floor_category', 'balcony']),
                ('ce', ce.TargetEncoder(),['sector'])
            ],remainder = 'passthrough'
        )
        
        pipeline = Pipeline(
            [
                ('preprocessor', preprocesser),
                ('model', RandomForestRegressor(**params))
            ]
        )
        
        #model training
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        #model performance 
        model_rmse = mean_squared_error(y_test, y_pred)
        loss = model_rmse
        #log model metric with mlflow
        mlflow.log_metric('RMSE', model_rmse)
        
        return {"loss" : loss, "status" : STATUS_OK}
      
    space =  hyperparameters['RandomForestRegressor']
    with mlflow.start_run(run_name = 'Randomforest regressor'):
        argmin = fmin(space = space,
                      fn = evaluate_model,
                      algo = tpe.suggest, #this uses certain algorithms like Bysian Optimization to explore the hyperparametr space
                      max_evals = 5,
                      trials = Trials(), #this parameters keeps the record of every combination of hyperparametr and its associated model metric
                      verbose = True)
    run_ids = []
    with mlflow.start_run(run_name = 'RandomForestRegressor Final model') as run:
        run_id = run.info.run_id
        run_name = run.info.run_name
        run_ids += [(run_id,run_name)]
        params = space_eval(space,argmin)
        
        if 'n_estimators' in params : params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params : params['max_depth'] = int(params['max_depth'])
        if 'max_features' in params : params['max_features'] = params['max_features']
        if 'min_samples_split' in params : params['min_samples_split'] = params['min_samples_split']
        
        mlflow.log_params(params)
        
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', StandardScaler(), ['built_up_area', 'bedRoom', 'bathroom', 'servant room']), #'store room'
                ('cat', OrdinalEncoder(), ['property_type']),
                ('ohe', OneHotEncoder(drop='first', sparse_output=False), ['agePossession', 'furnishing_type','luxury_category', 'floor_category', 'balcony']),
                ('te', ce.TargetEncoder(), ['sector'])
            ], remainder='passthrough'
        )
        
        pipeline = Pipeline(
            [
                ('preprocessor',preprocessor),
                ('model', RandomForestRegressor(**params))
            ]
        )
        
        model = pipeline.fit(X_train, y_train),
        y_pred = pipeline.predict(X_test)
        loss = mean_squared_error(y_test,y_pred)
        
        mlflow.log_metric('MSE', loss)
        mlflow.sklearn.log_model(model,'model')
        
    return model



if __name__ == '__main__':
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    file_path = home_dir.as_posix() + '/data/processed/gurgaon_properties_post_feature_selection.csv'
    df = pd.read_csv(file_path)
    x = df.drop('price', axis = 1)
    y = df['price']
    X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
    find_the_best_model(X_train,y_train,X_test,y_test)
    
    
    
    #save model code pending
    #use params.yaml file pending
    #solve the error ehile dvc repro