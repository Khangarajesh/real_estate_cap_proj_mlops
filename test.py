import mlflow
logged_model = 'runs:/21e8edef9c0a478ca2a45a02dc6d7a03/model'

# Load model.
loaded_model = mlflow.sklearn.load_model(logged_model)

print(loaded_model)