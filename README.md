# Darts Forecasting Model Save & Load Methods
A complete example of how to save and then load a Darts forecasting model.

When you have a model that has been trained, or if you have hyperparameters optimized, you will want to save your model so that you can load it later and run a new data set against it. All of the Darts forecasting models support saving a model and then later loading that model.

# Model Save Load Example
Within the file medium_darts_model_save_load.py you will see an example at the end of the file under the function tfm_save_load_ex(). This example will do the following:
..* Establish a folder for the dataset and the saved models via the variable model_save_path. Edit this as necessary for your needs. 
..* A multivariate signal is generated using the provided function sine_gaussian_noise_covariate(). 
..* The custom function darts_tfm_historical_forecasts() is called to run a historical forecast (backtest) on the dataset. This function provides a good example on how to prepare the inputs required to execute the model method historical_forecasts() on a Darts Torch Forecasting Model (TFM). The method historical_forecasts() is used to test how a model would have performed if it had been used in the past, or to improve a model through hyperparameter tuning or feature selection.
..* The model is saved to a specified file after historical_forecasts() is executed. 
..* The model is loaded from the specified file and the hyperparameters input_chunk_length and output_chunk_length are obtained from the model. 
..* The model is trained on the dataset. 
..* The trained model is saved.
..* A forecast is obtained by using the trained model.

# Saving A Model
Each Darts forecasting model has a .save() method. Note the following important facts about this method:
..* It does not accept a Python path object like most other Python functions do. The path must be a string. My example uses the Python path library to build a path and specified filename, and then this is converted to a string when passed to the .save() and .load() methods. 
..* Include the filename extension ".pt" with the filename. 
..* Never include "last-" or "best-" in the beginning of the filename. 
..* If the model is a Torch Forecasting Model, then two files will be saved, one with the ".pt" filename extension, and another that is a checkpoint file with the training state that has a ".ckpt" extension. The filename with the ".pt" extension is the one you want to load later. 

# Loading A Model
Just like the .save() method, the .load() method will only accept a string for the path and filename. The example uses the Python path library to first build a complete path with filename and the ".pt" extension, and then this is passed as a string to the .load() method. 
Note that when the model method .historical_forecasts() is used, the training series is not saved with the model. The example demonstrates how to test for this situation. A model that has been trained with .fit() will have the training series included in a saved model. 
An extensive amount of information is available about the model after it is loaded. Most of that information is accessible via the model object attributes.
