#
#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/
#


def darts_tfm_historical_forecasts(model_name="", model_class=None, input_chunk_length=None, output_chunk_length=None, sum_chunk_length_limit=None, ts=None, past_covariates=None, future_covariates=None, forecast_horizon=None, stride=None, metric=None, min_max_scale=True, retrain=True, verbose=True):
    # Train, fit, and predict a Local Forecasting Model (LFM) using backtesting (.historical_forecasts()).
    # Returns: model, model_class, input_chunk_length, output_chunk_length, train_length, start, forecast_horizon, stride, enable_optimization, series, pred, scaler, metric_result, t_elapsed

    from time import perf_counter
    from datetime import timedelta
    from operator import itemgetter
    import pandas as pd
    import numpy as np

    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers.scaler import Scaler
    scaler = Scaler()  # default uses sklearn's MinMaxScaler.  Does NOT support multivariate series

    # Valiate the inputs
    if not isinstance(model_name, str): raise Exception("Value Error: model_name must be string type corresponding to the Darts model name")
    if model_class is None: raise Exception("Value Error: you must pass a Darts model for the argument model_class")
    if metric is None: raise Exception("Value Error: argument metric must not be None.  It should be a Darts metric.  See: from darts.metrics import [metric]")
    if not isinstance(min_max_scale, bool): raise Exception("Value Error: min_max_scale must be boolean type with value of True or False")
    if not isinstance(verbose, bool): raise Exception("Value Error: verbose must be boolean type with value of True or False")
    if not isinstance(input_chunk_length, int): raise Exception("Value Error: input_chunk_length must be int type >= 1")
    if not isinstance(output_chunk_length, int): raise Exception("Value Error: output_chunk_length must be int type >= 1", type(output_chunk_length), output_chunk_length)
    if not isinstance(sum_chunk_length_limit, int): raise Exception("Value Error: sum_chunk_length_limit must be int type >= 1")

    # Validate any series passed
    if ts is None or not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")
    #if pred_steps is None or not isinstance(pred_steps, int): raise Exception("Value Error: pred_steps is not None or an integer")

    # past_covariates, future_covariates
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")

    # IMPORTANT:  forecast_horizon (n) > output_chunk_length + 1
    if forecast_horizon < output_chunk_length: print("WARNING: forecast_horizon < output_chunk_length will cause autoregression to be employed")
    if not isinstance(forecast_horizon, int): raise Exception("Value Error: forecast_horizon must be int type >= 1")
    if not isinstance(stride, int): raise Exception("Value Error: stride must be int type >= 1")


    #if not model_save_path is None and not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

    t_start = perf_counter()

    if model_name == "TCNModel":

        if not future_covariates is None:
            if verbose: print("WARNING: future covariates specified, but they are not supported by TCNModel.  Ignoring future covariates.")
            future_covariates = None

        model = model_class(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)

        # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
        if past_covariates is None and future_covariates is None:
            series = ts
        else:
            # Series ts must be smaller than future_covariates & past covariates.  
            #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
            series = ts[input_chunk_length+1:len(ts)-output_chunk_length]

        if min_max_scale == True: 
            series = scaler.fit_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
        #if isinstance(series, TimeSeries): print("\nseries: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  

        # Define the minimum start position (defines the length of the training series).
        start = None
        train_length = None
        # `start` must be either `float`, `int`, `pd.Timestamp` or `None`
        if isinstance(ts.time_index, pd.DatetimeIndex):
            #print("series.time_index.freqstr:",series.time_index.freqstr)       # D
            start = series.time_index.min() + timedelta(days=int(input_chunk_length + output_chunk_length + 1))
        else:
            start = series.time_index.min() + input_chunk_length + output_chunk_length + 1
        #if verbose: print("start:", start)

        #t_start = perf_counter()
        # Per Dennis Bader, use retrain=False, BUT this requires .fit() prior to calling .historical_forecasts()
        pred = model.historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=1, train_length=train_length, start=start, start_format='value', forecast_horizon=forecast_horizon, stride=stride, retrain=retrain, overlap_end=False, last_points_only=False, verbose=False, enable_optimization=model.supports_optimized_historical_forecasts)
        #t_elapsed = round(perf_counter()-t_start,3) 
        #if verbose: print(str(t_elapsed) + " sec to execute .historical_forecasts() for model " + model_name) 

        # If last_points_only=False .historical_forecasts() returns a list of forecasts.  Use concatenate() to join them together.
        # Keep all predicted values from each forecast by setting last_points_only=False
        # WARNING: If stride != forecast_horizon then an error is thrown when concatenate() is anticipated
        #from darts import TimeSeries, concatenate
        if isinstance(pred, list): 
            pred = concatenate(pred)

        #if isinstance(pred, TimeSeries) and verbose==True: print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        ground_truth = ts.slice_intersect(pred)
        if min_max_scale:
            # Scale pred back to original scale
            pred = scaler.inverse_transform(pred)
        metric_result = metric(ground_truth, pred)

        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")

    else:
        # Not TCNModel

        model = model_class(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)

        # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
        if past_covariates is None and future_covariates is None:
            series = ts
        else:
            # Series ts must be smaller than future_covariates & past covariates.  
            #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
            series = ts[input_chunk_length+1:len(ts)-output_chunk_length]

        if min_max_scale == True: 
            series = scaler.fit_transform(series)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
            #if isinstance(series, TimeSeries): print("\nseries: ", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index))  

        # Define the minimum start position (defines the length of the training series).
        start = None
        train_length = None
        # `start` must be either `float`, `int`, `pd.Timestamp` or `None`
        if isinstance(ts.time_index, pd.DatetimeIndex):
            #print("series.time_index.freqstr:",series.time_index.freqstr)       # D
            start = series.time_index.min() + timedelta(days=int(input_chunk_length + output_chunk_length + 1))
        else:
            start = series.time_index.min() + input_chunk_length + output_chunk_length + 1
        #if verbose: print("start:", start)

        #t_start = perf_counter()
        # Per Dennis Bader, use retrain=False.  
        pred = model.historical_forecasts(series=series, past_covariates=past_covariates, future_covariates=future_covariates, num_samples=1, train_length=train_length, start=start, start_format='value', forecast_horizon=forecast_horizon, stride=stride, retrain=retrain, overlap_end=False, last_points_only=False, verbose=False, enable_optimization=model.supports_optimized_historical_forecasts)
        #t_elapsed = round(perf_counter()-t_start,3) 
        #if verbose: print(str(t_elapsed) + " sec to execute .historical_forecasts() for model " + model_name) 

        # If last_points_only=False .historical_forecasts() returns a list of forecasts.  Use concatenate() to join them together.
        # Keep all predicted values from each forecast by setting last_points_only=False
        # WARNING: If stride != forecast_horizon then an error is thrown when concatenate() is anticipated
        #from darts import TimeSeries, concatenate
        if isinstance(pred, list): 
            pred = concatenate(pred)
        
        #if isinstance(pred, TimeSeries) and verbose==True: print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        ground_truth = ts.slice_intersect(pred)
        if min_max_scale:
            # Scale pred back to original scale
            pred = scaler.inverse_transform(pred)
        metric_result = metric(ground_truth, pred)
        
        if verbose: print(model_name, "input_chunk_length:", input_chunk_length, "\tstride = forecast_horizon:", forecast_horizon, "\toutput_chunk_length:", output_chunk_length, "\tmetric:", round(metric_result,5), "\tstart:", start, "\t", series.time_index.min(), " ..", series.time_index.max(), "\tlength: ", len(series.time_index), "\n")
        if verbose: print(model_name, "input_chunk_length: ", input_chunk_length, "\toutput_chunk_length: ", output_chunk_length, "\tmetric: ", metric_result)

    t_elapsed = round(perf_counter()-t_start,3) 
    if verbose: print(str(t_elapsed) + " sec to train,fit,predict TFM")   

    enable_optimization=model.supports_optimized_historical_forecasts
    return model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed



def reset_matplotlib():
    # Reset the colors to default for v2.0 (category10 color palette used by Vega and d3 originally developed at Tableau)
    import matplotlib as mpl
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])



def plt_model_trained(ts=None, past_covariates=None, future_covariates=None, pred_input=None, pred=None, title=""):
    # Plots a trained model .predict() resuls with pred_input and pred plotted over the original series ts.
    # If ts is multivariate, plots each component (column) in a separate axis.
    # If past or future covariates are passed, they are plotted on an individual axis. 
    # Use plt_model_training for plotting a model being trained.  

    import matplotlib.pyplot as plt
    from darts import TimeSeries

    # Validate any series passed
    if not ts is None and  not isinstance(ts, TimeSeries): raise Exception("Value Error: ts is not a Darts TimeSeries")
    if not past_covariates is None and not isinstance(past_covariates, TimeSeries): raise Exception("Value Error: past_covariates is not a Darts TimeSeries")
    if not future_covariates is None and not isinstance(future_covariates, TimeSeries): raise Exception("Value Error: future_covariates is not a Darts TimeSeries")
    if not pred_input is None and not isinstance(pred_input, TimeSeries): raise Exception("Value Error: pred_input is not a Darts TimeSeries")
    if not pred is None and not isinstance(pred, TimeSeries): raise Exception("Value Error: pred is not a Darts TimeSeries")
    if not isinstance(title, str): raise Exception("Value Error: title must be a string")

    reset_matplotlib()

    if ts.width == 1:
        
        num_plots = ts.width
        if not past_covariates is None: num_plots += 1
        if not future_covariates is None: num_plots += 1
        i = 0

        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True)
        fig.suptitle(title)
        
        if not ts is None: ts.plot(ax=ax[i], label="ts", linewidth=3.0, linestyle='--', color='tab:gray')        
        if not pred_input is None: pred_input.plot(ax=ax[i], label="pred_input", color='C1')       
        ax[i].xaxis.set_label_text('')
        if not pred is None: pred.plot(ax=ax[i], label="pred", color='C3')
        i += 1

        if not past_covariates is None:
            ax[i].xaxis.set_label_text('')
            past_covariates.plot(ax=ax[i], label="past_covariates", color='C4')
            ax[i].xaxis.set_label_text('')
            i += 1
        
        if not future_covariates is None: 
            future_covariates.plot(ax=ax[i], label="future_covariates", color='C4')
            ax[i].xaxis.set_label_text('')
            i += 1

    else:
        # ts.width > 1 (multiple columns/components)
        num_plots = ts.width
        if not past_covariates is None: num_plots += 1
        if not future_covariates is None: num_plots += 1

        fig, ax = plt.subplots(num_plots, figsize=(14,6), sharex=True, sharey=True, layout='constrained')
        fig.suptitle(title)
        
        # Create individual series for plotting from each multivariate series
        df = ts.pd_dataframe()
        i = 0
        for col in df.columns:
            ds = df[col]
            ds.plot(ax=ax[i], linewidth=3.0, linestyle='--', color='tab:gray', label='ground truth')
            ax[i].title.set_text('ds.name')
            ax[i].xaxis.set_label_text('')
            i += 1
        
        if not pred_input is None:
            df = pred_input.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="pred_input", color='C1')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1

        if not pred is None:
            df = pred.pd_dataframe()
            i = 0
            for col in df.columns:
                ds = df[col]
                ds.plot(ax=ax[i], label="pred", color='C3')
                ax[i].title.set_text(ds.name)
                ax[i].xaxis.set_label_text('')
                i += 1

        i = ts.width
        if not past_covariates is None:
            past_covariates.plot(ax=ax[i])
            ax[i].title.set_text("past_covariates")
            ax[i].xaxis.set_label_text('')
            ax[i].legend_ = None
            i += 1

        if not future_covariates is None:
            future_covariates.plot(ax=ax[i])
            ax[i].title.set_text("future_covariates")
            ax[i].xaxis.set_label_text('')
            ax[i].legend_ = None
            i += 1

        fig.legend(loc='outside upper right', fontsize='small',labels=['ground_truth','pred_input','pred'])
    
    plt.tight_layout()
    plt.show()



def sine_gaussian_noise_covariate(length=400, add_trend=False, include_covariate=False, multivariate=False, start=None):
    # Generates Pandas dataframe of a noisy sine wave with optional trend and optional additional covariate column.
    # The noisy sine wave will have a seasonal period = 10.  
    # The noisy signal will have an amplitude generally greater than +/- 1.0
    # All data of numpy float32 for model effiency. 
    # The index is a datetime index that defaults to starting at 2000-01-01.  Assign alternative value as start=pd.Timestamp(year=2001, month=12, day=27)
    # If multivariate==True, then 3x different signals are returned, and one will be the noise sine wave.
    # If add_trend==True, then a trend will be applied to the univariate/multivariate signals, but not the covariate.  

    # Series created by taking the some of a sine wave and a gaussian noise series.
    # The intensity of the gaussian noise is also modulated by a sine wave (with 
    # a different frequency). 
    # The effect of the noise gets stronger and weaker in an oscillating fashion.
    # Derived from:  https://unit8co.github.io/darts/examples/08-DeepAR-examples.html#Variable-noise-series
    
    from darts import TimeSeries
    import darts.utils.timeseries_generation as tg
    from darts.utils.missing_values import fill_missing_values
    import pandas as pd
    import numpy as np

    if start is None: start = pd.Timestamp("2000-01-01")
    if not isinstance(start, pd.Timestamp): raise Exception("Value Error: start is not a valid timestamp.")

    trend = tg.linear_timeseries(length=length, end_value=4, start=start)    # end_value=4
    season1 = tg.sine_timeseries(length=length, value_frequency=0.05, value_amplitude=1.0, start=start)  # value_amplitude=1.0
    noise = tg.gaussian_timeseries(length=length, std=0.6, start=start)  
    noise_modulator = (
        tg.sine_timeseries(length=length, value_frequency=0.02, start=start)
        + tg.constant_timeseries(length=length, value=1, start=start)
    ) / 2
    covariates = noise_modulator.astype(np.float32)      # convert to numpy 32 bit float for model efficiency
    noise = noise * noise_modulator
    if add_trend:
        ts = sum([noise, season1, trend])
    else:
        ts = sum([noise, season1])
    ts = ts.astype(np.float32)      # convert to numpy 32 bit float for model efficiency

    if multivariate == True:
        ts1 = ts.copy()
        del ts
        
        # Create a second TimeSeries from df that is smooth and with different scale and a mean offset
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
        df = ts1.pd_dataframe()
        ds = df.rolling(window=indexer, min_periods=4).sum()
        ts2 = TimeSeries.from_series(pd_series=ds) 
        ts2 = ts2.rescale_with_value(0.5)
        ts2 = ts2 + 1.0
        ts2 = fill_missing_values(ts2)
        del indexer, ds, df

        # Create a third TimeSeries from ts1 using the .map() TimeSeries method
        ts3 = ts1.map(lambda x: x ** 2) 
        ts3 = ts3 - 1.1        # apply a mean offset

        # Create a multivariate TimeSeries from the three univariate time series.
        ts = ts1.stack(ts2)
        ts = ts.stack(ts3)
        del ts1, ts2, ts3

        # rename the columns
        ts = ts.with_columns_renamed(ts.columns, ['ts1','ts2','ts3'])


    df = ts.pd_dataframe()

    if include_covariate:
        ds = covariates.pd_series()
        df = df.assign(new_col_ds=ds)
        if ts.width == 1:
            df.rename(columns={df.columns[0]: 'noisy_sin', df.columns[1]: 'covariate'}, inplace=True, errors="raise")
        else:
            df.rename(columns={df.columns[3]: 'covariate'}, inplace=True, errors="raise")
    else:
        if ts.width == 1: df.rename(columns={df.columns[0]: 'noisy_sin'}, inplace=True, errors="raise")
    df.index.rename("Date", inplace=True)

    return df




if __name__ == '__main__':
    pass

    # ---------------------------------------------------------------------------
    # Demonstrate how to use the model .historical_forecasts() method AND 
    # how to save a model, load a model, train the model if necessary, and then
    # execute a forecast with the model.


    def tfm_save_load_ex():

        from darts.metrics import rmse, rmsle
        from darts.dataprocessing.transformers.scaler import Scaler
        from darts.dataprocessing.transformers import StaticCovariatesTransformer
        from darts import TimeSeries, concatenate
        from darts.utils.statistics import check_seasonality
        import torch

        from time import perf_counter
        from pathlib import Path
        from operator import itemgetter
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)
        
        # for reproducibility
        # https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        np.random.seed(1)

        scaler = Scaler()  # default uses sklearn's MinMaxScaler. 
        transformer = StaticCovariatesTransformer()

        min_max_scale = True

        def get_ts_n_covariates(df=None):

            # Get three Darts univariate TimeSeries from the first 3 columns in df & built a multivariate TimeSeries
            ts1 = TimeSeries.from_series(pd_series=df.iloc[:,0]) 
            ts2 = TimeSeries.from_series(pd_series=df.iloc[:,1]) 
            ts3 = TimeSeries.from_series(pd_series=df.iloc[:,2]) 
            ts = ts1.stack(ts2)
            ts = ts.stack(ts3)
            del ts1, ts2, ts3
            # ts is now a multivariate TimeSeries with 3x components (columns)
            print("\nts: ", ts.time_index.min(), " ..", ts.time_index.max(), "\tlength: ", len(ts.time_index), '\n')  

            # Define static covariates
            # Note: Add any static covariates to series 'ts' BEFORE splitting 'ts' into train, val, test, pred_input..
            static_covariates = None
            static_covariates = pd.DataFrame(data={"cont": [0, 2, 1], "cat": ["a", "c", "b"]})
            if not static_covariates is None:
                # Add the static covariates to the TimeSeries
                ts = ts.with_static_covariates(static_covariates)
                # Encode/transform categorical static covariates to match what was done by get_darts_model_splits() during model training.
                # [[0.0 'a'],[2.0 'c'],[1.0 'b']]   =>  [[0.  0. ],[1.  2. ],[0.5 1. ]]
                ts = transformer.fit_transform(ts)
                print(ts.static_covariates)

            # Define any covariates
            val = None
            past_covariates = None
            future_covariates = None
            val_past_covariates = None
            val_future_covariates = None
            # Create the raw full length series for the covariates from the 4th column of df, one for each in the multivariate series 
            covariates = TimeSeries.from_series(pd_series=df.iloc[:,3]) 
            past_covariates = covariates.stack(covariates) 
            past_covariates = past_covariates.stack(covariates) 
            future_covariates = covariates.stack(covariates) 
            future_covariates = future_covariates.stack(covariates) 
            del covariates
            if not val is None:
                if not past_covariates is None: val_past_covariates = past_covariates.copy()
                if not future_covariates is None: val_future_covariates = future_covariates.copy()

            return ts, val, past_covariates, val_past_covariates, future_covariates, val_future_covariates

        # Define a path to save the data file and the model with the best arguments input_chunk_length & output_chunk_length. 
        # Revise the path below for your system.  Below saves to folder 'data' in the parent of the current working folder.
        # Later in the script, the model filename will be defined using model_save_path with .joinpath() just prior to saving the trained model. 
        model_save_path = Path(Path.cwd().parent).joinpath('data/')
        if not model_save_path.exists(): raise Exception("Folder specified by 'path_file' doesn't exist: "+ str(model_save_path))

        # Generate a data set with 3x signals (multivariate) & a covariate (sine wave).
        df = sine_gaussian_noise_covariate(length=800, add_trend=False, include_covariate=True, multivariate=True)
        #df.reset_index(inplace=True, drop=True)     # Convert the index from datetime to integer (for testing only)
        
        # Save the generated signal to a Parquet file
        path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
        df.to_parquet(path=path_file, compression='gzip')
        del df

        # Read the Parquet file to dataframe and assign the first 400 rows to df
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        df = pd.read_parquet(path=path_file) 
        df = df.iloc[0:400, 0:]

        # Get the target series and covariates from df
        ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)

        from darts.models import (TFTModel,DLinearModel,NLinearModel,TiDEModel,TCNModel)

        # darts_tfm_historical_forecasts() does not optimize input_chunk_length and output_chunk_length.  
        # Normally you want to do that, but for this example I pass optimized values.
        model, train_length, start, enable_optimization, series, pred, scaler, metric_result, t_elapsed = darts_tfm_historical_forecasts(model_name='DLinearModel', model_class=DLinearModel, input_chunk_length=20, output_chunk_length=20, sum_chunk_length_limit=len(ts), ts=ts, past_covariates=past_covariates, future_covariates=future_covariates, forecast_horizon=40, stride=40, metric=rmse, min_max_scale=min_max_scale, retrain=True, verbose=True)

        if not model_save_path is None:
            # Save the model in order to retain the optimized model arguments.
            # Note: the training series is not saved when the .historical_forecast() method is used.
            # https://unit8co.github.io/darts/userguide/forecasting_overview.html#saving-and-loading-models
            # For TFM, two files are saved:  
            #   the model parameters with the specified extension .pt
            #   the training state with the method added extension of .ckpt
            # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
            # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
            path_file = model_save_path.joinpath("DLinearModel.pt")
            print("Saving model DLinearModel to: " + str(path_file))
            model.save(str(path_file))


        # Run a prediction using the best saved model

        # Read the saved model
        path_file = model_save_path.joinpath("DLinearModel.pt")
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        model = DLinearModel.load(str(path_file))
        print("Model arguments:")
        print("\tinput_chunk_length:", model.input_chunk_length)
        print("\toutput_chunk_length:", model.output_chunk_length)

        if model.training_series is None:
            # Train the model on df

            # Read the Parquet file to dataframe and assign the first 400 rows to df
            path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
            if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
            df = pd.read_parquet(path=path_file) 
            df = df.iloc[0:400, 0:]

            ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)
            
            # Trim the target series (for training & prediction) to the maximum length (limited by past/future coveriates & input_chunk_length/output_chunk_length)
            if past_covariates is None and future_covariates is None:
                series = ts
            else:
                # Series ts must be smaller than future_covariates & past covariates.  
                #print("Min series length for past & future covariates: ", len(ts)-input_chunk_length-output_chunk_length+1)
                series = ts[model.input_chunk_length+1:len(ts)-model.output_chunk_length]

            if min_max_scale == True: 
                series = scaler.fit_transform(series)
                if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
                if not val_past_covariates is None: val_past_covariates = scaler.transform(val_past_covariates)
                if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
                if not val_future_covariates is None: val_future_covariates = scaler.transform(val_future_covariates)

            model.fit(series=series, past_covariates=past_covariates, future_covariates=future_covariates, val_series=val_series, val_past_covariates=val_past_covariates, val_future_covariates=val_future_covariates)

            # save the trained model
            # For TFM, two files are saved:  
            #   the model parameters with the specified extension .pt
            #   the training state with the method added extension of .ckpt
            # WARNING: Do not include “last-” or “best-” in the beginning of the filename.
            # Darts model.save() method doesn't accept a pathlib object, only a string for the path/filename.
            path_file = model_save_path.joinpath("DLinearModel_trained.pt")
            print("Saving trained model DLinearModel to: " + str(path_file))
            model.save(str(path_file))

        print("\nmodel.training_series: ", model.training_series.time_index.min(), " ..", model.training_series.time_index.max(), "\tlength: ", len(model.training_series.time_index))  

        # Read the Parquet file to dataframe and assign the last 400 rows to df
        path_file = model_save_path.joinpath('sine_gaussian_noise_multivariate_covariate.parquet')
        if not path_file.exists(): raise Exception("File doesn't exist: ", path_file)
        df = pd.read_parquet(path=path_file) 
        df = df.iloc[400:, 0:]

        # Get the target series and covariates from df
        ts, val_series, past_covariates, val_past_covariates, future_covariates, val_future_covariates = get_ts_n_covariates(df)

        if min_max_scale == True: 
            ts = scaler.fit_transform(ts)
            if not past_covariates is None: past_covariates = scaler.transform(past_covariates)
            if not val_past_covariates is None: val_past_covariates = scaler.transform(val_past_covariates)
            if not future_covariates is None: future_covariates = scaler.transform(future_covariates)
            if not val_future_covariates is None: val_future_covariates = scaler.transform(val_future_covariates)

        # Create a series pred_input from ts for input to .predict(series=pred_input) with a length > input_chunk_length
        # start at row 50 and end with a length of 100 rows
        pred_input = ts[50:50+100]            
        # How to slice by specific date values                            
        #pred_input = ts.slice(pd.Timestamp(year=2006, month=6, day=1), pd.Timestamp(year=2006, month=9, day=8))
        print("\npred_input: ", pred_input.time_index.min(), " ..", pred_input.time_index.max(), "\tlength: ", len(pred_input.time_index))  

        # Define the the number of steps (series rows) or n beyond the end of pred_input for the prediction.
        pred_steps = 150

        pred = model.predict(n=pred_steps, series=pred_input, past_covariates=past_covariates, future_covariates=future_covariates)  # defaults to series=train when pred_input==None
        print("\npred: ", pred.time_index.min(), " ..", pred.time_index.max(), "\tlength: ", len(pred.time_index))  

        # Calculate the root mean square error (RMSE)
        metric_rmse = rmse(ts.slice_intersect(pred), pred)
        print("DLinearModel RMSE: ", round(metric_rmse,5))   
        
        # Plot
        title = "DLinearModel RMSE: " + str(round(metric_rmse,5))
        plt_model_trained(ts, past_covariates, future_covariates, pred_input, pred, title)


    tfm_save_load_ex()

    # ---------------------------------------------------------------------------
