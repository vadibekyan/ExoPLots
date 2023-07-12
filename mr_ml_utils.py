#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def score_dataset(X, y, model):
    score = cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error',
        )
    mae = (-1*score)
    return mae.mean()


def optuna_RF_Reg(X_train, y_train, n_trials, display = False):
    import optuna
    from optuna.visualization import plot_contour
    from optuna.visualization import plot_param_importances

    def objective(trial):

        RF_reg_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 20, 200, step = 20),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 6),
            #min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 6),
            max_features = trial.suggest_int("max_features", 3, len(X_train.columns)),
            random_state = 0
        )

        RF_reg = RandomForestRegressor(**RF_reg_params).fit(X_train, y_train)
        score = score_dataset(X_train, y_train, RF_reg)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    RFR_best_params = study.best_params
    
    if display == True:
        plot_param_importances(study).show()
        plot_contour(study).show()


    print (f'\nThe best hyperparameters for RF Regressor {RFR_best_params}')
    #y_train_predict = RandomForestRegressor(**RFR_best_params).fit(X_train, y_train).predict(X_train)
    #print (f'\nAME of the BEST RF Regressor: Training data {(mean_absolute_error(y_train, y_train_predict))}')

    return RFR_best_params


def modified_learning_curve(X_train, y_train, X_test, y_test, model, N, CV = 5, display = False, figure = True):

    train_sizes =np.linspace(0.05, 0.99, N)
    print ()

    results = pd.DataFrame(columns = ['train_size', 'mae_train_mean', 'mae_train_std', 'mae_test_mean', 'mae_test_std'], index = range(0, N))


    for i, train_size in enumerate(train_sizes):
        mae_train = []
        mae_test = []

        for j in range(CV):

                # Random state is not specified as cross-validation is applied.
                X_train_tmp, _, y_train_tmp, _ = train_test_split(X_train, y_train, test_size=(1-train_size))

                model.fit(X_train_tmp, y_train_tmp)

                # Make predictions on the test set
                y_pred_test = model.predict(X_test)
                y_pred_train_tmp = model.predict(X_train_tmp)

                # Evaluate the model using mean absolute error
                mae_train.append(mean_absolute_error(y_train_tmp, y_pred_train_tmp))
                mae_test.append(mean_absolute_error(y_test, y_pred_test))

        
        results['train_size'][i] = len(X_train_tmp)
        results['mae_train_mean'][i] = np.mean(mae_train)
        results['mae_train_std'][i] = np.std(mae_train)
        results['mae_test_mean'][i] = np.mean(mae_test)
        results['mae_test_std'][i] = np.std(mae_test)

        if display:
            print (f'For a training size of {len(X_train_tmp)}, MAE(train) = {np.mean(mae_train):.3f} and MAE(test) = {np.mean(mae_test):.3f}')

    # Iterate over each column in the DataFrame
    for column in results.columns:
        # Check if the column has object data type
        if results[column].dtype == 'object':
            # Convert the column to numeric type
            results[column] = pd.to_numeric(results[column], errors='coerce')

    if figure == True:
        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")

        plt.plot(results['train_size'], results['mae_train_mean'], 'o-', color="r",
                label="Training Score")

        plt.fill_between(results['train_size'], results['mae_train_mean'] + results['mae_train_std'], 
                            results['mae_train_mean'] - results['mae_train_std'], 
                            alpha=0.15, color='r')

        plt.plot(results['train_size'], results['mae_test_mean'], 'o-', color="g",
                label="Cross-Validation Score")

        plt.fill_between(results['train_size'], results['mae_test_mean'] + results['mae_test_std'], 
                            results['mae_test_mean'] - results['mae_test_std'], 
                            alpha=0.15, color='g')

        plt.legend(loc="best")
        plt.show()
    
    return results


def RF_training_with_display(X,y, test_size = 0.15, n_trials = 20):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    RFR_base_model = RandomForestRegressor(random_state =0)
    MAE_base = score_dataset(X_train, y_train, RFR_base_model)

    print (f'The base MAE score from the trainin set is {MAE_base:.0f}')

    RFR_best_params = optuna_RF_Reg(X_train, y_train, n_trials, display = False)

    best_model = RandomForestRegressor(**RFR_best_params)

    final_trained_model_full_data = RandomForestRegressor(**RFR_best_params).fit(X, y)
    final_trained_model_train_data = RandomForestRegressor(**RFR_best_params).fit(X_train, y_train)

    y_train_predict = final_trained_model_train_data.predict(X_train)
    y_test_predict = final_trained_model_train_data.predict(X_test)
    print (f'\nAME of the BEST RF Regressor: Training data {(mean_absolute_error(y_train, y_train_predict))}')
    print (f'\nAME of the BEST RF Regressor: Test data {(mean_absolute_error(y_test, y_test_predict))}')

    results = modified_learning_curve(X_train, y_train, X_test, y_test, final_trained_model_train_data, 5)

    return best_model, final_trained_model_train_data, final_trained_model_full_data
