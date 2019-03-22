def df_to_npmx (df):
    """
    convert pandas dataframe to numpy matrix as input for neural networks

    Arguments:
    df -- a list of dataframes or one dataframe

    Returns:
    df -- a list of numpy matrixes or one numpy matrix
    """
    if type(df) == list:
        df_list = []
        print("pass in a list of pandas dataframes")
        print("will return a list of {} numpy matrixes".format(len(df)))
        for df_i in df:
            npmx_i = df_i.to_numpy()
            df_list.append(npmx_i)
        return df_list
    else:
        print("pass in one pandas dataframes\nwill return one converted dataframes")
        npmx = df.to_numpy()
        return npmx

def plot_loss(history):
    """
    plot model loss and model error in keras model

    Arguments:
    history -- Callback that records events into a History objectself.
    This callback is automatically applied to every Keras model.
    The History object gets returned by the fit method of models.

    Returns:
    nothing -- this function does not return stuff.
    """
    import matplotlib.pyplot as plt

    print('train set loss: {:.4f}'.format(history.history['loss'][-1]))
    print('dev set loss: {:.4f}'.format(history.history['val_loss'][-1]))
    print('train set error: {:.4f}'.format(history.history['mean_squared_logarithmic_error'][-1]))
    print('dev set error: {:.4f}'.format(history.history['val_mean_squared_logarithmic_error'][-1]))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['mean_squared_logarithmic_error'])
    plt.plot(history.history['val_mean_squared_logarithmic_error'])
    plt.title('Model Loss & Error')
    plt.ylabel('Loss & Error')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Dev Loss', 'Train Error', 'Dev Error'], loc='upper right')
    plt.show()

def model(auto, train_X, train_Y, dev_X, dev_Y, l2=0.01, lr=0.0001, batch_size=32, epochs=100):

    import numpy as np
    import pandas as pd
    import keras
    from keras.models import Model
    from keras import layers
    from keras import losses
    from keras import optimizers
    from keras import initializers
    from keras import regularizers
    from keras_pandas.Automater import Automater

    X = auto.input_nub
    X = layers.Dense(64, activation='relu',
                     kernel_initializer=initializers.he_normal(seed=42),
                     bias_initializer=initializers.Zeros(),
                     kernel_regularizer=regularizers.l2(l2),
                     bias_regularizer=regularizers.l2(l2))(X)
    X = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     bias_regularizer=regularizers.l2(l2))(X)
    X = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01),
                     bias_regularizer=regularizers.l2(0.01))(X)
    X = layers.Dense(1, activation='linear')(X)
    X = auto.output_nub(X)

    model = Model(inputs=auto.input_layers, outputs=X)
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.mean_squared_logarithmic_error,
                  metrics = [losses.mean_squared_logarithmic_error])

    history = keras.callbacks.History()
    model.fit(train_X, train_Y, validation_data=[dev_X, dev_Y], shuffle=True, verbose=2,
              batch_size=batch_size, epochs=epochs, callbacks=[history])
    return model, history

def submit(auto, model, raw_test, test_X):
    import numpy as np
    import pandas as pd
    pred_y_standardized = model.predict(test_X)
    pred_y = auto.inverse_transform_output(pred_y_standardized)

    submit = pd.DataFrame(data=[[i for i in raw_test.index], [i[0] for i in pred_y]]).T
    submit.columns = ['Id', 'SalePrice']
    submit = submit.astype('int32')
    submit.to_csv('submit.csv', index=False)

def automater ():
    from keras_pandas.Automater import Automater

    numerical = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                 'MiscVal', 'MoSold', 'YrSold',
                 'SalePrice']

    categorical = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                   'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                   'Neighborhood','Condition1', 'Condition2', 'BldgType',
                   'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                   'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                   'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                   'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                   'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                   'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                   'MiscFeature', 'SaleType', 'SaleCondition']

    data_type_dict = {'numerical': numerical, 'categorical': categorical}
    output_var = 'SalePrice'
    auto = Automater(data_type_dict=data_type_dict, output_var=output_var)
    return auto

def model_1(auto, train_X, train_Y, dev_X, dev_Y, l2=0.01, lr=0.0001, batch_size=32, epochs=100):
    import keras
    from keras.models import Model
    from keras import layers
    from keras import losses
    from keras import optimizers
    from keras import initializers
    from keras import regularizers
    from keras_pandas.Automater import Automater

    X = auto.input_nub
    X = layers.Dense(64, activation='relu',
                     kernel_initializer=initializers.he_normal(seed=42),
                     bias_initializer=initializers.Zeros(),
                     kernel_regularizer=regularizers.l2(l2),
                     bias_regularizer=regularizers.l2(l2))(X)
    X = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     bias_regularizer=regularizers.l2(l2))(X)
    X = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     bias_regularizer=regularizers.l2(l2))(X)
    X = layers.Dense(1, activation='linear')(X)
    X = auto.output_nub(X)

    model = Model(inputs=auto.input_layers, outputs=X)
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=losses.mean_squared_logarithmic_error,
                  metrics = [losses.mean_squared_logarithmic_error])

    history = keras.callbacks.History()
    model.fit(train_X, train_Y, validation_data=[dev_X, dev_Y], shuffle=True, verbose=2,
              batch_size=batch_size, epochs=epochs, callbacks=[history])
    return model

def model_2(auto, l1=0.01, l2=0.01, lr=0.0001, batch_size=32, epochs=100):
    import numpy as np
    import keras
    from keras.models import Model
    from keras import layers
    from keras import losses
    from keras import optimizers
    from keras import initializers
    from keras import regularizers
    from keras_pandas.Automater import Automater

    X = auto.input_nub
    X = layers.Dense(32, activation='relu',
                     kernel_initializer=initializers.he_normal(seed=42),
                     bias_initializer=initializers.Zeros())(X)
    X = layers.Dense(16, activation='relu')(X)
    X = auto.output_nub(X)

    model = Model(inputs=auto.input_layers, outputs=X)
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=losses.mean_squared_logarithmic_error,
                  metrics = [losses.mean_squared_logarithmic_error])

    return model

def gridsearchcv(model, param_grid, train_X, train_Y, dev_X, dev_Y):
    from sklearn.model_selection import (cross_val_score, GridSearchCV)
    from sklearn.metrics import make_scorer

    def rmsle(y, y0):
        return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))

    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    model = GridSearchCV(model, param_grid=param_grid, scoring=rmsle_scorer, cv=3)
    model.fit(train_X, train_Y)
    print('grid search best parameters:', model.best_params_)
    print('grid search best scores: {:.4f}'.format(model.best_score_))

    train_scores = cross_val_score(model, train_X, train_Y, scoring=rmsle_scorer, cv=3)
    train_score = train_scores.mean()
    print('cv score: {:.4f}'.format(train_score))

    pred_y = model.predict(dev_X)
    dev_score = rmsle(dev_Y, pred_y)
    print('dev score: {:.4f}'.format(dev_score))

    return model, train_score, dev_score

def columns():
    numerical = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                'MiscVal', 'MoSold', 'YrSold']
    categorical = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood','Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                'MiscFeature', 'SaleType', 'SaleCondition']

    return numerical, categorical

def features():
    numerical = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
                'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
                'MiscVal', 'MoSold', 'YrSold']
    categorical = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood','Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                'MiscFeature', 'SaleType', 'SaleCondition']

    return numerical, categorical
