import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import preprocessing
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
# from global_land_mask import globe
# from keras.models import Sequential
# from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def plot(storm, flood):
    fig = px.scatter_mapbox(storm, lat='LAT', lon='LON')
    fig.add_densitymapbox(lat=flood[' Lat'], lon=-1 * flood[' Lon'])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
def process_hurricane_data(hurricane_id):
    # Read in data
    hurricanes = pd.read_csv('data/hurricane/ibtracs.NA.list.v04r00.csv')
    list_of_names = hurricanes['NAME'].to_list()
    names = []
    [names.append(x) for x in list_of_names if x not in names]
    hurricanes = hurricanes.groupby(['SID'])
    training_data = hurricanes.get_group(hurricane_id)
    training_data.reset_index(inplace=True)
    # filter points not in texas basin
    training_data.drop(np.where(training_data['LAT'] < 25)[0],inplace=True)
    training_data.reset_index(inplace=True)

    # collect points 300km from land
    nodes = training_data['DIST2LAND'].values
    dist = nodes - 300
    critical_pt = np.argmin(abs(dist))
    training_data = training_data.loc[critical_pt]
    training_data=np.array([training_data['LAT'],training_data['LON'],training_data['USA_WIND'],training_data['USA_PRES'],
                            training_data['STORM_SPEED'],training_data['STORM_DIR'],
                            training_data['LANDFALL']])

    return training_data
def process_flood_data(basin,hurricane_id):
    flood=pd.read_csv(f'data/flood/{hurricane_id}_{basin}.csv')
    flood.to_csv(f'data/flood/{hurricane_id}_{basin}.csv')
    # replaces any '99' values with 0: no flood here.
    flood['Depth_max'] = flood['Depth_max'].replace(to_replace=99.90000, value=0)
    cords = [flood[' Lat'], -1*flood[' Lon']]
    flood = flood['Depth_max']
    flood=np.array(np.transpose(flood))
    return flood, cords

def plot_flood(cords, y):
    fig = px.scatter_mapbox(lat=cords[0],lon=cords[1],color=y)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

def plot_accuracy(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss.png')


hurricane_ids = ['2003188N11307','2004247N10332','2005192N11318','2005236N23285',
                 '2005261N21290','2007255N27265','2008203N18276','2008238N13293',
                 '2008245N17323',
                 '2017228N14314','2019192N29274','2020205N26272','2020233N14313',
                 '2020279N16284','2021256N21265']

hurricane_list = np.zeros((len(hurricane_ids),7))
flood_list = np.zeros((len(hurricane_ids),15341))
for k in range(len(hurricane_ids)):
    print('processing hurricane #',hurricane_ids[k])
    hurricane = process_hurricane_data(hurricane_ids[k])
    hurricane_list[k,:] = hurricane
    flood, key_cords = process_flood_data('Corpus',hurricane_ids[k])
    flood_list[k,:] = flood


min_max_scaler = preprocessing.MinMaxScaler()
hurricane_list = min_max_scaler.fit_transform(hurricane_list)
print('done with data processing.')
print('learning model')
n_inputs, n_outputs = hurricane_list.shape[1], flood_list.shape[1]
train_X=hurricane_list[3:15];test_X=hurricane_list[0:2];train_Y=flood_list[3:15];test_Y=flood_list[0:2]

# get the model for neural networks
regr = MLPRegressor(random_state=1, max_iter=500).fit(train_X, train_Y)
predictions = regr.predict(test_X)
score = regr.score(test_X,test_Y)
plot_flood(key_cords,predictions[1])
plot_flood(key_cords,test_Y[1])
# plot_accuracy(history)

# get the model for random forest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_X, train_Y)
plt.barh(['lat', 'lon', 'wind speed', 'pressure','storm speed','storm direction', 'landfall'],rf.feature_importances_)
predictions = rf.predict(test_X)
score1 = r2_score(test_Y,predictions)
plot_flood(key_cords, predictions[1])
print('done')