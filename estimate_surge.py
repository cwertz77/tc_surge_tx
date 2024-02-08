import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics as m

from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import os


def plot(storm):
    fig = px.density_mapbox(storm, lat='LAT', lon='LON',z='USA_WIND')
    fig.update_layout(mapbox_style="carto-positron")
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
    dist = nodes - 150
    critical_pt = np.argmin(abs(dist))
    training_data = training_data.loc[critical_pt]
    training_data=np.array([training_data['LAT'],training_data['LON'],training_data['USA_WIND'],training_data['USA_PRES'],
                            training_data['STORM_SPEED'],training_data['STORM_DIR'],
                            training_data['LANDFALL']])

    return training_data
def process_flood_data(basin,hurricane_id):
    flood=pd.read_csv(f'data/flood/{hurricane_id}_{basin}.csv')
    #flood.to_csv(f'data/flood/{hurricane_id}_{basin}.csv')
    # replaces any '99' values with 0: no flood here.
    flood['Depth_max'] = flood['Depth_max'].replace(to_replace=99.90000, value=0)
    cords = [flood[' Lat'], -1*flood[' Lon']]
    flood = flood['Depth_max']
    flood=np.array(np.transpose(flood))
    return flood, cords

def plot_flood(cords, y,title):
    fig = px.scatter_mapbox(lat=cords[0],lon=cords[1],color=y,color_continuous_scale='ice_r',
                            title=title,labels={'color':'Surge Height (ft)'},zoom=6.5,range_color=[0,22])
    fig.update_layout(mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(coloraxis_colorbar_ticks="outside",
                      coloraxis_colorbar_title_side="right",
                      coloraxis_colorbar_tickvals=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      coloraxis_colorbar_tickfont_size=24,
                      coloraxis_colorbar_title_font_size=24)
    fig.write_image(f'{title}.png')
    # fig.show()

hurricane_ids = ['2003188N11307','2004247N10332','2005192N11318','2005236N23285',
                 '2005261N21290','2007255N27265','2008203N18276','2008238N13293',
                 '2008245N17323',
                 '2017228N14314','2019192N29274','2020205N26272','2020233N14313',
                 '2020279N16284','2021256N21265']

X = np.zeros((len(hurricane_ids),7))
Y = np.zeros((len(hurricane_ids),15341))
for k in range(len(hurricane_ids)):
    print('processing hurricane #',hurricane_ids[k])
    hurricane = process_hurricane_data(hurricane_ids[k])
    X[k,:] = hurricane
    flood, key_cords = process_flood_data('Corpus',hurricane_ids[k])
    Y[k,:] = flood

min_max_scaler = preprocessing.MinMaxScaler()
hurricane_list = min_max_scaler.fit_transform(X)
print('done with data processing.')
print('learning model')
n_inputs, n_outputs = X.shape[1], Y.shape[1]
train_X=hurricane_list[3:15];test_X=hurricane_list[0:2];train_Y=Y[3:15];test_Y=Y[0:2]

# get the model for random forest
rf = RandomForestRegressor(random_state=42)
rf.fit(train_X, train_Y)
feature_names = ['Latitude', 'Longitude', 'Wind Speed', 'Pressure','Storm Speed','Storm Direction', 'Landfall']
rf_features=pd.DataFrame(data=hurricane_list,columns=feature_names)
# vif_data = pd.DataFrame()
# # vif_data["feature"] = rf_features.columns
# # vif_data["VIF"] = [variance_inflation_factor(rf_features.values, i)
# #                           for i in range(len(rf_features.columns))]
# # print(vif_data)

# correlation_matrix=rf_features.corr()
# plt.figure(figsize=(11,11))
# sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',linewidths=0.5)
# plt.savefig('rf_correlation_matrix.png')

# plt.barh(feature_names,rf.feature_importances_)
# plt.title("Feature Importance for Random Forest Regressor")
# plt.savefig('rf_feature_importance.png')
predictions = rf.predict(test_X)
r2=m.r2_score(test_Y, predictions)
print("R2 score: ", m.r2_score(test_Y, predictions))
print("Root Mean squared error: ", m.mean_squared_error(test_Y, predictions,squared=False))
print("Mean absolute error: ", m.mean_absolute_error(test_Y, predictions))
plot_flood(key_cords,test_Y[1], "SLOSH_Surge_Prediction")
plot_flood(key_cords, predictions[1], "Random_Forest_Surge_Prediction")
plot_flood(key_cords, test_Y[1]-predictions[1], "SLOSH-RF")

# model for CNN
cnn = Sequential()
cnn.add(Conv1D(24,2,activation='relu',input_shape=(7,1)))
cnn.add(Flatten())
cnn.add(Dense(24,activation='relu'))
cnn.add(Dense(15341,activation='relu'))
cnn.compile(loss='mse',optimizer='adam')
cnn.summary()
cnn.fit(train_X,train_Y,batch_size=12,epochs=1000,verbose=0)
predictions=cnn.predict(test_X)
print("R2 score: ", m.r2_score(test_Y, predictions))
print("Root Mean squared error: ", m.mean_squared_error(test_Y, predictions,squared=False))
print("Mean absolute error: ", m.mean_absolute_error(test_Y, predictions))
plot_flood(key_cords,predictions[1],"Convolutional_Neural_Network_Surge_Prediction")
plot_flood(key_cords, test_Y[1]-predictions[1], "SLOSH-CNN")

