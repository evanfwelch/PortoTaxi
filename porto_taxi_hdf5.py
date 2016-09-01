# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:20:37 2016

@author: Evan

I recently got interested in ridesharing apps like Lyft. I came across this 
Porto Taxi dataset from a Kaggle competition. Instead of using it to predict
 taxi trajectories, I'll play around with some visualizatin, clustering, and 
 light ridesharing simulation as a way to get familiar with this type
 of problem.
"""

import numpy as np 
import pandas as pd
from pandas import HDFStore
import scipy
from scipy.spatial.distance import cdist, pdist, euclidean
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from matplotlib import mlab as ML
import os
import zipfile
import math
import json
import itertools
import sklearn
from sklearn.cluster import KMeans
from collections import OrderedDict

DATA_PATH = '..\\in\\train.csv.zip'
HDFSTORE_PATH = '..\\data\\parsed_and_derived_data_LITEMEMORY.h5'
# just first result from Google, used for mercator projection
PORTO_LONLAT = [-8.6291, 41.1579]

R_EARTH = 6371.0
CHUNK_SIZE = 1e5

def haversine_distance(A,B):
    
    chg_lon = math.radians(B[0]-A[0])
    chg_lat = math.radians(B[1]-A[1])
    a = math.sin(chg_lat/2)**2 + math.cos(A[1]*math.pi/180) * math.cos(B[1]*math.pi/180) * math.sin(chg_lon/2)**2  
    c = 2 * math.asin(math.sqrt(a))  
    return R_EARTH * c
    
def mercator_projection(lonlat, origin = PORTO_LONLAT):    
        X = (1.0 if lonlat[0]>=origin[0] else -1.0)*haversine_distance([lonlat[0],origin[1]],origin)
        Y = (1.0 if lonlat[1]>=origin[1] else -1.0)*haversine_distance([origin[0],lonlat[1]],origin)
        return [X,Y]

def parse_and_hdfstore_data(DEBUG_MODE=False):
    """ Opens the raw data, reads and filters it, 
    then saves down relevant parts to HDF5 file"""
    # unzip
    zf = zipfile.ZipFile(DATA_PATH)
    usecols =['TRIP_ID','CALL_TYPE','TAXI_ID','TIMESTAMP',
    'DAY_TYPE', 'MISSING_DATA','POLYLINE']
    
    # define some converters to parse columns
    def convert_polyline(x): return json.loads(x)
    def convert_timestamp(x): return pd.datetime.fromtimestamp(float(x))
        
    # define what we will exclude
    def is_outlier(row):
        list_of_disqualifications = [
            row['trip_seconds'] > 2*3600,
            row['trip_km'] > 150,
            row['average_kmph'] > 200,
            row['trip_to_crow_ratio'] > 12        
            ]
        return any(list_of_disqualifications)
    
    # open HDFStore
    hd_store = HDFStore(HDFSTORE_PATH)
    
    # read in .csv in chunks
    chunkno = 0 
    
    with zf.open('train.csv') as f:
        iter_csv = pd.read_csv(f, 
                               converters={
                                   'POLYLINE':convert_polyline, 
                                   'TIMESTAMP':convert_timestamp},
                               chunksize=CHUNK_SIZE, 
                               iterator=True, 
                               usecols=usecols)
                               
        # call one chunk at a time
        for chunk in iter_csv:
             if not DEBUG_MODE or (DEBUG_MODE and chunkno < 3):
                 # load only the trips that were hailed "organically" from street level as these most resemble problem lyft faces
                 chunk = chunk[chunk.apply(lambda row: row['CALL_TYPE']=='C' and row['MISSING_DATA']!=True,axis=1)].drop(['CALL_TYPE','MISSING_DATA'],axis=1)
                 # generate some more features
                 chunk = derive_columns(chunk)
                 # filter outliers
                 chunk = chunk[chunk.apply(is_outlier,axis=1) != True]
                 # peel off POLYLINE
                 trips = np.vstack([[ii,jj,pt[0],pt[1]] for ii,poly in chunk[['TRIP_ID','POLYLINE']].values for jj,pt in enumerate(poly)])
                 df_trips = pd.DataFrame(data=trips, columns=['TRIP_ID','T','Lon','Lat'])
                 
                 # now drop
                 chunk = chunk.drop(['POLYLINE'],axis=1)
                 # store to HDF
                 hd_store.append('df_all',chunk,format='table',data_columns=True)
                 hd_store.append('df_trip',df_trips,format='table', data_columns=True)
                 chunkno += 1
                 print('have read in %s chunks of size %s; at %s rows' % (chunkno, CHUNK_SIZE, hd_store['df_all'].shape[0]))
             else: 
                 break
         # re-index the stored table
    print('Stored %s rows' % hd_store['df_all'].shape[0])
    
    # trim the 1.5 percentile
    alpha = 1.5
    x_bounds = np.percentile(hd_store['df_all']['destination_mercator_x'],[alpha,100.0-alpha])
    y_bounds = np.percentile(hd_store['df_all']['destination_mercator_y'],[alpha,100.0-alpha])
    hd_store.append('df',hd_store.select('df_all',[pd.Term('destination_mercator_x','>',[x_bounds[0]]), pd.Term('destination_mercator_x','<',[x_bounds[1]]),pd.Term('destination_mercator_y','>',[y_bounds[0]]),pd.Term('destination_mercator_y','<',[y_bounds[1]])]),format='table',data_columns=True)
    
    print('stored %s rows' % hd_store['df'].shape[0])
    return hd_store

def derive_columns(chunk):
    """ derive some features from the HDFStored path data and filter accordingly"""  
    # generate some trip dependent metrics
    chunk = chunk.dropna()

    chunk['trip_seconds'] = chunk.POLYLINE.apply(lambda x: (len(x)-1)*15)
    chunk = chunk[chunk.trip_seconds > 30]
    chunk['trip_km'] = chunk.POLYLINE.apply(lambda x: None if len(x) < 2 else sum([haversine_distance(x[ii],x[ii+1]) for ii in range(len(x)-1)]))
    chunk['crow_km'] = chunk.POLYLINE.apply(lambda x: haversine_distance(x[0],x[-1]))
    chunk['trip_to_crow_ratio'] = chunk.trip_km/chunk.crow_km
    #    chunk['origin'] = chunk.POLYLINE.apply(lambda x: x[0])
    chunk['origin_lon'] = chunk.POLYLINE.apply(lambda x: x[0][0])
    chunk['origin_lat'] = chunk.POLYLINE.apply(lambda x: x[0][1])
    chunk['destination_lon'] = chunk.POLYLINE.apply(lambda x: x[-1][0])
    chunk['destination_lat'] = chunk.POLYLINE.apply(lambda x: x[-1][1])
    
    chunk['average_kmph'] = chunk.apply(lambda row: 3600.0*row['trip_km']/row['trip_seconds'], axis=1)
    
    
    chunk['destination_mercator_x']= chunk.apply(lambda row: mercator_projection([row['destination_lon'],row['destination_lat']])[0], axis=1)
    chunk['destination_mercator_y']= chunk.apply(lambda row: mercator_projection([row['destination_lon'],row['destination_lat']])[1], axis=1)
    chunk['origin_mercator_x']= chunk.apply(lambda row: mercator_projection([row['origin_lon'],row['origin_lat']])[0], axis=1)
    chunk['origin_mercator_y']= chunk.apply(lambda row: mercator_projection([row['origin_lon'],row['origin_lat']])[1], axis=1)
    
    
    return chunk
    

def graph_all_trips(hd_store): 
    """# I adapted this method from kaggle contributor Fluxus who plots endpoints and trims outliers"""
    
    # aggregate all points visited
    #    lst_lonlats = np.array(list(itertools.chain(*[poly for poly in hd_store['df'].POLYLINE])))
    
    # cut off outlier points
    lon_bounds = np.percentile(hd_store['df_trip']['Lon'],[2,98])
    lat_bounds = np.percentile(hd_store['df_trip']['Lat'],[2,98])
    
    # create image
    bins = 513
    lat_bins = np.linspace(lat_bounds[0], lat_bounds[1], bins)
    lon_bins = np.linspace(lon_bounds[0], lon_bounds[1], bins)
    H2, _, _ = np.histogram2d(hd_store['df_trip']['Lat'],hd_store['df_trip']['Lon'], bins=(lat_bins, lon_bins))
    
    
    # flip first axis, shift by 1 and take logarithm
    img = np.log(H2[::-1, :] + 1)
    
    # generate figure
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title('Taxi paths')
    plt.axis('off')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    fig.savefig('..\\figures\\taxi_paths')
    return (fig,ax)
    
def destination_clustering(hd_store):
    """ Perform some k-means clustering on the mercator projected destinations to identify how much variance can be explained by various metropolitan clusters"""
    # prepare destinations for KMeans fitting
    #    euclidean_input = np.array([lonlat for lonlat in hd_store.df.destination_mercator])
    euclidean_input = hd_store['df'][['destination_mercator_x','destination_mercator_y']].as_matrix()
    # choose k such that additinal variance explained is relatlively small
    k_range = range(1,11)
    # fit model for each K and save list of centroids    
    lst_kmeans = [KMeans(n_clusters=k).fit(euclidean_input) for k in k_range]
    
    best_fit = [ sum([haversine_distance(euclidean_input[ii],km.cluster_centers_[km.labels_[ii]])**2 for ii in range(euclidean_input.shape[0])])/euclidean_input.shape[0] for km in lst_kmeans]
    
    f,ax = plt.subplots(1,1)
    plt.plot(k_range,best_fit,'bp-')
    ax.set_title('Mean Squared Distance to Nearest of K centroids')
    ax.set_xlabel('K')
    f.savefig('..\\figures\\MSE_kmeans')
    return (f, ax, lst_kmeans)
    
def voronoi_graph(km, hd_store):
    fig, ax = plt.subplots(1,1)
    
    #    x = hd_store.df['destination_mercator'].apply(lambda x: x[0])
    #    y = hd_store.df['destination_mercator'].apply(lambda x: x[1])
    
    plt.hexbin(hd_store.df['destination_mercator_x'],hd_store.df['destination_mercator_y'],bins='log', cmap=plt.cm.YlOrRd_r,gridsize=1000)
    
    vor = Voronoi(km.cluster_centers_)
    voronoi_plot_2d(vor,ax, show_vertices=False, line_colors='w')
    # number each cluster    
    for center in range(len(km.cluster_centers_)):
        cent = km.cluster_centers_[center]
        ax.annotate(str(center),xy=cent,color='white',fontweight='bold',fontsize=14)
    plt.show()
    fig.savefig('..\\figures\\voronoi')
    return fig, ax
    
def build_transition_matrix(km,hd_store):
    
    #    origin_in = np.array([coord for coord in hd_store.df['origin_mercator']])    
    #    dest_in = np.array([coord for coord in hd_store.df['destination_mercator']])    
    
    origin_predict = km.predict(hd_store.df[['origin_mercator_x','origin_mercator_y']])
    dest_predict = km.predict(hd_store.df[['destination_mercator_x','destination_mercator_y']])
    
    # now crosstabulate
    ctab = pd.crosstab(origin_predict,dest_predict)
    totals = ctab.sum(axis=0)
    
    for ii,col in enumerate(ctab.columns):
        ctab[col] = ctab[col]/totals[ii]
    
    # turn frequencies into rates
    
    
    fig, ax = plt.subplots(1,1)
    
    plt.pcolor(ctab,cmap=plt.cm.Oranges)
    plt.yticks(np.arange(0.5, len(ctab.index), 1), ctab.index)
    plt.xticks(np.arange(0.5, len(ctab.columns), 1), ctab.columns)
    plt.colorbar()
    plt.show()
    fig.savefig('..\\figures\\transition_matrix')
    
    return ctab, fig, ax

def make_ride_distribution_chart(hd_store):
    
    
    fig,ax = plt.subplots()
    dist = hd_store.df.groupby(['TAXI_ID']).trip_km.aggregate(np.count_nonzero).sort_values(ascending=0).cumsum()/hd_store.df.shape[0]
    gini = np.trapz(dist.values)*2/len(hd_store.df.TAXI_ID.unique())
    ax.plot(list(range(len(dist))),dist.values)
    ax.set_title('gini coefficient is %4.2f' % gini)
    plt.show()
    fig.savefig('..\\figures\\gini')
    return (fig,ax)

def make_seasonal_graphs(hd_store):
    
    # create subplots
    fig,ax = plt.subplots(4,5)
    lst_lambdas = [lambda x: 24 if x.hour==0 else x.hour, 
                   lambda x: x.dayofweek, lambda x: x.month, lambda x: x.day]
    
    # aggregators to apply for seasonality charts
    lst_aggregators = [('trip_seconds',np.sum),
                       ('trip_seconds',np.count_nonzero),
                       ('trip_seconds',np.median),
                        ('trip_km',np.median),
                        ('average_kmph',np.median)]
    
    # subplot titles
    titles = ['hourly','daily','monthly','intramonth']
    counter = ['total_trip_time','ride_count','median_trip_time',
    'median_trip_dist','median_median_speed']
    
    # generate charts
    for idx,ax_row in enumerate(ax):
        # group by the seasonal bucket
        grouped = hd_store['df'].groupby(hd_store['df'].TIMESTAMP.apply(lst_lambdas[idx]))
        #make each subplot
        for jj in range(5):
            agg = grouped[lst_aggregators[jj][0]].aggregate(lst_aggregators[jj][1])
            # plot histogram
            ax_row[jj].bar(agg.index, agg.values, width=1)
            ax_row[jj].set_title(titles[idx] + '_' + counter[jj])
    
    # polish chart and save
    fig.suptitle('Seasonal Chart', fontsize=24)
    plt.show()
    mgr = plt.get_current_fig_manager()
    mgr.window.showMaximized()
    fig.savefig('..\\figures\\seasonal')
    return (fig,ax)
    
    
def make_trip_vs_crow_distance_scatter(hd_store):
    fig, ax = plt.subplots(1,1)
    trip_mean = hd_store['df']['trip_km'].mean()
    crow_mean = hd_store['df']['crow_km'].mean()
    
    trip_std = hd_store['df']['trip_km'].std()
    crow_std = hd_store['df']['crow_km'].std()
    
    def include_trip(trip,z_scores=3.0):
        include = True
        include = include and trip['trip_km'] > (trip_mean - z_scores*trip_std)
        include = include and trip['trip_km'] < (trip_mean + z_scores*trip_std)
        include = include and trip['crow_km'] > (crow_mean - z_scores*crow_std)
        include = include and trip['crow_km'] < (crow_mean + z_scores*crow_std)
        return include
        
    included_trips = hd_store['df'].apply(include_trip,axis=1)
    
    x = hd_store['df']['trip_km'][included_trips]
    y = hd_store['df']['crow_km'][included_trips]
    
    
    plt.hexbin(x,y, cmap=plt.cm.Blues,gridsize=100)
    
    plt.show()
    fig.savefig('..\\figures\\trip_v_crow')
    return fig, ax


if __name__ == '__main__':
    
    FORCE_RELOAD = False
    if FORCE_RELOAD or not os.path.exists(HDFSTORE_PATH):
        print('re-establishing HDFStore from scratch')        
        os.remove(HDFSTORE_PATH) if os.path.exists(HDFSTORE_PATH) else None
        hd_store = parse_and_hdfstore_data(DEBUG_MODE=False)
    else:
        print('attempting to load preexisting HDFStore')
        hd_store = HDFStore(HDFSTORE_PATH)
    
    
    plt.ioff()
        
    # plot all points visited on trip and save figure
    print('creating a graph of all trips')
    f_trips, ax_trips = graph_all_trips(hd_store)
    plt.close(f_trips)
    
    #lets do some k-means clustering on the destinations to see if we can find any structure
    print('doing some k-means clustering on the destinations')
    f_kmeans, ax_kmeans, lst_kmeans = destination_clustering(hd_store)
    plt.close(f_kmeans)
    
    # observation: the marginal improvement in the explanation of variance with clusters tapers off by k = 5 or 6
    K = 5
    km = lst_kmeans[5]
    print('creating a voronoi diagram with K = %s' % K)
    fig_voronoi, ax_voronoi = voronoi_graph(km,hd_store)
    plt.close(fig_voronoi)
    # clearly there are a few major regions to this city
        
    # let's build a transition matrix to understand what is going on
    print('constructing a transition matrix')
    df_ctab, f_ctab, ax_ctab = build_transition_matrix(km,hd_store)
    plt.close(f_ctab)
    
    # show ride vs cab distribution
    print('plot the distribution of rides')
    f_gini, ax_gini = make_ride_distribution_chart(hd_store)
    plt.close(f_gini)
    
    # now lets do some seasonal analysis
    print('plot seasonality')
    f_seas, ax_seas = make_seasonal_graphs(hd_store)
    plt.close(f_seas)
    # in lieu of making a street map to compute distances/ETAs, lets do a rough approximation based on eucliean as-crow-flies-distnace
    # estimate the ratio of trip length to crow-flies distance
    print('scatterplot of trip vs crow distance')
    f_tvc, ax_tvc = make_trip_vs_crow_distance_scatter(hd_store)
    plt.close(f_tvc)
    
    
