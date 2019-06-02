#import the different packages used throughoue#import the different packages used throughout
#from mpl_toolkits.basemap import Basemap
import xarray as xr
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances, davies_bouldin_score
from bayes_opt import BayesianOptimization
import math
import boto3
import os
from os.path import expanduser
import json
import time
import logging
import argparse
ROOT_DIR = '/home/ubuntu/precip/Precip_eScience/'
os.chdir(ROOT_DIR)
logging.basicConfig(filename='trmm.log', level=logging.INFO)

def extract_regionalData(year,month,region,latmin,latmax,longmin,longmax,runningNum):
    SURF_RAIN = np.empty((0))
    Latent_Heating = np.empty((0,19))
    logging.info(Latent_Heating.shape)
    LAT = np.empty((0))
    LONG = np.empty((0))
    TIME = np.empty((0),dtype='datetime64')
    Rain_Type = np.empty((0))

    filename = str(year)+"_"+str(month).zfill(2)
    files = glob.glob("data/Trmm/"+region+'/'+filename+"/*.nc4")
    for File in files:
        try:
            regionalXarray = xr.open_dataset(File) 
            Surf_Rain = regionalXarray.surf_rain.values.flatten()
            Surf_Rain = np.nan_to_num(Surf_Rain)
            [Lat,Time,Long] = np.meshgrid(regionalXarray.latitude.values,regionalXarray.time.values,regionalXarray.longitude.values)
            Lat = Lat.flatten()
            Long = Long.flatten()
            Time = Time.flatten()

            keep_indices = np.where((Surf_Rain>.4)&(Lat>latmin)&(Lat<latmax)&(Long>longmin)&(Long<longmax))
            Latent_Heating = np.append(Latent_Heating,np.squeeze(np.reshape(np.moveaxis(regionalXarray.latent_heating.values,1,3),(-1,19))[keep_indices,:]),axis=0)
            
            SURF_RAIN = np.append(SURF_RAIN,Surf_Rain[keep_indices])
            LAT = np.append(LAT,Lat[keep_indices])
            LONG = np.append(LONG,Long[keep_indices])
            TIME = np.append(TIME,np.array(Time[keep_indices],dtype='datetime64'))
            Rain_Type = np.append(Rain_Type,regionalXarray.rain_type.values.flatten()[keep_indices])
        except:
            logging.info(File)


    logging.info(Latent_Heating.shape)    
    regionalXarray = xr.Dataset({'surf_rain': (['clusteredCoords'], SURF_RAIN),
                                'latent_heating': (['clusteredCoords','altitude_lh'], Latent_Heating),
                                'latitude': (['clusteredCoords'], LAT),
                                'longitude': (['clusteredCoords'], LONG),
                                'time': (['clusteredCoords'], TIME),
                                'rain_type': (['clusteredCoords'],Rain_Type)},
                                coords = {'clusteredCoords': runningNum + np.arange(len(TIME)),
                                        'altitude_lh': np.array(regionalXarray.altitude_lh)})

    logging.info('made new array')
    runningNum = runningNum + len(TIME)

    return regionalXarray, runningNum


def save_s3_data(labels,eps,globalArray,filename):
    #package the matrices as a dataset to save as a netcdf
    labelsArray = xr.Dataset(
        data_vars = {'Labels': (('clusteredCoords'),labels)},
        coords = {'clusteredCoords': globalArray.clusteredCoords})

    globalArray = globalArray.merge(labelsArray)

    #save as a netcdf
    globalArray.to_netcdf(path = filename+"Clustered_Data_Globe" + eps + ".nc4", compute = True)
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('trmm')
    home = os.getcwd()
    
    bucket.upload_file(filename+"Clustered_Data_Globe" + eps + ".nc4",'trmm/'+filename+"Clustered_Data_Globe" + eps + ".nc4")

    os.remove(filename+"Clustered_Data_Globe" + eps + ".nc4")

#function that reads local data from TRMM in the EC2 instances
def read_TRMM_data(year,month):
    #create empty matrices to hold the extracted data
    
    logging.info("in read TRMM")
    globalArray = []
    regionNames = ['H01']#['EPO', 'AFC', 'CIO', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'MSA', 'SAM', 'SAS', 'TRA', 'USA', 'WMP', 'WPO']
    latmin = -90
    latmax = 90
    longmin = -30
    longmax = -20
    runningNum = 0
    #Load in data for that month for each region
    for region in regionNames:
        filename = str(year)+"_"+str(month).zfill(2)
        regionalArray, runningNum = extract_regionalData(year,month,region,latmin,latmax,longmin,longmax,runningNum)
        globalArray.append(regionalArray)

        #Load in previous day of data
        year_prev = year
        month_prev = month-1
        if month==1: 
            year_prev = year-1
            month_prev = 12

        if year_prev>1997:
            filename = str(year_prev)+"_"+str(month_prev).zfill(2)
            files = glob.glob("data/Trmm/"+region+"/"+filename+"/*.nc4")
            days = [int(f[-17:-15]) for f in files]
            indices = np.argwhere(days>np.max(days)-1)
            logging.info(indices)
            files = files[np.array(indices).astype(int)]

            try:
                regionalXarray = xr.open_dataset(File) 
                Surf_Rain = regionalXarray.surf_rain.values.flatten()
                Surf_Rain = np.nan_to_num(Surf_Rain)
                [Lat,Time,Long] = np.meshgrid(regionalXarray.latitude.values,regionalXarray.time.values,regionalXarray.longitude.values)
                Lat = Lat.flatten()
                Long = Long.flatten()
                Time = Time.flatten()

                keep_indices = np.where((Surf_Rain>.4)&(Lat>latmin)&(Lat<latmax)&(Long>longmin)&(Long<longmax))
                
                Latent_Heating = np.append(Latent_Heating,np.squeeze(np.reshape(np.moveaxis(regionalXarray.latent_heating.values,1,3),(-1,19))[keep_indices,:]),axis=0)
                SURF_RAIN = np.append(SURF_RAIN,Surf_Rain[keep_indices])
                LAT = np.append(LAT,Lat[keep_indices])
                LONG = np.append(LONG,Long[keep_indices])
                TIME = np.append(TIME,np.array(Time[keep_indices],dtype='datetime64'))
                Rain_Type = np.append(Rain_Type,regionalXarray.rain_type.values.flatten()[keep_indices])
            except:
                logging.info(File)


            logging.info(Latent_Heating.shape)    
            regionalXarray = xr.Dataset({'surf_rain': (['clusteredCoords'], SURF_RAIN),
                                    'latent_heating': (['clusteredCoords','altitude_lh'], Latent_Heating),
                                'latitude': (['clusteredCoords'], LAT),
                                'longitude': (['clusteredCoords'], LONG),
                                'time': (['clusteredCoords'], TIME),
                                'rain_type': (['clusteredCoords'],Rain_Type)},
                                coords = {'clusteredCoords': runningNum + np.arange(len(TIME)),
                                        'altitude_lh': np.array(regionalXarray.altitude_lh)})

            globalArray.append(regionalXarray)
            runningNum = runningNum + len(TIME)


            # for i in range(len(indices)):
            #     file = files[int(indices[i])]

            #     singleXarray = xr.open_dataset(file)
            #     singleXarray = singleXarray.drop('swath')
            #     stackedArray = singleXarray
            #     #stackedArray = singleXarray.stack(clusteredCoords=('latitude', 'longitude','time'))
            #     stackedArray.where(stackedArray.surf_rain>.4,drop=True)
            #     globalArray.append(stackedArray)

        #Load in next day of data
        year_next = year
        month_next = month+1
        if month==12: 
            year_next = year+1
            month_next = 1

        if year_next<2014:
            filename = str(year_next)+"_"+str(month_next).zfill(2)
            files = glob.glob("data/Trmm/"+region+"/"+filename+"/*.nc4")
            days = [int(f[-17:-15]) for f in files]
            indices = np.argwhere(days<np.min(days)+1)
            logging.info(indices)
            files = files[indices.astype(int)]

            SURF_RAIN = np.empty((0))
            Latent_Heating = np.empty((0,19))
            LAT = np.empty((0))
            LONG = np.empty((0))
            TIME = np.empty((0),dtype='datetime64')
            Rain_Type = np.empty((0))

            filename = str(year)+"_"+str(month).zfill(2)
            files = glob.glob("data/Trmm/"+region+'/'+filename+"/*.nc4")
            for File in files:

                try:
                    regionalXarray = xr.open_dataset(File) 
                    Surf_Rain = regionalXarray.surf_rain.values.flatten()
                    Surf_Rain = np.nan_to_num(Surf_Rain)
                    [Lat,Time,Long] = np.meshgrid(regionalXarray.latitude.values,regionalXarray.time.values,regionalXarray.longitude.values)
                    Lat = Lat.flatten()
                    Long = Long.flatten()
                    Time = Time.flatten()

                    keep_indices = np.where((Surf_Rain>.4)&(Lat>latmin)&(Lat<latmax)&(Long>longmin)&(Long<longmax))
                    
                    Latent_Heating = np.append(Latent_Heating,np.squeeze(np.reshape(np.moveaxis(regionalXarray.latent_heating.values,1,3),(-1,19))[keep_indices,:]),axis=0)
                    SURF_RAIN = np.append(SURF_RAIN,Surf_Rain[keep_indices])
                    LAT = np.append(LAT,Lat[keep_indices])
                    LONG = np.append(LONG,Long[keep_indices])
                    TIME = np.append(TIME,np.array(Time[keep_indices],dtype='datetime64'))
                    Rain_Type = np.append(Rain_Type,regionalXarray.rain_type.values.flatten()[keep_indices])
                except:
                    logging.info(File)


            logging.info(Latent_Heating.shape)    
            regionalXarray = xr.Dataset({'surf_rain': (['clusteredCoords'], SURF_RAIN),
                                    'latent_heating': (['clusteredCoords','altitude_lh'], Latent_Heating),
                                'latitude': (['clusteredCoords'], LAT),
                                'longitude': (['clusteredCoords'], LONG),
                                'time': (['clusteredCoords'], TIME),
                                'rain_type': (['clusteredCoords'],Rain_Type)},
                                coords = {'clusteredCoords': runningNum + np.arange(len(TIME)),
                                        'altitude_lh': np.array(regionalXarray.altitude_lh)})

            globalArray.append(regionalXarray)
            runningNum = runningNum + len(TIME)

            
    globalArray = xr.concat(globalArray)
    logging.info('successful combo of arrays')
    #save as a netcdf
    globalArray.to_netcdf(path = filename+"Clustered_Data_Globe_test_AllTRMMData.nc4", compute = True)
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('trmm')
    home = os.getcwd()
    
    bucket.upload_file(filename+"Clustered_Data_Globe_test_AllTRMMData.nc4",'trmm/'+filename+"Clustered_Data_Globe_test_AllTRMMData.nc4")

    os.remove(filename+"Clustered_Data_Globe_test_AllTRMMData.nc4")
    #globalArray = xr.auto_combine(globalArray,concat_dim='clusteredCoords')

    return globalArray
    
#function that connects to the S3 bucket, downloads the file, reads in the data, and deletes the file
def download_s3_data(year,month):
    regionNames = ['EPO', 'AFC', 'CIO', 'H01', 'H02', 'H03', 'H04', 'H05', 'H06', 'H07', 'H08', 'MSA', 'SAM', 'SAS', 'TRA', 'USA', 'WMP', 'WPO']
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('trmm')
    home = os.getcwd()

    #Load in data for that month for each region
    for region in regionNames:
        filename = str(year)+"_"+str(month).zfill(2)
        logging.info(filename + " " + region)
        if not os.path.exists(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/')):
            os.makedirs(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/'))
        for obj in bucket.objects.filter(Delimiter='', Prefix= region+'/'+filename+'/'):
            if obj.key[-4:] == ".nc4":
                logging.info(obj.key)
                bucket.download_file(obj.key,os.path.join(home,'data/Trmm/'+obj.key))

        #download previous month of data
        year_prev = year
        month_prev = month-1
        if month==1: 
            year_prev = year-1
            month_prev = 12

        if year_prev>1997:
            filename = str(year_prev)+"_"+str(month_prev).zfill(2)
            if not os.path.exists(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/')):
                os.makedirs(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/'))
            for obj in bucket.objects.filter(Delimiter='', Prefix=region+'/'+filename+'/'):
                if obj.key[-4:] == ".nc4":

                    bucket.download_file(obj.key,os.path.join(os.path.join(home,'data/Trmm/'+region+'/'+filename,obj.key[17:])))

        #download  next month of data
        year_next = year
        month_next = month+1
        if month==12: 
            year_next = year+1
            month_next = 1

        if year_next<2014:
            filename = str(year_next)+"_"+str(month_next).zfill(2)
            if not os.path.exists(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/')):
                os.makedirs(os.path.join(home,'data/Trmm/'+region+'/'+filename+'/'))
            for obj in bucket.objects.filter(Delimiter='', Prefix=region+'/'+filename+'/'):
                if obj.key[-4:] == ".nc4":

                    bucket.download_file(obj.key,os.path.join(os.path.join(home,'data/Trmm/'+region+'/'+filename,obj.key[17:])))
    return
    
#Translate the time into delta time since the first datapoint (in hours)
def time_to_deltaTime(Time):
    InitialTime = np.min(Time)
    DeltaTime = []
    DeltaTime[:] = [int(x-InitialTime)/(10**9*60*60) for x in Time] #from nanoseconds to hours
    DeltaTime = np.array(DeltaTime) #convert from list to array
    
    return DeltaTime

#remove clusters in 5 days of next month and only in previous month
def remove_dublicate(Data, Time, labels, month, year):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    time = pd.DatetimeIndex(Time)
    index = np.argwhere((labels==-1) & (time.month!=month))

    Data = np.delete(Data,index,0)
    Time = np.delete(Time,index)
    time = np.delete(time,index)
    labels = np.delete(labels,index)
    
    nmonth = month+1
    pmonth = month-1
    if nmonth>12: nmonth = 1
    if pmonth<1: pmonth = 12

    for i in range(n_clusters_):
        cluster = Data[labels==i,:]
        tcluster = time[labels==i]
        #remove clusters exclusively in the next month (captured in the next month)
        if np.amin(np.array(tcluster.month))==nmonth:
            Data = Data[labels!=i,:]
            Time = Time[labels!=i]
            time = time[labels!=i]
            labels = labels[labels!=i]
        #remove clusters who end in the last day of the next month (captured in the next month)
        elif np.max(tcluster).month==nmonth & np.max(tcluster).day>1:
            Data = Data[labels!=i,:]
            Time = Time[labels!=i]
            time = time[labels!=i]
            labels = labels[labels!=i]
        #remove clusters exclusively in the previous month
        if np.amax(np.array(tcluster.month))==pmonth:
            Data = Data[labels!=i,:]
            Time = Time[labels!=i]
            time = time[labels!=i]
            labels = labels[labels!=i]

    return Data, Time, labels
#Create array to Cluster the rainfall events, Scale the grid lat/lon so it is weighted 'fairly' compared to time
def data_to_cluster(stackedArray):
    #Extract [Lat, Lon, DeltaTime]

    Xdata = np.array([np.array(stackedArray.latitude),np.array(stackedArray.longitude),time_to_deltaTime(np.array(stackedArray.time))])
    Xdata = Xdata.T

    return Xdata

def cluster_and_label_data(Distance,eps,min_samps):
    model = DBSCAN(eps=eps, min_samples=min_samps,metric=distance_sphere_and_time)
    model.fit(Distance)

    labels = model.labels_
    
    return labels

def cluster_optics_labels(Data,eps,min_samps):
    #model = DBSCAN(eps=eps, min_samples=min_samps,metric='precomputed')
    model = OPTICS(max_eps=eps*1000,min_samples=min_samps,metric=distance_sphere_and_time)
    model.fit(Data)

    labels = model.labels_
    
    return labels

#Use Bayesian Optimization on the data to get the best parameters for the clustering
def optimal_params_optics(Data):
    Opt = optimize_optics(Data,'davies') #it seems like silhouette takes substantially longer?
    min_samples = int(Opt['params']['min_samples'])
    max_eps = Opt['params']['max_eps']

    return max_eps, min_samples

#function that fits dbscan for given parameters and returns the davies bouldin score evaluation metric 
def optics_eval_db(max_eps,min_samples,data):
    model = OPTICS(max_eps=max_eps, min_samples=min_samples,metric=distance_sphere_and_time)
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = davies_bouldin_score(data, labels)
        
    return score

#function that fits dbscan for given parameters and returns the silhouette score evaluation metric 
def optics_eval_sil(max_eps,min_samples,data):
    model = OPTICS(max_eps=max_eps, min_samples=min_samples,metric=distance_sphere_and_time)
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = metrics.silhouette_score(data,labels)
        
    return score

#Applies bayesian optimization to determine DBSCAN parameters that maximize the evaluation metric (specified as input)
def optimize_optics(data,metric='silhouette'):
    """Apply Bayesian Optimization to DBSCAN parameters."""
    def optics_evaluation_sil(max_eps, min_samples):
        """Wrapper of DBSCAN evaluation."""
        min_samples = int(min_samples) #insure that you are using an integer value for the minimum samples parameter
        return optics_eval_sil(max_eps=max_eps, min_samples=min_samples, data=data)

    def optics_evaluation_db(max_eps, min_samples):
        """Wrapper of DBSCAN evaluation."""
        min_samples = int(min_samples) #insure that you are using an integer value for the minimum samples parameter
        return optics_eval_db(max_eps=max_eps, min_samples=min_samples, data=data)

    if metric == 'davies':
        optimizer = BayesianOptimization(
            f=optics_evaluation_db,
            pbounds={"max_eps": (10, 25000000), "min_samples": (5, 25)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=2
        )
        
    else:
        optimizer = BayesianOptimization(
            f=optics_evaluation_sil,
            pbounds={"max_eps": (10, 250*100000), "min_samples": (5, 25)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=2
        )
    
    optimizer.maximize(init_points=10, n_iter=10)

    logging.info("Final Result: %s", optimizer.max)
    return optimizer.max

#Use Bayesian Optimization on the data to get the best parameters for the clustering
def optimal_params(Data):
    Opt = optimize_dbscan(Data,'davies') #it seems like silhouette takes substantially longer?
    min_samps = int(Opt['params']['min_samp'])
    eps = Opt['params']['EPS']

    return eps, min_samps

#function that fits dbscan for given parameters and returns the davies bouldin score evaluation metric 
def dbscan_eval_db(eps,min_samples,data):
    model = DBSCAN(eps=eps, min_samples=min_samples,metric=distance_sphere_and_time)
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = davies_bouldin_score(data, labels)
        
    return score

#function that fits dbscan for given parameters and returns the silhouette score evaluation metric 
def dbscan_eval_sil(eps,min_samples,data):
    model = DBSCAN(eps=eps, min_samples=min_samples,metric=distance_sphere_and_time)
    model.fit(data)
    labels = model.labels_
    if len(set(labels))<2:
        score = 0
    else:
        score = metrics.silhouette_score(data,labels)
        
    return score

#Applies bayesian optimization to determine DBSCAN parameters that maximize the evaluation metric (specified as input)
def optimize_dbscan(data,metric='silhouette'):
    """Apply Bayesian Optimization to DBSCAN parameters."""
    def dbscan_evaluation_sil(EPS, min_samp):
        """Wrapper of DBSCAN evaluation."""
        min_samp = int(min_samp) #insure that you are using an integer value for the minimum samples parameter
        return dbscan_eval_sil(eps=EPS, min_samples=min_samp, data=data)

    def dbscan_evaluation_db(EPS, min_samp):
        """Wrapper of DBSCAN evaluation."""
        min_samp = int(min_samp) #insure that you are using an integer value for the minimum samples parameter
        return dbscan_eval_db(eps=EPS, min_samples=min_samp, data=data)

    if metric == 'davies':
        optimizer = BayesianOptimization(
            f=dbscan_evaluation_db,
            pbounds={"EPS": (10, 150), "min_samp": (5, 25)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=0
        )
        
    else:
        optimizer = BayesianOptimization(
            f=dbscan_evaluation_sil,
            pbounds={"EPS": (10, 150), "min_samp": (5, 25)}, #bounds on my parameters - these are very rough guesses right now
            random_state=1234,
            verbose=0
        )
    
    optimizer.maximize(n_iter=10)

    logging.info("Final Result: %s", optimizer.max)
    return optimizer.max

#This function takes in a file name (downloaded from trmm.atmos.washington.edu) and extracts the variables
#that I care about (latitude, longitude, altitude, surface rain, latent heat). It does the inital data checks and
#throws out profiles with missing information or minimal rainfall. It returns the variables I care about that pass these
#checks

#calcuate the distance (in degrees) between 2 points in lat/long
def lat_long_to_arc(lat1,long1,lat2,long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    if cos>1: cos=1
    if cos<-1: cos=-1
    arc = math.acos( cos )

    return arc

def distance_sphere_and_time(x,y):
    Rad_Earth = 6371 #km earth's radius
    MesoScale = 200 #Mesoscale is up to a few hundred km'
    FrontSpeed = 30 # km/h speed at which a front often moves

    Scale_Time_to_Distance = FrontSpeed

    d = Rad_Earth*lat_long_to_arc(x[0],x[1],y[0],y[1])
    D = math.sqrt(d**2+(Scale_Time_to_Distance*(x[2]-y[2]))**2)

    return D

def create_distance_matrix(Data,FrontSpeed,Rad_Earth):
    Scale_Time_to_Distance = FrontSpeed

    Distance = np.zeros((len(Data),len(Data)))
    for i in range(len(Data)):
        for j in range(i,len(Data)):
            d = Rad_Earth*lat_long_to_arc(Data[i,0],Data[i,1],Data[j,0],Data[j,1])
            D = math.sqrt(d**2+(Scale_Time_to_Distance*(Data[i,2]-Data[j,2]))**2)
            Distance[i,j] = D
            Distance[j,i] = D
    return Distance

def main_script(year, month):
    #Define Key Values Here
    SR_minrate = 2 #only keep data with rainrate greater than this value
    opt_frac = .5 #fraction of data to use when determining the optimal dbscan parameters
    Rad_Earth = 6371 #km earth's radius
    MesoScale = 150 #Mesoscale is up to a few hundred km'
    FrontSpeed = 30 # km/h speed at which a front often moves
    filename = str(year)+"_"+str(month).zfill(2)
    # download_s3_data(year,month)
    globalArray = read_TRMM_data(year,month)
    logging.info('process clustering metrics')
    DatatoCluster = data_to_cluster(globalArray)
    logging.info('about to cluster data')
    eps = MesoScale 
    min_samples = 1
    
    labels = cluster_and_label_data(DatatoCluster,eps,min_samples)
    logging.info("Fit the Data!")
    
    save_s3_data(labels,eps,min_samples,globalArray,filename)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Script run DBSCAN clustering on TRMM data')
    parser.add_argument('-y', '--year')
    args = parser.parse_args()
    year = int(args.year)
    month = 7
    # for month in range(1,13):
    logging.info("In Month: %s", (month))
    main_script(year,month)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
