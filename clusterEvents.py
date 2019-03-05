#import the different packages used throughout
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

def save_s3_data(labels,eps,minSamples,Data,Time,filename):
    #package the matrices as a dataset to save as a netcdf
    data_events = xr.Dataset(
        data_vars = {'Data': (('time', 'vector'),Data), 
                     'Labels': (('time'),labels)},
        coords = {'time': Time,
                  'vector': range(len(Data[0,:]))},
        attrs = {'eps': eps,
                'minimumSamples': minSamples})

    #save as a netcdf
    data_events.to_netcdf(path = filename+"Clustered_Data.nc4", compute = True)
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('himatdata')
    home = os.getcwd()
    
    bucket.upload_file(filename+"Clustered_Data.nc4",'Trmm/EPO/Cluster_results_March5/'+filename+'_Clustered_Data.nc4')

    os.remove(filename+"Clustered_Data.nc4")

#function that reads local data from TRMM in the EC2 instances
def read_TRMM_data(year,month,SR_minrate):
    #create empty matrices to hold the extracted data
    Lat_Heat = []
    surf_r = []
    LAT = []
    LON = []
    TIME = []
    count = 0
    A = []
    logging.info("in read TRMM")
    filename = str(year)+"_"+str(month).zfill(2)

    #Load in data for that month
    for file in glob.glob("data/Trmm/EPO/"+filename+"/*.nc4"):
        logging.info("Downloaded file: %s", file)

        L, S, A, la, lo, Ti = extract_data(xr.open_dataset(file),SR_minrate)
        #append the new data in the matrices
        if count==0:
            Lat_Heat = L
            LAT = la
            LON = lo
            TIME = Ti
            count += 1
        else:
            Lat_Heat = np.concatenate((Lat_Heat,L),axis =0)
            LAT = np.concatenate((LAT,la),axis =0)
            LON = np.concatenate((LON,lo),axis =0)
            TIME = np.concatenate((TIME,Ti),axis =0)
        surf_r = np.append(surf_r,S)

    #Load in previous 5 days of data
    year_prev = year
    month_prev = month-1
    if month==1: 
        year_prev = year-1
        month_prev = 12

    if year_prev>1997:
        filename = str(year_prev)+"_"+str(month_prev).zfill(2)
        files = glob.glob("data/Trmm/EPO/"+filename+"/*.nc4")
        days = [int(f[-17:-15]) for f in files]
        indices = np.argwhere(days>np.max(days)-5)

        for i in range(len(indices)):
            file = files[int(indices[i])]

            L, S, A, la, lo, Ti = extract_data(xr.open_dataset(file),SR_minrate)
            #append the new data in the matrices
            Lat_Heat = np.concatenate((Lat_Heat,L),axis =0)
            LAT = np.concatenate((LAT,la),axis =0)
            LON = np.concatenate((LON,lo),axis =0)
            TIME = np.concatenate((TIME,Ti),axis =0)
            surf_r = np.append(surf_r,S)

    #Load in next 5 days of data
    year_next = year
    month_next = month+1
    if month==12: 
        year_next = year+1
        month_next = 1

    if year_next<2014:
        filename = str(year_next)+"_"+str(month_next).zfill(2)
        files = glob.glob("data/Trmm/EPO/"+filename+"/*.nc4")
        days = [int(f[-17:-15]) for f in files]
        indices = np.argwhere(days<np.min(days)+5)

        for i in range(len(indices)):
            file = files[int(indices[i])]

            L, S, A, la, lo, Ti = extract_data(xr.open_dataset(file),SR_minrate)
            #append the new data in the matrices
            Lat_Heat = np.concatenate((Lat_Heat,L),axis =0)
            LAT = np.concatenate((LAT,la),axis =0)
            LON = np.concatenate((LON,lo),axis =0)
            TIME = np.concatenate((TIME,Ti),axis =0)
            surf_r = np.append(surf_r,S)

    #Put all the data into one array, where rows are individual observations and the columns are 
    #[Latitude, Longitude, Surface Rain, Latent Heat Profile]
    Data = np.concatenate((LAT.reshape(len(LAT),1),LON.reshape(len(LON),1),surf_r.reshape(len(surf_r),1),Lat_Heat),axis=1)
    Data = np.squeeze(Data)

    #Remove repeated values
    uniqueData, indices = np.unique(Data,axis=0,return_index=True)

    return uniqueData, TIME[indices], A
    
#function that connects to the S3 bucket, downloads the file, reads in the data, and deletes the file
def load_s3_data(SR_minrate):
    #create empty matrices to hold the extracted data
    Lat_Heat = []
    surf_r = []
    LAT = []
    LON = []
    TIME = []
    count = 0
    
    home = expanduser("~")

    with open(os.path.join(home,'creds.json')) as creds_file:
        creds_data = json.load(creds_file)

    #Access from S3
    s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
             aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
    bucket = s3.Bucket('himatdata')
    home = os.getcwd()

    for obj in bucket.objects.filter(Delimiter='', Prefix='Trmm/EPO/2000_01/'):
        if obj.key[-4:] == ".nc4":

            bucket.download_file(obj.key,os.path.join(os.path.join(home,'S3_downloads/',obj.key[17:])))

        #file = 'oneProfile/TPR7_uw1_00538.19980101.000558_EPO.nc4'
            L, S, A, la, lo, Ti = extract_data(xr.open_dataset(os.path.join(home,'S3_downloads/',obj.key[17:])),SR_minrate)
           #append the new data in the matrices
            if count==0:
                Lat_Heat = L
                LAT = la
                LON = lo
                TIME = Ti
                count += 1
            else:
                Lat_Heat = np.concatenate((Lat_Heat,L),axis =0)
                LAT = np.concatenate((LAT,la),axis =0)
                LON = np.concatenate((LON,lo),axis =0)
                TIME = np.concatenate((TIME,Ti),axis =0)
            surf_r = np.append(surf_r,S)
             
            #delete the local file
            os.remove(os.path.join(home,'S3_downloads/',obj.key[17:]))
        

    #Put all the data into one array, where rows are individual observations and the columns are 
    #[Latitude, Longitude, Surface Rain, Latent Heat Profile]
    Data = np.concatenate((LAT.reshape(len(LAT),1),LON.reshape(len(LON),1),surf_r.reshape(len(surf_r),1),Lat_Heat),axis=1)
    Data = np.squeeze(Data)
    
    #Remove repeated values
    uniqueData = np.unique(Data,axis=0)
    
    return uniqueData, TIME, A
    
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
        elif np.max(tcluster).month==nmonth & np.max(tcluster).day>4:
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
def data_to_cluster(Data):
    #Extract [Lat, Lon, DeltaTime]
    Xdata = np.vstack((Data[:,1],Data[:,2],Data[:,0]))
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

def extract_data(file, SR_min=5):
    #Extract the data you want from file
    altitude_lh = file.altitude_lh.data
    surf_rain = file.surf_rain.data
    latent_heating = file.latent_heating.data

    lat = file.latitude.data
    lon = file.longitude.data
    time = file.time.data
    
    #create grid of altitude, lat, and lon coordinates
    LON, LAT = np.meshgrid(lon, lat)

    #size of lat and lon as variables
    nlat = len(lat)
    nlon = len(lon)
    nalt = len(altitude_lh)

    #reshape as column vector (note the indicing is now column*ncolumns+row)
    surf_rain = np.reshape(surf_rain,[nlat*nlon])
    LH = np.reshape(latent_heating,[nalt,nlat*nlon])
    LON = np.reshape(LON,[nlat*nlon])
    LAT = np.reshape(LAT,[nlat*nlon])

    #Remove values with NaN and rainfall less than cut-off
    surf_R = surf_rain[~np.isnan(surf_rain)]
    surf_r = surf_R[surf_R>=SR_min]

    Lat_Heat = LH[:,~np.isnan(surf_rain)]
    Lat_Heat = Lat_Heat[:,surf_R>=SR_min]
    Lat_Heat = np.squeeze(Lat_Heat)

    LAT = LAT[~np.isnan(surf_rain)]
    LAT = LAT[surf_R>=SR_min]
    LAT = np.squeeze(LAT)

    LON = LON[~np.isnan(surf_rain)]
    LON = LON[surf_R>=SR_min]
    LON = np.squeeze(LON)

    #Remove any profiles where there is missing latent heat info
    surf_r = surf_r[~pd.isnull(Lat_Heat).any(axis=0)]
    LAT = LAT[~pd.isnull(Lat_Heat).any(axis=0)]
    LON = LON[~pd.isnull(Lat_Heat).any(axis=0)]
    Lat_Heat = Lat_Heat[:,~pd.isnull(Lat_Heat).any(axis=0)]
    Time = np.repeat(time,len(surf_r))
    
    return Lat_Heat.T, surf_r.T, altitude_lh, LAT.T, LON.T, Time.T

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
#    Data, Time, A = load_s3_data(SR_minrate)
    Data, Time, A = read_TRMM_data(year,month,SR_minrate)
    DeltaTime = time_to_deltaTime(Time)
    
    Data = np.concatenate((DeltaTime.reshape(len(DeltaTime),1), Data), axis=1)
    Data = np.squeeze(Data)
    
    DatatoCluster = data_to_cluster(Data)
    
    logging.info("Determining parameters")

    # eps, min_samples = optimal_params(Data[0:int(len(DatatoCluster)*opt_frac),:])
    
    eps = MesoScale 
    min_samples = 21
    
    labels = cluster_and_label_data(DatatoCluster,eps,min_samples)
    logging.info("Fit the Data!")
    
    Data, Time, labels = remove_dublicate(Data, Time, labels, month, year)

    save_s3_data(labels,eps,min_samples,Data,Time,filename)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Script run DBSCAN clustering on TRMM data')
    parser.add_argument('-y', '--year')
    args = parser.parse_args()
    year = int(args.year)
    for month in range(1,13):
        logging.info("In Month: %s", (month))
        main_script(year,month)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
