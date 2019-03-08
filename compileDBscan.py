#import the different packages used throughout
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
import math
import datetime
import boto3
from os.path import expanduser
import os
import json
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def read_Data():
	#loop through data downloaded from S3 and compile it into a large array of all data, times, and labels.
	#make sure the label numbers are unique between files

	#create empty matrices
    Data = []
    labels = []
    Time = []
    count = 0

    #Load in data for that month
    for file in glob.glob('S3_downloads/*.nc4'):
        print("Opening file: %s", file)
        F = xr.open_dataset(file)
        month = float(file[-21:-19])
        pmonth = month-1
        if pmonth<0: pmonth = 12
            
        #remove the data that wasn't clustered (cluster number = -1) and append to the dataset
        if count==0:
            D = F.Data.data
            T = F.time.data
            L = F.Labels.data
            ind = L>-1
            D = D[ind,:]
            T = T[ind]
            L = L[ind]
            Data = D
            labels = L
            Time = T
            count += 1
        else:
            D = F.Data.data
            T = F.time.data
            L = F.Labels.data
            ind = L>-1
            D = D[ind,:]
            T = T[ind]
            L = L[ind]
            
            Data = np.concatenate((Data,D),axis=0)
            L += np.max(labels)
            labels = np.concatenate((labels,L),axis=0)
            Time = np.concatenate((Time,T),axis =0)
    
    return Data, labels, Time

 def Dataset_stats(Data, Labels):
 	#calculate the mean and span of each cluster, as well as the observations in each cluster

    num_clusters_ = np.unique(Labels) #dbscan cluster numbers
    n_clusters_ = len(num_clusters_) #total clusters in set

    #empty matrices to hold the output
    cluster_spans = np.zeros((n_clusters_, len(Data[0,:])))
    cluster_means = np.zeros((n_clusters_,len(Data[0,:])))
    cluster_count = np.zeros((n_clusters_))

    #for each cluster, calculate the mean, range, and count the number of observations
    for i in range(n_clusters_):
        cluster = Data[Labels==num_clusters_[i],:]
        cluster_spans[i,:] = np.amax(cluster,axis=0)-np.amin(cluster,axis=0)
        cluster_means[i,:] = np.nanmean(cluster,axis=0)
        cluster_count[i] = len(cluster)
        
    return cluster_spans, cluster_means, cluster_count

def Download_DBscanOutput():
	#download the output from DBscan for local access
	home = expanduser("~")

	with open(os.path.join(home,'creds.json')) as creds_file:
	    creds_data = json.load(creds_file)

	#Access from S3
	s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
	         aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
	bucket = s3.Bucket('himatdata')
	home = os.getcwd()

	#for every netcdf file in the cluster results download it to the S3_Downloads folder
	for obj in bucket.objects.filter(Delimiter='/', Prefix='Trmm/EPO/Cluster_results_March5/' ):
        if obj.key[-4:] == ".nc4":
            print(obj.key)
            bucket.download_file(obj.key,os.path.join(os.path.join(home,'S3_downloads/',obj.key[-26:])))


def kmeans_Cluster(Xdata,max_clusters, nscore):
	#clusters using kmeans Xdata into 2 to max_clusters, and calculates the silhouette score nscore times for each k count

	n_clusters = range(2,max_clusters)

	#stndardize the data before clustering
	Xscaler = StandardScaler()
	X = Xscaler.fit_transform(Xdata)

	#empty matrix to hold output from clustering
	met = np.zeros((len(n_clusters),nscore))

	#for each number of clusters to create, fit the k means and calculate the silhouette score nscore times
	for i in range(len(n_clusters)):
	    kmeans = KMeans(n_clusters=n_clusters[i], random_state=0).fit(X)
	    for j in range(nscore):
	        met[i,j] = metrics.silhouette_score(Xdata, kmeans.labels_, sample_size=10000, random_state=j*10)

	 return n_clusters, met

def save_compiled_Data(Data,labels,Time,cluster_spans,cluster_means,cluster_count):

	#create xarray data array
	num_clusters_db = np.unique(labels)
	data_events = xr.Dataset(
	    data_vars = {'Latent Heat': (('obs_number', 'altitude'),Data[:,4:]), 
	                 'Rain Rate': (('obs_number'), Data[:,3]),
	                 'DBLabel': (('obs_number'), labels),
	                 'Time': (('obs_number'), Time),
	                 'Latitude': (('obs_number'),Data[:,1]),
	                 'Longitude': (('obs_number'), Data[:,2]),
	                 'DB_spans Latent Heat': (('db_number','altitude'),cluster_spans[:,4:]),
	                 'DB_means Latent Heat': (('db_number','altitude'),cluster_means[:,4:]),
	                 'DB_spans Rain Rate': (('db_number'),cluster_spans[:,3]),
	                 'DB_means Rain Rate': (('db_number'),cluster_means[:,3]),
	                 'DB_spans Time': (('db_number'),cluster_spans[:,0]),
	                 'DB_means Time': (('db_number'),cluster_means[:,0]),
	                 'DB_count': (('db_number'),cluster_count)},
	    coords = {'obs_number': range(len(Time)),
	              'altitude': np.array([.5, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5]),
	              'db_number': num_clusters_db})

	data_events.to_netcdf(path = "DB_compiled_Clustered_Data.nc4", compute = True)

	home = expanduser("~")

	with open(os.path.join(home,'creds.json')) as creds_file:
	    creds_data = json.load(creds_file)

	#Access from S3
	s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
	         aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
	bucket = s3.Bucket('himatdata')
	home = os.getcwd()

	bucket.upload_file('DB_compiled_Clustered_Data.nc4','Trmm/EPO/Cluster_results_March5/DB_compiled_Clustered_Data.nc4')

	os.remove('DB_compiled_Clustered_Data.nc4')

def save_kmeans_data(k_n_clusters, met):
	#save the results of the  kmeans clustering to S3
	data_clusters = xr.Dataset(
		data_vars = {'silhouette_score': (('k_n_clusters','iterations'),met)},
		coords = {'k_n_clusters': k_n_clusters,
				  'iterations': range(len(met[0,:]))})
	data_clusters.to_netcdf(path = "kmeans_output.nc4", compute=True)

	home = expanduser("~")

	with open(os.path.join(home,'creds.json')) as creds_file:
	    creds_data = json.load(creds_file)

	#Access from S3
	s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
	         aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
	bucket = s3.Bucket('himatdata')
	home = os.getcwd()

	bucket.upload_file('kmeans_output.nc4','Trmm/EPO/Cluster_results_March5/kmeans_output.nc4')

	os.remove('kmeans_output.nc4')


if __name__ == '__main__':
    start_time = time.time()
    Download_DBscanOutput()
    Data, labels, Time = read_Data()
    cluster_spans, cluster_means, cluster_count = Dataset_stats(Data,labels)

    save_compiled_Data(Data,labels,Time,cluster_spans,cluster_means,cluster_count)

    k_n_clusters, met = kmeans_Cluster(cluster_means[:,4:],20,10)
    save_kmeans_data(k_n_clusters,met)

    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))