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

def load_Data():
	#download the compiled dbscan output and return the xarray data
	home = expanduser("~")

	with open(os.path.join(home,'creds.json')) as creds_file:
	    creds_data = json.load(creds_file)

	#Access from S3
	s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
	         aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
	bucket = s3.Bucket('himatdata')
	home = os.getcwd()

    bucket.download_file('Trmm/EPO/Cluster_results_March5/DB_compiled_Clustered_Data.nc4',os.path.join(os.path.join(home,'S3_downloads/DB_compiled_Clustered_Data.nc4')))
    F = xr.open_dataset('S3_downloads/DB_compiled_Clustered_Data.nc4')

    return F

def kMeans_fit(F,nclusters):
	#perform the kmeans fit using a specified number of clusters
	Xdata = F.DB_means Latent Heat

	#normalize the data
	Xscaler = StandardScaler()
	X = Xscaler.fit_transform(Xdata)
	
	#fit it with kmeans and get the cluster centers and labels
	kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X)
	klabels = kmeans.labels_
	centers = kmeans.cluster_centers_
	CENTERS = Xscaler.inverse_transform(centers)

	return klabels, CENTERS

def klabel_original_data(F,klabels):
	#assign the k label to each original profile based on the dbscan cluster number
	klabels_originalData = np.repeat(-1,len(F.DBLabel))

	for i in range(len(F.db_number)):
    	klabels_originalData[np.argwhere(F.DBLabel==F.db_number[i])] = klabels[i]

    return klabels_originalData

def save_netCDF(F,klabels_originalData,CENTERS,klabels):
	#save teh results of kmeans into a compiled dataset with the dbscan data

	#xarray of new data to add to the dbscan data
	data_kmeans = xr.Dataset(
	    data_vars = {'KLabel': (('obs_number'), klabels_originalData),
	    			 'K_centers': (('k_number','altitude'),CENTERS)
	    			 'KlabelDB': (('db_number'), klabels)},
	    coords = {'obs_number': F.obs_number,
	              'altitude': F.altitude,
	              'k_number': np.unique(klabels),
	              'db_number': F.db_number})

	#combine the two xarrays into one
	Combined_Array = xr.merge([F,data_kmeans])

	#save the xarray and upload it to s3
	Combined_Array.to_netcdf(path = "DB_Kmeans_compiled_Clustered_Data.nc4", compute = True)

	home = expanduser("~")

	with open(os.path.join(home,'creds.json')) as creds_file:
	    creds_data = json.load(creds_file)

	#Access from S3
	s3 = boto3.resource('s3',aws_access_key_id=creds_data['key_id'],
	         aws_secret_access_key=creds_data['key_access'],region_name='us-west-2')
	bucket = s3.Bucket('himatdata')
	home = os.getcwd()

	bucket.upload_file('DB_Kmeans_compiled_Clustered_Data.nc4','Trmm/EPO/Cluster_results_March5/DB_compiled_Clustered_Data.nc4')

	#remove the local copy
	os.remove('DB_Kmeans_compiled_Clustered_Data.nc4')

if __name__ == '__main__':
	nclusters = 10 #SET THIS AFTER LOOKING AT SILHOUETTE SCORE
    start_time = time.time()
    F = load_Data()
    klabels, CENTERS = kMeans_fit(F,nclusters)
    klabels_originalData = klabel_original_data(F,klabels)

    save_netCDF(F,klabels_originalData,CENTERS,klabels)

    #remove files locally
    os.remove('S3_downloads/*.nc4')

    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
