import multiprocessing
from multiprocessing import Pool
import itertools
import time
import boto3
import paramiko
import sys
import os
from os.path import expanduser
import json
from multiprocessing.managers import BaseManager, SyncManager
import signal
import subprocess

#handle SIGINT from SyncManager object
def mgr_sig_handler(signal, frame):
    print('Not closing the manager')

#initilizer for SyncManager
def mgr_init():
    signal.signal(signal.SIGINT, mgr_sig_handler)
    #signal.signal(signal.SIGINT, signal.SIG_IGN) # <- OR do this to just ignore the signal
    print('Initialized Mananger')

def load_creds():
        """
            Utility function to read s3 credential file for
            data upload to s3 bucket.
        """
        home = expanduser("~")
        with open(os.path.join(home, 'creds_multi.json')) as creds_file:
            creds_data = json.load(creds_file)
        return creds_data


class Multi_instance(object):
    def __init__(self,year, shared_list):
        self.CMD_0 = "source /home/ubuntu/miniconda3/bin/activate precip_test"
        self.CMD_1 = "wget -O /home/ubuntu/precip/Precip_eScience/clusterEvents.py https://raw.githubusercontent.com/lkuntz/Precip_eScience/master/clusterEvents.py"
        self.CMD_2 = "/home/ubuntu/miniconda3/bin/python /home/ubuntu/precip/Precip_eScience/clusterEvents.py -y {}".format(year)
        self.KEY = paramiko.RSAKey.from_private_key_file('winter19_incubator.pem')
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.AMI = ""
        self.MIN_COUNT = 1
        self.MAX_COUNT = 1
        self.INSTANCE_TYPE = "t2.micro"
        self.SECURITY_GROUP = []
        self.KEY_NAME = 'winter19_incubator'
        self.TAG_DELIMITER = '-'
        self.TAG_NAME = {"Key": "Name", "Value": 'Shiv_Incubator19{}{}'.format(self.TAG_DELIMITER,year)}
        self.REGION = "us-west-2"
        self.CREDS_DATA = {}
        self.CLEANUP_TIME = 30

    def load_creds(self):
        """
            Utility function to read s3 credential file for
            data upload to s3 bucket.
        """
        home = expanduser("~")
        with open(os.path.join(home, 'creds_multi.json')) as creds_file:
            self.CREDS_DATA = json.load(creds_file)
        
    def spin_instance(self):
        session = boto3.Session(aws_access_key_id=self.CREDS_DATA['key_id'],aws_secret_access_key=self.CREDS_DATA['key_access'])
        ec2 = session.resource('ec2',region_name=self.REGION)
        instances = ec2.create_instances(ImageId= self.AMI, MinCount= self.MIN_COUNT, MaxCount= self.MAX_COUNT,
                                         InstanceType= self.INSTANCE_TYPE, SecurityGroupIds=self.SECURITY_GROUP,
                                         KeyName= self.KEY_NAME,
                                         TagSpecifications=[{'ResourceType': 'instance','Tags': [self.TAG_NAME]}])
        instance = instances[0]
        shared_list.append(instance.id)
        try:
            ec2_client = session.client('ec2',region_name=self.REGION)
            waiter = ec2_client.get_waiter('instance_status_ok')
            waiter.wait(InstanceIds=[instance.id])
            print("The instance now has a status of 'ok'!")
            instance.load()
            self.client.connect(hostname=instance.public_dns_name, username="ubuntu", pkey=self.KEY)
            cmd = [self.CMD_0, self.CMD_1, self.CMD_2]
            channel = self.client.invoke_shell()
            for command in cmd:
                stdin, stdout, stderr = self.client.exec_command(command)
                exit_status = stdout.channel.recv_exit_status()          # Blocking call
                if exit_status == 0:
                   print ("Done Executing: ", command)
                else:
                   print("Stdout output is: ", stdout.read())
                   print("Error occured is: ", stderr.read())

            print("Executed all of the commands. Now will exit \n")
            self.client.close()
            ec2.instances.filter(InstanceIds=[instance.id]).terminate()
        except:
            instance = instances[0]
            ec2.instances.filter(InstanceIds=[instance.id]).terminate()
        
def _multiprocess_handler(year):
    batch_job = Multi_instance(int(year[0]), year[1])
    batch_job.load_creds()
    batch_job.spin_instance()
        
if __name__ == '__main__':
    start_time = time.time()
    manager = SyncManager()
    manager.start(mgr_init)  #fire up the child manager process
    shared_list = manager.list()
    years = [(str(i).zfill(4), shared_list) for i in range(1998, 2014)]
    process = 8
    process = multiprocessing.cpu_count() if process > multiprocessing.cpu_count() else process
    chunks = len(years)//process if len(years) > process else 1
    pool = Pool(process)
    try:
       pool.map(_multiprocess_handler, years, chunksize=chunks)
    except KeyboardInterrupt:
        print("Keyboard Interrupt will now call cleanup script") 
    print("calling cleanup script")
    subprocess.Popen("python cleanup_worker.py", shell=True)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))
