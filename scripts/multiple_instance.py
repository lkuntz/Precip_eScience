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
    def __init__(self, year, shared_list, instance_type = 'spot'):
        self.CMD_0 = "source /home/ubuntu/miniconda3/bin/activate precip_test"
        self.CMD_DASK = "/home/ubuntu/miniconda3/bin/conda install -y dask"
        self.CMD_DISTRIBUTE = "/home/ubuntu/miniconda3/bin/conda install -y dask distributed"
        self.CMD_1 = "wget -O /home/ubuntu/precip/Precip_eScience/clusterEvents.py https://raw.githubusercontent.com/lkuntz/Precip_eScience/master/clusterEvents.py"
        self.CMD_2 = "/home/ubuntu/miniconda3/bin/python /home/ubuntu/precip/Precip_eScience/clusterEvents.py -y {}".format(year)
        self.KEY = paramiko.RSAKey.from_private_key_file('winter19_incubator.pem')
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.AMI = ""
        self.MIN_COUNT = 1
        self.MAX_COUNT = 1
        self.INSTANCE_TYPE = ""
        self.SECURITY_GROUP = []
        self.SECURITY_GROUP_NAME = ['']
        self.KEY_NAME = 'winter19_incubator'
        self.TAG_DELIMITER = '-'
        self.TAG_NAME = {"Key": "Name", "Value": 'Shiv_Incubator19{}{}'.format(self.TAG_DELIMITER,year)}
        self.REGION = "us-west-2"
        self.CREDS_DATA = {}
        self.CLEANUP_TIME = 30
        self.instance_type = instance_type.lower()
        self.SPOT_PRICE = '0.95'
        self.AVAILABILITYZONE = "us-west-2a"
        self.YEAR = year
        self.SPINNED_INSTANCE = None

    def instance_type_check(self):
        valid_instance = {'spot', 'reserved'}
        if self.instance_type not in valid:
            raise ValueError("results: Instance must be one of %r." % valid_instance)

    def load_creds(self):
        """
            Utility function to read s3 credential file for
            data upload to s3 bucket.
        """
        home = expanduser("~")
        with open(os.path.join(home, 'creds_multi.json')) as creds_file:
            self.CREDS_DATA = json.load(creds_file)

    def spin_instance(self):
        self.SESSION = boto3.Session(aws_access_key_id=self.CREDS_DATA['key_id'],aws_secret_access_key=self.CREDS_DATA['key_access'])
        self.ec2 = self.SESSION.resource('ec2',region_name=self.REGION)
        if self.instance_type == 'reserved':
            instances = ec2.create_instances(ImageId= self.AMI, MinCount= self.MIN_COUNT, MaxCount= self.MAX_COUNT,
                                            InstanceType= self.INSTANCE_TYPE, SecurityGroupIds=self.SECURITY_GROUP,
                                            KeyName= self.KEY_NAME,
                                            TagSpecifications=[{'ResourceType': 'instance','Tags': [self.TAG_NAME]}])
            self.SPINNED_INSTANCE = instances[0]
            shared_list.append(self.SPINNED_INSTANCE.id)
        elif self.instance_type == 'spot':
            try:
                client = boto3.client('ec2', aws_access_key_id=self.CREDS_DATA['key_id'], aws_secret_access_key=self.CREDS_DATA['key_access'], region_name= self.REGION)
                response = client.request_spot_instances(
                        DryRun=False,
                        SpotPrice=self.SPOT_PRICE,
                        InstanceCount=1,
                        Type='one-time',
                        LaunchSpecification={
                                'ImageId': self.AMI,
                                'KeyName': self.KEY_NAME,
                                'SecurityGroups': self.SECURITY_GROUP_NAME,
                                'InstanceType': self.INSTANCE_TYPE,
                                'Placement': {
                                    'AvailabilityZone': self.AVAILABILITYZONE,
                                },

                       }
                     )
                request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
                print("Waiting for the spot request to be fulfilled !!!")
                waiter = client.get_waiter('spot_instance_request_fulfilled')
                waiter.wait(SpotInstanceRequestIds=[request_id])
                get_response = client.describe_spot_instance_requests(SpotInstanceRequestIds=[request_id])
                mytags = [self.TAG_NAME]
                tag = client.create_tags(
                    Resources = [get_response['SpotInstanceRequests'][0]["InstanceId"]],
                    Tags= mytags
                )
                ec2 = boto3.resource('ec2', aws_access_key_id=self.CREDS_DATA['key_id'], aws_secret_access_key=self.CREDS_DATA['key_access'],region_name= self.REGION)
                self.SPINNED_INSTANCE = ec2.Instance(get_response['SpotInstanceRequests'][0]["InstanceId"])
                shared_list.append(self.SPINNED_INSTANCE.id)
            except Exception as err:
               print('Following month {} run failed, error message:'.format(self.YEAR), err)


    def run_commands(self):
        try:
            ec2_client = self.SESSION.client('ec2',region_name=self.REGION)
            waiter = ec2_client.get_waiter('instance_status_ok')
            waiter.wait(InstanceIds=[self.SPINNED_INSTANCE.id])
            print("The instance now has a status of 'ok'!")
            self.SPINNED_INSTANCE.load()
            self.client.connect(hostname=self.SPINNED_INSTANCE.public_dns_name, username="ubuntu", pkey=self.KEY)
            cmd = [self.CMD_0, self.CMD_DASK, self.CMD_DISTRIBUTE, self.CMD_1, self.CMD_2]
            channel = self.client.invoke_shell()
            for command in cmd:
                print(command)
                stdin, stdout, stderr = self.client.exec_command(command)
                exit_status = stdout.channel.recv_exit_status()          # Blocking call
                if exit_status == 0:
                   print ("Done Executing: ", command)
                else:
                   print("Stdout output is: ", stdout.read())
                   print("Error occured is: ", stderr.read())

            print("Executed all of the commands. Now will exit \n")
            self.client.close()
            self.ec2.instances.filter(InstanceIds=[self.SPINNED_INSTANCE.id]).terminate()
        except Exception as err:
            print('Following month {} run failed, error message:'.format(self.YEAR), err)
            self.ec2.instances.filter(InstanceIds=[self.SPINNED_INSTANCE.id]).terminate()


def _multiprocess_handler(year):
    batch_job = Multi_instance(int(year[0]), year[1])
    batch_job.load_creds()
    batch_job.spin_instance()
    batch_job.run_commands()

if __name__ == '__main__':
    start_time = time.time()
    manager = SyncManager()
    manager.start(mgr_init)  #fire up the child manager process
    shared_list = manager.list()
    years = [(str(i).zfill(4), shared_list) for i in range(1998, 2000)]
    process = 2
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
