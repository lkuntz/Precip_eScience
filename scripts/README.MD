## TRMM Data Clustering multi instances

This repo consists of file `multiple_instance.py` which uses AMI's to spin up multiple clusters and run clustering script ```clusterEvents.py``` and terminates on completion.
## Usage

Before executing the `data_fetch.py` initial setup requires setting up requisite credential file for AWS S3.

Provide one time `key_id` and `key_access` to be stored in the `home` folder as `creds_multi.json` which will be used by `multiple_instance.py`.
```
import os
from os.path import expanduser
import json

home = expanduser("~")
# store one time credentials in the home directory
creds = {'key_id' : '',
         'key_access' : ''}
with open(os.path.join(home,'creds_multi.json'), 'a') as cred:
    json.dump(creds, cred)
```

Post setting up credentials next steps involves :

Class `Multi_instance()` needs to be initialized with:
  
      1.) Providing AMI id to spin instances: self.AMI = ""
      2.) Providing Security group for the AMI to attach to : self.SECURITY_GROUP = ['']
      3.) Providing path to the .pem file for paramiko to do ssh : self.KEY = paramiko.RSAKey.from_private_key_file('sample.pem')
      
## Runtime

For `EPO` it takes ~12 hours on a c5.4x large machine scaled over 8 instances.

## Things to Know

While using Paramiko library to ssh and send commands, since it submits the job to linux kernel itself its necessary to provide **absolute paths** instead of **relative paths** for any command's. 
Ex:- 
```
self.CMD_0 = "source /home/ubuntu/miniconda3/bin/activate precip_test"
self.CMD_1 = "/home/ubuntu/miniconda3/bin/python /home/ubuntu/precip/Precip_eScience/clusterEvents.py -y {}".format(year)
```
Also this needs to have any references in the script to be run to be also **absolute**. For this setting the current directory in the file itself helps fixing any issues relating to paths.
Ex:
```
ROOT_DIR = '/home/ubuntu/precip/Precip_eScience/'
os.chdir(ROOT_DIR)
```
Sample example of setting current working directory can been seen in the [clusterEvents.py](https://github.com/lkuntz/Precip_eScience/blob/master/clusterEvents.py)
