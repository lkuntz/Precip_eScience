## TRMM Data Retrieval

This repo consists of file `data_fetch.py` which grabs data from [TRMM](http://trmm.atmos.washington.edu) website and upload to s3 bucket for future usage.

## Usage

Before executing the `data_fetch.py` initial setup requires setting up requisite credential file for AWS S3.

Provide one time `key_id` and `key_access` to be stored in the `home` folder as `creds.json` which will be used by `data_fetch.py`.
```
import os
from os.path import expanduser
import json

home = expanduser("~")
# store one time credentials in the home directory
creds = {'key_id' : '',
         'key_access' : ''}
with open(os.path.join(home,'creds.json'), 'a') as cred:
    json.dump(creds, cred)
```

Post setting up credentials next steps involves :

Class `TRMM_data()` needs to be initialized with:
  
      1.) Providing authentication for TRMM website: self.auth_data = ('','')
      2.) Providing region to download data for e.g- 'EPO': self.output_folder= 'EPO/'
      
## Runtime

For `EPO` it takes ~5 hours on a c5.4x large machine multiprocessed over 15 CPU's.
