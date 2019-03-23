"""A tool for retrieving EC2 instances and terminating them"""

import os
from os.path import expanduser
import json
import boto3
import time
from multiple_instance import Multi_instance
import logging

logging.basicConfig(filename='cleanup.log', level=logging.INFO)


def load_creds():
    """
        Utility function to read s3 credential file for
        data upload to s3 bucket.
    """
    home = expanduser("~")
    with open(os.path.join(home, 'creds_multi.json')) as creds_file:
        creds_data = json.load(creds_file)
    return creds_data

def main(instance_tag_name, region):
    creds_data = load_creds()
    session = boto3.Session(aws_access_key_id=creds_data['key_id'],aws_secret_access_key=creds_data['key_access'])
    # Connect to EC2
    ec2 = session.resource('ec2', region_name=region)

    # Get information for all pending/running instances
    running_instances = ec2.instances.filter(Filters=[{
        'Name': 'instance-state-name',
        'Values': ['pending','running']}])
    instance_id = []
    for instance in running_instances:
        for tag in instance.tags:
            if tag['Value'].startswith(instance_tag_name):
                name = tag['Value']
                instance_id.append(instance.id)
    for instance_kill in instance_id:
        ec2.instances.filter(InstanceIds=[instance_kill]).terminate()
        logging.info("Terminated instace ID: %s", instance_kill)
    logging.info("Number of terminated instances: %s", len(instance_id))

if __name__ == '__main__':
    dummy_year = 1990
    dummy_manager = []
    cleanup_job = Multi_instance(dummy_year, dummy_manager)
    time_sleep = cleanup_job.CLEANUP_TIME
    logging.info("Waiting for %d secs", time_sleep)
    time.sleep(time_sleep)
    instance_tag_name = cleanup_job.TAG_NAME['Value'].split(cleanup_job.TAG_DELIMITER)[0]
    logging.info("Tag being cleaned up is : %s", instance_tag_name)
    main(instance_tag_name, cleanup_job.REGION)
