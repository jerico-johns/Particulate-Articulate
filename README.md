# capstone
Particulate Prediction

Jerico Johns, David Djambazov, Srishti Mehra


Jerico Notes: 

Starting Cloud Instance and SSHing: 
—  In EC2 site, launch instance
- cd to wherever pem key is saved.. (cd Desktop)
- To SSH:  ssh -i "capstone.pem" ec2-user@ec2-3-135-241-27.us-east-2.compute.amazonaws.com

Download Training & Test Data (note Metadata.csvs are saved in capstone github repo) 
- To download training data: aws s3 cp s3://drivendata-competition-airathon-public-us/pm25/train/ train/ --no-sign-request --recursive
- To download test data: aws s3 cp s3://drivendata-competition-airathon-public-us/pm25/test/ test/ --no-sign-request --recursive

Jupyter Lab: 
- To install Jupyter Lab: pip install jupyterlab
- To run:  jupyter lab --allow-root --ip=*
- Then http://{public ec2 ip}:8888/lab?token=114231714cba0bae121a3fb588af12b29dcdff5034106869

To install gdal: 
- Create conda env: conda create --name snakes python=3.9
- Activate conda env (with python 3.9 and gdal): conda activate capstone 
- Install gdal: conda install gdal 