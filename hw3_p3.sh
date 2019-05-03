# TODO: create shell script for running your DANN model

# Download the model
wget -O ./DANN_usps_mnistm.pth
wget -O ./DANN_mnistm_svhn.pth
wget -O ./DANN_svhn_usps.pth

python predict.py --model $2 --dataset $1 --output $3
