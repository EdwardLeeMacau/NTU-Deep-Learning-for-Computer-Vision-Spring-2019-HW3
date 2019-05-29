# TODO: create shell script for running your GAN/ACGAN model
if ! [ -f "./DCGAN_generator.pth" ]; then
    wget -O ./DCGAN_generator.pth
fi

if ! [ -f "./ACGAN_generator.pth" ]; then
    wget -O ./ACGAN_generator.pth https://www.dropbox.com/s/4at71l9xxrpqrk8/acgan_generator.pth?dl=0
fi

if ! [ -d $1 ]; then
    mkdir $1
fi

python3 generate.py dcgan --model ./DCGAN_generator.pth --output $1
python3 generate.py acgan --model ./ACGAN_generator.pth --output $1
