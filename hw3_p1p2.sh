# TODO: create shell script for running your GAN/ACGAN model
if ! [ -f "./DCGAN_generator.pth" ]; then
    wget -O ./DCGAN_generator.pth
fi

if ! [ -f "./ACGAN_generator.pth" ]; then
    wget -O ./ACGAN_generator.pth 
fi

if ! [ -d $1 ]; then
    mkdir $1
fi

python generate.py dcgan --model ./DCGAN_generator.pth --output $1
python generate.py acgan --model ./ACGAN_generator.pth --output $1
