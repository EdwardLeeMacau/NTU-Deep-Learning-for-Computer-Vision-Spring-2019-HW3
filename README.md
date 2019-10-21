# HW3 ― GAN, ACGAN and UDA

## Demonstration

<table>
<caption><h3>GAN</h3></caption>
<tr>
</tr>
</table>

<table>
<caption><h3>ACGAN</h3></caption>
<tr>
</tr>
</table>

## Performance

### Target: mnistm
Namespace(alpha=0.25, batch_size=64, dataset=None, model='models/dann/20190504_025_batch16/DANN_usps_mnistm_30.pth', output='./output/dann/mnistm_pred.csv', target='mnistm', threads=8)
Model loaded from models/dann/20190504_025_batch16/DANN_usps_mnistm_30.pth
Source_test:    usps, 2007
Target_test:    mnistm, 10000
usps_Test:
[class_acc: 96.66% ] [class_loss: 0.1416] [domain_acc: 79.12 %] [domain_loss: 0.5516]
mnistm_Train:
[class_acc: 35.24% ] [class_loss: 2.8489] [domain_acc: 39.43 %] [domain_loss: 0.8002]
mnistm_Test:
[class_acc: 35.80% ] [class_loss: 2.8391] [domain_acc: 39.42 %] [domain_loss: 0.8069]

### Target: svhn
Namespace(alpha=0.25, batch_size=64, dataset=None, model='models/dann/20190504_025_batch16/DANN_mnistm_svhn_15.pth', output='./output/dann/svhn_pred.csv', target='svhn', threads=8)
Model loaded from models/dann/20190504_025_batch16/DANN_mnistm_svhn_15.pth
Source_test:    mnistm, 10000
Target_test:    svhn, 26032
mnistm_Test:
[class_acc: 97.74% ] [class_loss: 0.0673] [domain_acc: 99.75 %] [domain_loss: 0.0237]
svhn_Train:
[class_acc: 44.31% ] [class_loss: 1.7580] [domain_acc: 0.30 %] [domain_loss: 3.6742]
svhn_Test:
[class_acc: 49.95% ] [class_loss: 1.5604] [domain_acc: 0.13 %] [domain_loss: 3.5836]

### Target: usps
Namespace(alpha=0.25, batch_size=64, dataset=None, model='models/dann/20190504_025_batch16/DANN_svhn_usps_18.pth', output='./output/dann/usps_pred.csv', target='usps', threads=8)
Model loaded from models/dann/20190504_025_batch16/DANN_svhn_usps_18.pth
Source_test:    svhn, 26032
Target_test:    usps, 2007
svhn_Test:
[class_acc: 82.77% ] [class_loss: 0.5907] [domain_acc: 1.04 %] [domain_loss: 0.8784]
usps_Train:
[class_acc: 70.74% ] [class_loss: 0.9812] [domain_acc: 74.49 %] [domain_loss: 0.6583]
usps_Test:
[class_acc: 67.16% ] [class_loss: 1.1177] [domain_acc: 74.49 %] [domain_loss: 0.6577]

## How to use

1. Download dataset

    ```
    bash get_dataset.sh
    ```

2. Download pretrained model and inference for Problem 1 and Problem 2

    ```
    bash hw3_p1p2.sh
    ```

3. Download pretrained model and inference for Problem 3

    ```
    bash hw3_p3.sh
    ```

3. Download pretrained model and inference for Problem 4

    ```
    bash hw3_4.sh
    ```

## Requirements

## More Information

Please read [requirement](./REQUIREMENT.md) to get more information about this HW.

## Performance Report

### Problem 1 GAN

### Problem 2 ACGAN

### Problem 3 DANN

### Problem 4 Improved DANN
