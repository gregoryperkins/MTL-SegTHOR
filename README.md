# MTL-SegTHOR
Dependent Multi-Task Learning for the Segmentation of Thoracic Organs at Risk in CT Images

author: Tao He

Institution: Sichuan University

email: taohe@stu.scu.edu.cn

Tookit need 
Python 3, pytorch 1.1.0

### Prepare SegTHOR dataset
> reading preprocessing.ipynb in details

the source data download from https://ent.normandie-univ.fr/filex/get?k=oZgYIeT5lnbxhtHZ2u8 \
or  download from my Baidu Netdisk https://pan.baidu.com/s/1dQHYKIkUd5qCXIvdxSijNg; password: i41q \
The data path organized like: \
../data/data_source/Patient_01/GT.nii \
../data/data_source/Patient_01/Patient_01.nii
Then, using preprocessing.ipynb to process the SegTHOR dataset for 4-fold cross-validation

### Network training
> See main.py parser.argument in details

A test run is  like this \
python3 main.py -b 16 --gpu 0,1,2,3 --model_name ResUNet101 --save_dir SavePath/SM --lr 0.01 --if_dependent 0 --if_closs 0

#### Important parameters:
- model_name -> indicate which encoder network is used. Please check models/model_loader.py in details.
- if_closs -> if using the classification loss (MTL). if_closs=0, MTL is not deployed.
- if_dependent -> if using the WMCE loss function. if_dependent=0, binary relevance with BCE loss functions is used.

### Network testing
> See prediction.ipynb in details
 
**Please don't hesitate to contact me if you have any question about the data, method, or code !**
