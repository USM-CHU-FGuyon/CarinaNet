ETT-Carina Distance from ICU ChestXray images
====

This repository implements the ideas from **link to paper**. 

Download the codes
-

Clone this repository using the following command :
```
git clone https://github.com/USM-CHU-FGuyon/CarinaNet.git
```

download the trained weights from [Drive](https://drive.google.com/file/d/1BePzPjqM4oMDDbPWS5Npe7khDuxRgzW1/view?usp=sharing), and place them at ```TRAI_ICU/CarinaNet/CarinaNet/pytorch-retinaNet/model_final.pt```


Input images 
-
Images given in input should be chest X-ray radiographs of sufficient resolution on which the endotracheal tube should be visible.

Supported format are JPG and PNG.

Annotations Format
-
If annotations are available, they should be formatted as such in a json file : 
```
{
    "path/to/img1.png" : {
            "CARINA": [NaN, NaN],
            "ETT": [NaN, NaN],
            "qualite": NaN
          },
    "path/to/img2.png" :{
            "CARINA":[452,526],
            "ETT":[436, 560],
            "qualite":"poor"
          }
}
```
Any item that is not annoted should be specified as ```NaN```. ```qualite``` can also be specified as an ```int``` or a ```str``` to categorize several grades of radiograph quality.
This file should be saved under ```TRAI_ICU/Dataset/{your_dataset_name}/annotations.json```


Output
-
The model figures and intermediate results are saved in the ```outputs/{your_dataset_name}``` directory.


Running the code
- 

### Setting up the Dataset class

All the images should be stored in a single directory at path ```path/to/your/images```.

The ```TRAI_ICU/Dataset/__init__.py``` is initialized as such :

```
from Dataset import Dataset

dataset = Dataset.Dataset(name = 'your_dataset_name',
                          path_to_img = 'path/to/your/images/',
                          annoted = False, #or True if an annotation.json file was provided
                          path_to_pixel_spacing = 'path/to/pixel_spacing_file.csv' 
                          #or pixel_to_mm = 0.2 for constant pixel spacing 
                          )
```
### Specifying Pixel spacing
The pixel spacing (scale between pixel and millimeters) should be specified to output values in metric scale. It can be found in the metadata of Dicom files.

The pixel spacing can be set constant for the dataset by passing the ```pixel_to_mm``` argument to ```Dataset.Dataset``` as shown above.

If the pixel spacing is image-dependant, a csv file can be provided, and the path given in the ```path_to_pixel_spacing``` field.

The format should be 
````
a.png, 0.139
b.png, 0.2
c.png, 0.2
````
The first column should contain the names of the images and the second should contain the pixel-to-mm conversion constant.

### Running the model 

The ```main.py``` script runs the preprocessing, the CarinnaNet model and does the post processing if annotations were given.

```
python main.py
```


### Training the model

The RetinaNet model was taken from pytorch-retinanet on [github](https://github.com/yhenon/pytorch-retinanet), The information for re-training can be found in ```CarinaNet/CarinaNet/pytorch-retinanet/README.md```.





