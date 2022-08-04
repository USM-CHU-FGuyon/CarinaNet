
from Dataset import Dataset

dataset = Dataset.Dataset(name= 'test_images',
                          path_to_img = '../data/example/images/',
                          annoted = False, #set to True to use the annotations in Dataset/{name}/annotations.json
                          path_to_pixel_spacing='../data/example/pixel_spacing.csv'
                          )
