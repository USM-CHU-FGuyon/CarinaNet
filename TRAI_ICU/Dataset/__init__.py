from Dataset import Dataset

#pth='E:/data/'
pth='Y:/'


# =============================================================================
# dataset = Dataset.Dataset(name= 'MIMIC_CXR_test',
#                           path_to_img = f'{pth}MIMIC-CXR/TEST_SET/mimic_CXR_test/',
#                           annoted = True, #set to True to use the annotations in Dataset/{name}/annotations.json
#                           path_to_pixel_spacing=f'{pth}MIMIC-CXR/TEST_SET/pixel_spacing.csv'
#                           )
# 
# =============================================================================


dataset = Dataset.Dataset(name= 'TRAI_ICU',
                          path_to_img = f'{pth}TRAI-ICU/images',
                          annoted = True, #set to True to use the annotations in Dataset/{name}/annotations.json
                          xls_annot_path = f'{pth}TRAI-ICU/data/TRAI_ICU_v3.xlsx',
                          pixel_to_mm=0.2,
                          INFERENCE_MODE=True
                          )

