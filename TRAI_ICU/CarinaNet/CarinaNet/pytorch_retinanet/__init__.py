
import torch, sys, os


download_link = 'https://drive.google.com/file/d/1BePzPjqM4oMDDbPWS5Npe7khDuxRgzW1/view?usp=sharing'
model_name = 'model_final.pt'
path_to_model = os.path.join(os.path.dirname(__file__), model_name)

CUDA_AVAILABLE = torch.cuda.is_available()
print(f'CUDA available: {CUDA_AVAILABLE}')
sys.path.append(os.path.dirname(__file__))


print('\no Loading CarinaNet...')
try : 
    if CUDA_AVAILABLE:
        retinanet = torch.load(path_to_model)
    else:
        retinanet = torch.load(path_to_model, map_location=torch.device('cpu'))
except FileNotFoundError: 
    raise FileNotFoundError(f'Model weights not found at "{path_to_model}"\n\n'
                            f'Please download the weights from \n\n         {download_link}\n\n'
                            f'and save the file at \n\n{path_to_model}')
print('     -> Done.\n')

if torch.cuda.is_available():
    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
else:
    retinanet = retinanet.module.to(torch.device('cpu'))
    retinanet = torch.nn.DataParallel(retinanet)

retinanet.eval()
