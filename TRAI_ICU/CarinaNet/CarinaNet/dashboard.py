from Dataset import dataset


def main():
    if dataset.INFERENCE_MODE:
        return
    carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')
    try :
        
        
        err_dic_carina = {index: val['CARINA']['err'] for index, val in carinaNet_summary.items()}
        err_dic_probe = {index: val['ETT']['err'] for index, val in carinaNet_summary.items()}
    
        dataset.dashboard.error_histogram(err_dic_carina, 'CarinaNet', bins=50, unit='cm', note = '_carinaNet')
        dataset.dashboard.error_histogram(err_dic_probe, 'CarinaNet', bins=50, unit='cm', note = '_probeNet')
    except KeyError as e :
        print('\n\n /!\ Failed to make dashboard\nError : ',e)
