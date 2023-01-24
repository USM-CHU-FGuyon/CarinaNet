from Dataset import dataset
from . import  roc_confidence,  position_annot_reader, error_correlation
from . import confusion_matrix, error_confidence, quality_error, human_confusion_matrix


def main(carinaNet_summary, err_dic, spacing):
    
    err_dic_test = {key :  {i: v for i, v in err_dic[key].items() if i in dataset.test_indices} for key in err_dic}
    
    spacing_test = {key : val for key, val in spacing.items() if key in dataset.test_indices}

    confusion_matrix.plot(spacing_test)    

    position_annot_reader.plot(spacing)

    roc_confidence.plot_roc(carinaNet_summary, err_dic)

    human_confusion_matrix.plot(spacing)

    error_correlation.plot(err_dic, test_only = True, annotate_large_err = True)
    error_correlation.plot(err_dic, test_only = False)

    dataset.dashboard.error_histogram(err_dic['dist'],'CarinaNet', unit = 'cm')

    