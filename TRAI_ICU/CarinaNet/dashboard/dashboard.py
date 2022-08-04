from Dataset import dataset
from . import  roc_confidence,  position_annot_reader, error_correlation
from . import confusion_matrix, error_confidence, quality_error, human_confusion_matrix


# def err_and_quality(err_dic):
#
#     err_dic_test = {index : val for index, val in err_dic['dist'].items() if index in dataset.test_indices}
#
#     data = {lab: [] for lab in dataset.annot.quality_label.values()}
#
#     for index, err in err_dic_test.items():
#         data[dataset.annot.quality(index)].append(err)


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

    