import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from Dataset import dataset


d=2

def _plot_confusion_matrix(spacing, compartment_func, display_labels, TEST_ONLY = True):
    name = f'{["Full", "Test"][TEST_ONLY]}_{len(display_labels)}_compartments_{dataset.name}'

    if TEST_ONLY:
        dic = {index : value for index, value in spacing.items() if index in dataset.test_indices}
    else :
        dic = spacing

    carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')
    
    GT_spacing = [compartment_func(dic[index]['GT']) for index in dic]
    pred_spacing = [compartment_func(dic[index]['pred']-dataset.metrics.uncertainty(min(carinaNet_summary[index]['CARINA']['confidence'],carinaNet_summary[index]['ETT']['confidence']))) for index in dic]

    fig, ax = plt.subplots()
    try : 
        cmd = ConfusionMatrixDisplay.from_predictions(GT_spacing, pred_spacing,
                                                cmap = 'Blues', 
                                                ax = ax,
                                                display_labels = display_labels)
    except ValueError as e :
        if 'usually from a call to set_ticks, does not match the number of ticklabels' in str(e):
            cmd = ConfusionMatrixDisplay.from_predictions(GT_spacing, pred_spacing,
                                                    cmap = 'Blues', 
                                                    ax = ax)
        else : 
            raise
            
    cmd.ax_.set(xlabel = f'CarinaNet', ylabel = 'Ground Truth')
    ax.set_title(f'Confusion matrix on ETT-Carina distance on {len(GT_spacing)} predictions on {name} set')
    plt.show()
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}confusion_matrix_{name}.png')


def _compartment_1cm(x): #compartments are 0-1,1-2 ... 9-10, 10+
    return 0 if x<0 else (10 if x>10 else int(x))


def _compartment_pos(x): #compartments are 2-, 2+
    return x>d



def plot(spacing):
    """presenting results as made in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8082365/pdf/ryai.2020200026.pdf"""
    _plot_confusion_matrix(spacing, _compartment_pos, [f'<{d}', f'>{d}'], TEST_ONLY=False)
    _plot_confusion_matrix(spacing, _compartment_1cm, ['0','1','2','3','4','5','6','7','8','9','10+'], TEST_ONLY=False)
    _plot_confusion_matrix(spacing, _compartment_1cm, ['0','1','2','3','4','5','6','7','8','9','10+'], TEST_ONLY=True)
    
    _plot_confusion_matrix(spacing, _compartment_pos, [f'<{d}', f'>{d}'], TEST_ONLY=True)
