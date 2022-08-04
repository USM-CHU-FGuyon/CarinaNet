import matplotlib.pyplot as plt
from Dataset import dataset

def plot(loc_summary, el = 'element', test_only = True):
    """Shows the correlation between the algorithm confidence and the error."""
    if test_only :
        loc_summary = {index : val for index, val in loc_summary.items() if index in dataset.test_indices}
    fig, ax = plt.subplots()
    
    ax.set_xlabel('Error (cm)')
    ax.set_ylabel('Confidence')
    
    
    errors = {index : summ['err'] for index, summ in loc_summary.items()}
    confidences = {index: summ['confidence'] for index, summ in loc_summary.items()}
    
    ax.scatter(errors.values(), confidences.values(), alpha = 0.2)
    ax.set_title(f'Correlation between confidence and error in {el} localization'+test_only*'on test only')
    plt.show()
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}_err_confidence_correlation_{el}.png')
    