from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


def plot_validation_curve(x_train, y_train, x_val, y_val, classifier_class, param_name, param_range, title):
    train_scores = []
    val_scores = []
    for param in param_range:
        classifier = classifier_class(**{param_name: param})
        classifier.fit(x_train, y_train)
        train_prediction = classifier.predict(x_train)
        val_prediction = classifier.predict(x_val)
        # calculate f1 score
        train_score = f1_score(y_train, train_prediction, average='macro')
        val_score = f1_score(y_val, val_prediction, average='macro')
        train_scores.append(train_score)
        val_scores.append(val_score)
    plt.subplots_adjust(bottom=0.55)
    plt.rcParams.update({'font.size': 11})

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xscale('log', base=10)
    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Parameter C')
    ax.set_title(title)
    ax.set_ylim([0, 1])
    ax.plot(param_range, train_scores, linestyle='--', marker='o', color='r', label='Training score')
    ax.plot(param_range, val_scores, linestyle='--', marker='x', color='g', label='Validation score')
    ax.legend()
    plt.tight_layout()
    # save figure
    fig.savefig(f"graphics/validation_curve_{title}_big.pdf", dpi=300)
