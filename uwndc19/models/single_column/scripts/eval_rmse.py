import os
from tensorflow.contrib.estimator import read_eval_metrics
from uwndc19.core.utils import root_dir


def main(use_best_value: bool):
    models_dir = root_dir('training/single_column/models3')
    model_subdirs = os.listdir(models_dir)

    models_rmses = []
    for subdir in model_subdirs:
        if subdir.startswith('.'):
            continue

        # read the model summaries
        eval_dir = os.path.join(models_dir, subdir, 'eval')
        eval_results = read_eval_metrics(eval_dir)

        if use_best_value:
            # get the best RMSE value
            rmse = None
            for step, metrics in eval_results.items():
                val = metrics['rmse']
                if (rmse is None) or (val < rmse):
                    rmse = val
        else:
            # get the latest RMSE value
            rmse = eval_results[next(reversed(eval_results))]['rmse']

        models_rmses.append(rmse)

    print('Mean RMSE: %.04f' % (sum(models_rmses) / len(models_rmses)))


if __name__ == '__main__':
    main(use_best_value=True)
