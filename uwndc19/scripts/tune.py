import os
import tensorflow as tf
import ray
import yaml
from ray import tune
from tensorflow.python.platform import tf_logging
from uwndc19.core.estimator import train
from uwndc19.core.builder_factory import create_builder
from uwndc19.core.utils import root_dir
from uwndc19.hp_tuning.mutations import get_mutations, mutate_config, generate_mutation_name


def tune_hyperparameters(model_type: str, experiment_name: str):
    ray.init(ignore_reinit_error=True)

    tuning_config_dir = root_dir('configs/%s/hp_tuning/%s' % (model_type, experiment_name))
    models_dir = root_dir('training/%s/hp_tuning/%s' % (model_type, experiment_name))

    # read the base config
    with open(os.path.join(tuning_config_dir, 'config.yaml')) as f:
        base_config = yaml.load(f)

    # read mutations config
    with open(os.path.join(tuning_config_dir, 'mutations.yaml')) as f:
        mutations_grid = yaml.load(f)

    # get mutated configs
    mutations = get_mutations(mutations_grid)

    def tune_fn(tune_config, reporter):
        model_builder = create_builder(model_type, tune_config['config'])
        train(model_builder, tune_config['model_dir'], reporter)

    configuration = tune.Experiment(
        experiment_name,
        run=tune_fn,
        local_dir=os.path.join(models_dir, 'ray_results'),
        config={
            'mutation': tune.grid_search(mutations),
            'config': tune.sample_from(lambda spec: mutate_config(base_config, spec.config.mutation)),
            'model_dir': tune.sample_from(lambda spec: os.path.join(models_dir, 'models',
                                                                    generate_mutation_name(spec.config.mutation))),
        },
        trial_name_creator=tune.function(lambda trial: generate_mutation_name(trial.config['mutation'])),
        resources_per_trial={
            'cpu': 1,
            # 'gpu': 0.2,
        },
    )

    tune.run_experiments(configuration, resume=True)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    model_type = 'multiclass'
    experiment_name = 'hp_tuning_2'

    tune_hyperparameters(model_type, experiment_name)


if __name__ == '__main__':
    main()
