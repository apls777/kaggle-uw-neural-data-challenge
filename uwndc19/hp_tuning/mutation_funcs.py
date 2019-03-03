def toggle_batch_norm(config: dict, layer_id: int):
    batch_norm = config['model']['conv_layers'][layer_id].get('batch_norm', False)
    config['model']['conv_layers'][layer_id]['batch_norm'] = not batch_norm


def add_conv_layer(config: dict, num_filters: int, kernel_size: int, batch_norm: bool = False):
    config['model']['conv_layers'].append({
        'num_filters': num_filters,
        'kernel_size': kernel_size,
        'padding': 'same',
    })


def delete_last_conv_layer(config: dict):
    config['model']['conv_layers'] = config['model']['conv_layers'][:-1]


def change_conv_layer(config: dict, layer_id: int, num_filters: int = None, kernel_size: int = None,
                      batch_norm: bool = None):
    if num_filters is not None:
        config['model']['conv_layers'][layer_id]['num_filters'] = num_filters

    if kernel_size is not None:
        config['model']['conv_layers'][layer_id]['kernel_size'] = kernel_size

    if batch_norm is not None:
        config['model']['conv_layers'][layer_id]['batch_norm'] = batch_norm


def add_dense_layer(config: dict, num_units: int, dropout_rate: float = 0.0, l2_regularization: float = 0.0):
    config['model']['dense_layers'].append({
        'num_units': num_units,
        'dropout_rate': dropout_rate,
        'l2_regularization': l2_regularization,
    })


def change_dense_layer(config: dict, layer_id: int, num_units: int = None, dropout_rate: float = None,
                       l2_regularization: float = None):
    if num_units is not None:
        config['model']['dense_layers'][layer_id]['num_units'] = num_units

    if dropout_rate is not None:
        config['model']['dense_layers'][layer_id]['dropout_rate'] = dropout_rate

    if l2_regularization is not None:
        config['model']['dense_layers'][layer_id]['l2_regularization'] = l2_regularization


def change_logits_layer(config: dict, dropout_rate: float = None):
    if dropout_rate is not None:
        config['model']['logits_dropout_rate'] = dropout_rate
