import optuna
from transform_event_log import run


def objective_gnn(trial, train_loader, test_loader, dataset):
    gnn_param = {
        'hidden_channels': trial.suggest_int("hidden_channels", 128, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1e-1),
        'dropout_rate': trial.suggest_uniform('dropout_rate', 0.2, 0.8)
    }

    model, train_acc, test_acc = run(train_loader, test_loader, dataset, gnn_param)

    return test_acc[-1].item()


def get_best_gnn_param(train_loader, test_loader, dataset):
    '''
    returns best hyperparameter for GNN
    :param train_loader:
    :param test_loader:
    :param dataset:
    :return: best hyperparameter
    '''
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective_gnn(trial, train_loader, test_loader, dataset), n_trials=10)
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print(f'{key}: {value}')

    return best_trial.params