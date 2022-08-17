from .full_epoch_training import classic_train_epoch, alternate_ordinary_train_epoch, alternate_fixing_train_epoch

list_training_modules = {
    "classic": classic_train_epoch,
    "alternate_ordinary": alternate_ordinary_train_epoch,
    "alternate_fixing": alternate_fixing_train_epoch,
}