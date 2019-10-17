from sacred.observers import FileStorageObserver
from get_results import get_results
from main_loop import ex

ex.observers.append(FileStorageObserver.create('fashionmnist_attention_runs'))

for n_heads in [1, 2, 4, 7]:
    for n_layers in [1, 2, 3, 4, 5]:
        ex.run(config_updates={"n_layers": n_layers,
                               "n_heads": n_heads,
                               }
               )
        get_results()
