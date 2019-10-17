from sacred.observers import FileStorageObserver
from get_results import get_results
from main_loop import ex

ex.observers.append(FileStorageObserver.create('cifar10_attention_runs'))

try:
    get_results()
except Exception as e:
    print("There has been a problem loading some experiment...!")
    print(e)
    if input("Do you want to continue? ([y]/n)") != "y":
        import sys
        sys.exit()
ex.run()
get_results()

for position_embedding_version in ["rows_then_columns"]:
    for segment_embedding_version in ["rows_columns"]:
        get_results()
        ex.run(config_updates={"position_embedding_version": position_embedding_version,
                               "segment_embedding_version": segment_embedding_version,
                               }
               )
        get_results()

# for n_heads in [1, 2, 4, 7]:
#     for n_layers in [1, 2, 3, 4, 5]:
#         ex.run(config_updates={"n_layers": n_layers,
#                                "n_heads": n_heads,
#                                }
#                )
#         get_results()
