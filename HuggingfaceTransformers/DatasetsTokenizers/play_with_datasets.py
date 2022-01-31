## >> pip install datasets

from datasets import load_dataset
cola = load_dataset('glue', 'cola')

print(    cola['train'][25:28]  )

print(cola)

###############################

mrpc = load_dataset('glue', 'mrpc')
print(mrpc)
