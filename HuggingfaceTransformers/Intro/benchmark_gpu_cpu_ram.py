import torch

print("GPU memory in GBs is ")
print(torch.cuda.get_device_properties(0).total_memory / (1024**3))

########################################

from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

models = ["bert-base-uncased", "distilbert-base-uncased", "distilroberta-base", "distilbert-base-german-cased"]
batch_sizes = [4]
sequence_lengths=[32, 64, 128, 256, 512]
args = PyTorchBenchmarkArguments(models=models, batch_sizes=batch_sizes, sequence_lengths=sequence_lengths, multi_process=False)

benchmark = PyTorchBenchmark(args)
results = benchmark.run()
