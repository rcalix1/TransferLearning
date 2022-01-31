from datasets import list_datasets, list_metrics
all_d = list_datasets()
metrics = list_metrics()

print("# of datasets", len(all_d))
print("# of metrics", len(metrics))

print(all_d[:20])
print(metrics)
