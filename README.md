# Deep Networks Federated Learning (PyTorch)

Experiments are conducted using federated learning with deep neural networks on multiple datasets (e.g., MNIST, CIFAR10). The project explores novel approaches to federated training with customizable models and data distributions (IID and non-IID).

## Requirements

- Python
- PyTorch
- torchvision

## Data

Datasets such as MNIST, CIFAR10 are automatically downloaded via torchvision if not present locally.

To use your own dataset: place it in the `data/` directory and implement a PyTorch Dataset wrapper.

## Running the experiments

Baseline experiment: trains the model centrally (conventional training).

```python src/baseline_main.py --model=cnn --dataset=mnist --epochs=10```

To run on GPU (e.g., GPU 0):

```python src/baseline_main.py --model=cnn --dataset=mnist --gpu=0 --epochs=10```

Federated experiment: trains a global model from local client updates.

IID data:

```python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10```

Non-IID data:

```python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10```


## Running the experiments

Baseline experiment: trains the model centrally (conventional training).


Other parameters can be adjusted in the options section.

## Options

Default parameters can be found and changed in `options.py`, including:

- `--dataset`: Dataset to use (default: `mnist`, options: `mnist`, `fmnist`, `cifar`)
- `--model`: Model type (default: `cnn`, options: `mlp`, `cnn`)
- `--gpu`: GPU id to use (default: `None` for CPU)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 0.01)
- `--seed`: Random seed (default: 1)
- `--verbose`: Enable detailed logs (default: 1)

## Federated Learning Parameters

- `--iid`: Data distribution among clients (1 for IID, 0 for non-IID)
- `--num_users`: Total number of clients (default: 100)
- `--frac`: Fraction of clients sampled per round (default: 0.1)
- `--local_ep`: Number of local epochs per client update (default: 10)
- `--local_bs`: Batch size in local training (default: 10)
- `--unequal`: Split data equally (0) or unequally (1) in non-IID setting

## Results

The federated approach achieves comparable performance to centralized training on multiple datasets, demonstrating effective privacy-preserving collaborative learning. Model accuracy varies with data distribution and training configurations.

| Model | Dataset | IID Accuracy | Non-IID Accuracy |
|-------|---------|--------------|------------------|
| CNN   | MNIST   | 98.5%        | 85.3%            |
| MLP   | CIFAR10 | 78.9%        | 65.4%            |

## Contributing

Contributions and suggestions are welcome. Please fork the repo and open a pull request with your changes.

## License

MIT