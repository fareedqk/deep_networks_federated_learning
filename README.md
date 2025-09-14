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

~~~
python scripts/baseline_main.py --model=cnn --dataset=mnist --epochs=10
~~~

To run on GPU (e.g., GPU 0):

~~~
python scripts/baseline_main.py --model=cnn --dataset=mnist --gpu=0 --epochs=10
~~~

Federated experiment: trains a global model from local client updates.

IID data:

~~~
python scripts/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
~~~

Non-IID data:

~~~
python scripts/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
~~~


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

## Results on MNIST

Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters:

```Optimizer```: SGD\
```Learning Rate```: 0.01

Table 1: Test accuracy after training for 10 epochs:

| Model | Test Acc| 
|-------|---------|
| MLP   | 91.7    | 
| MLP   | 95.3    | 

Federated Experiment:

The experiment involves training a global model in the federated setting.

Federated parameters (default values):

```Fraction of users (C)```: 0.1\
```Local Batch size  (B)```: 10\
```Local Epochs      (E)```: 10\
```Optimizer            ```: SGD\
```Learning Rate        ```: 0.01

Table 2: Test accuracy after training for 10 global epochs with:

| Model | IID     | Non-IID |
|-------|---------|---------|
| MLP   | 87.3    | 72.4    |
| CNN   | 96.3    | 74.9    |

## Contributing

Contributions and suggestions are welcome. Please fork the repo and open a pull request with your changes, ciao.
