# SKRL F1Tenth Pipeline

This project implements an end-to-end reinforcement learning pipeline for the F1Tenth gym environment using the SKRL framework and Stable Baselines3. The pipeline includes data preparation, model training, and evaluation.

## Project Structure

- `src/skrl_qler/train.py`: Contains the implementation of the training pipeline using the SKRL framework. It defines the `SpatioTempDuelingTransformerNet` model architecture for reinforcement learning tasks, along with functions for loading data, creating a custom environment, and training a DQN agent.

- `src/skrl_qler/gym_interface.py`: Implements an end-to-end pipeline for training a policy using the Stable Baselines3 library. It prepares the dataset, defines the RLN model, sets up environment wrappers, and trains a PPO agent on the F1Tenth gym environment. It also includes functionality for logging and plotting training statistics.

- `src/skrl_qler/models.py`: Currently empty. This file can be used for defining additional models or utilities related to the project.

- `src/skrl_qler/gym_model.py`: Currently empty. This file can be used for defining gym-related models or utilities.

- `src/skrl_qler/skrl_f1tenth_pipeline.py`: Integrates the end-to-end pipeline from `gym_interface.py` using the SKRL framework and the model architecture from `train.py`. It sets up the F1Tenth gym environment, defines the necessary wrappers, and implements the training loop using SKRL.

- `requirements.txt`: Lists the dependencies required for the project, including libraries such as `torch`, `gymnasium`, `stable-baselines3`, and `skrl`.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the required format.
2. Run the training script:

```bash
python src/skrl_qler/train.py --npz <path_to_npz_files> --sup_model <path_to_supervised_model> --seq_len <sequence_length>
```

3. Monitor the training process through the generated plots and logs.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.