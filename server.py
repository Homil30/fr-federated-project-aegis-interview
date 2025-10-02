import flwr as fl
import torch
from model import SimpleFaceNet
from utils.aggregation_utils import parameters_to_state_dict
from flwr.common import parameters_to_ndarrays
import shutil

# -------------------------------
# Custom FedAvg Strategy
# -------------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            print(f"[ROUND {rnd}] Saving aggregated model...")

            # Convert Flower Parameters → list of ndarrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            # Create model and load parameters
            model = SimpleFaceNet()
            state_dict = parameters_to_state_dict(ndarrays, model.state_dict())
            model.load_state_dict(state_dict, strict=False)

            # Save model after each round
            round_path = f"global_model_round_{rnd}.pt"
            torch.save(model.state_dict(), round_path)

            # Always update "latest" model → global_model.pt
            shutil.copy(round_path, "global_model.pt")

        return aggregated_parameters, aggregated_metrics


# -------------------------------
# Run Federated Server
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Federated Server...")

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Run server for 3 rounds
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    print("🎉 Federated training completed successfully. ✅ global_model.pt saved")
