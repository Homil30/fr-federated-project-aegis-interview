import flwr as fl
import argparse
import torch
import os
from utils.aggregation_utils import parameters_to_state_dict
from models.simple_facenet import SimpleFaceNet

# Custom FedAvg that saves global model each round
class SaveStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_template, save_dir, **kwargs):
        super().__init__(**kwargs)
        self.model_template = model_template
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            state_dict = parameters_to_state_dict(self.model_template, aggregated_parameters)
            torch.save(state_dict, os.path.join(self.save_dir, f"round-{server_round}.pt"))
            print(f"[server] ✅ Saved global model (round {server_round})")
        return aggregated_parameters, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    args = parser.parse_args()

    model_template = SimpleFaceNet()

    strategy = SaveStrategy(
        model_template=model_template,
        save_dir=args.save_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

if __name__ == "__main__":
    main()
