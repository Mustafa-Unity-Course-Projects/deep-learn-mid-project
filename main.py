import torch

from train_ac import ac_model
from train_lights import light_model


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def predict(model, observation: list[float]):
    print(observation, ": ", model(torch.tensor([observation], dtype=torch.float32, device=device)).max(1).indices.view(1, 1))


def main():
    predict(ac_model, [0, 10.0])
    predict(ac_model, [0, 24.9])
    predict(ac_model, [0, 25.5])
    predict(ac_model, [0, 39.0])
    predict(ac_model, [1, 10.0])
    predict(ac_model, [1, 24.9])
    predict(ac_model, [1, 25.5])
    predict(ac_model, [1, 39.0])

    predict(light_model, [0, 0.1])
    predict(light_model, [0, 0.5])
    predict(light_model, [0, 1.0])
    predict(light_model, [1, 0.1])
    predict(light_model, [1, 0.49])
    predict(light_model, [1, 0.5])
    predict(light_model, [1, 0.51])
    predict(light_model, [1, 1.0])

    torch.save(light_model.state_dict(), "light_model_weights")
    torch.save(ac_model.state_dict(), "ac_model_weights")


if __name__ == "__main__":
    main()


