from data_loader import get_data_loader
from metrics_utility import MetricsCalculator
from torch import nn
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp
import itertools
import pandas as pd

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.type(y_pred.type())
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        return loss


arch = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
}

backbone_dataset = [
    ("resnet50", "imagenet"),
    ("timm-regnetx_064", "imagenet"),
    ("timm-regnety_064", "imagenet"),
]

loss = {
    "dice": smp.losses.DiceLoss(mode="binary"),
    "focal": smp.losses.FocalLoss(mode="binary"),
    "bce": BCEWithLogitsLoss(),
}

train, val, test = get_data_loader(
    "/workspaces/hw3_B11023038/ETT-v3/Fold1",
    shuffle=False,
    batch_size=16,
    num_workers=4,
    preprocess_fn=lambda x: x / 255.0,
)


def get_model(arch_name, backbone_name, loss_name):
    model = arch[arch_name](
        encoder_name=backbone_name[0],
        encoder_weights=backbone_name[1],
        in_channels=1,
        classes=1,
    )
    model = model.to(device)
    criterion = loss[loss_name]
    return model, criterion


def get_optimizer(model, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def train_model(arch_name, backbone_name, loss_name, fold, no, num_epochs=50):
    train, val, test = get_data_loader(
        f"/workspaces/hw3_B11023038/ETT-v3/Fold{fold}",
        shuffle=False,
        batch_size=8,
        num_workers=4,
        preprocess_fn=lambda x: x / 255.0,
    )

    count, patience = 0, 5
    best_vloss = 1e9
    model_path = "./model/{}_{}_{}_{}_{}".format(
        arch_name, backbone_name[0], loss_name, fold, no
    )
    model, loss_fn = get_model(arch_name, backbone_name, loss_name)
    optimizer = get_optimizer(model)

    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch + 1))
        model.train(True)
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train):
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(val):
                vinputs = vinputs.to(device, dtype=torch.float)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model.save_pretrained(
                model_path,
                metrics={"epoch": epoch, "loss": avg_vloss.cpu().tolist()},
                dataset="ETT-v3",
            )
            print("Model saved to {}".format(model_path))
            count = 0
        # else:
        #     count += 1
        #     if count >= patience:
        #         print("Early stopping")
        #         break

    model = smp.from_pretrained(model_path).to(device)
    model.eval()
    mc = MetricsCalculator(72, 7)

    with torch.no_grad():
        running_test_loss = 0.0
        for i, (tinputs, tlabels) in enumerate(test):
            assert len(tinputs) == len(tlabels)
            tinputs = tinputs.to(device, dtype=torch.float)
            tlabels = tlabels.to(device)
            toutputs = model(tinputs)
            tloss = loss_fn(toutputs, tlabels)
            mc(toutputs, tlabels)
            running_test_loss += tloss
        avg_test_loss = running_test_loss / (i + 1)
    del model
    metrics = mc.compute()
    metrics["loss"] = float(avg_test_loss.cpu())
    print(metrics)
    return metrics


if __name__ == "__main__":
    matrics_df = pd.DataFrame(
        columns=[
            "id",
            "fold",
            # "no",
            "iou",
            "error_cm",
            "error_0_5cm",
            "error_1cm",
            "loss",
        ]
    )
    # matrics_df = pd.read_csv("metrics.csv")
    arg_list = list(itertools.product(arch.keys(), backbone_dataset, loss.keys()))
    for arch_name, backbone_name, loss_name in arg_list:
        for fold in range(1, 6):
            print(
                f"Model: {arch_name}, Backbone: {backbone_name[0]}, Loss: {loss_name}"
            )
            for no in range(1):
                metrics = train_model(arch_name, backbone_name, loss_name, fold, no)
                matrics_df = pd.concat(
                    [
                        matrics_df,
                        pd.DataFrame(
                            {
                                "id": [f"{arch_name}_{backbone_name[0]}_{loss_name}"],
                                "fold": [fold],
                                # "no": [no],
                                "iou": [metrics["iou"]],
                                "error_cm": [metrics["error_cm"]],
                                "error_0_5cm": [metrics["error_0_5cm"]],
                                "error_1cm": [metrics["error_1cm"]],
                                "loss": [metrics["loss"]],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                matrics_df.to_csv("metrics_diff_loss.csv", index=False)
        # break
