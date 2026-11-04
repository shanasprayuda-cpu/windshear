import torch
import numpy as np
from evaluation.metrics import mean_auprc


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    run_name="default"
):

    history = []

    best = -np.inf


    checkpoint_path = (
        f"checkpoints/"
        f"{run_name}_best_model.pt"
    )


    for epoch in range(1, epochs + 1):

        # ==========================
        # TRAINING
        # ==========================

        model.train()

        loss_sum = 0.0


        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)


            optimizer.zero_grad()


            out = model(x)


            loss = criterion(
                out,
                y
            )


            loss.backward()

            optimizer.step()


            loss_sum += loss.item()



        avg_loss = (
            loss_sum /
            len(train_loader)
        )


        # ==================================================
        # VALIDATION MODE (MODEL SELECTION)
        # ==================================================

        if val_loader is not None:


            model.eval()


            yt=[]
            yp=[]


            with torch.no_grad():

                for x,y in val_loader:


                    out=model(
                        x.to(device)
                    )


                    yt.append(
                        y.numpy()
                    )


                    yp.append(
                        torch.sigmoid(out)
                        .cpu()
                        .numpy()
                    )


            score = mean_auprc(
                np.concatenate(yt),
                np.concatenate(yp)
            )


            print(
                f"Epoch {epoch}/{epochs} "
                f"Loss={avg_loss:.5f} "
                f"Mean AUPRC={score:.5f}"
            )


            history.append(
                {
                    "epoch":epoch,
                    "loss":avg_loss,
                    "auprc":score
                }
            )


            # SAVE BEST VALIDATION MODEL

            if score > best:

                best = score


                torch.save(
                    model.state_dict(),
                    checkpoint_path
                )


        # ==================================================
        # FULL TRAINING MODE
        # ==================================================

        else:


            print(
                f"Epoch {epoch}/{epochs} "
                f"Loss={avg_loss:.5f} "
                f"Mode=FULL TRAINING"
            )


            history.append(
                {
                    "epoch":epoch,
                    "loss":avg_loss,
                    "auprc":None
                }
            )


            # lowest loss checkpoint

            score = -avg_loss


            if score > best:

                best = score


                torch.save(
                    model.state_dict(),
                    checkpoint_path
                )


    return best, history