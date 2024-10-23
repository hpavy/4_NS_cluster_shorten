import torch
from model import pde
import numpy as np
import time
from utils import read_csv, write_csv
from pathlib import Path


def train(
    nb_itt,
    train_loss,
    test_loss,
    resample_rate,
    display,
    poids,
    model,
    loss,
    optimizer,
    X,
    U,
    n_pde,
    X_test_pde,
    X_test_data,
    U_test_data,
    n_data,
    rectangle,
    device,
    Re,
    time_start,
    f,
    x_std,
    y_std,
    u_mean,
    v_mean,
    p_std,
    t_std,
    u_std,
    v_std,
    folder_result,
    save_rate,
):
    nb_it_tot = nb_itt + len(train_loss["total"])
    # Nos datas initiales
    points_data_train = np.random.choice(len(X), n_data, replace=False)
    X_train_data = torch.from_numpy(X[points_data_train]).requires_grad_().to(device)
    U_train_data = torch.from_numpy(U[points_data_train]).requires_grad_().to(device)
    X_train_pde = rectangle.generate_random(n_pde).to(device)
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}"
        + "\n--------------------------"
    )
    print(
        f"--------------------------\nStarting at epoch: {len(train_loss['total'])}\n------------"
        + "--------------",
        file=f,
    )

    for epoch in range(len(train_loss["total"]), nb_it_tot):
        model.train()  # on dit qu'on va entrainer (on a le dropout)

        ## loss du pde
        pred_pde = model(X_train_pde)
        pred_pde1, pred_pde2, pred_pde3 = pde(
            pred_pde,
            X_train_pde,
            Re=Re,
            x_std=x_std,
            y_std=y_std,
            u_mean=u_mean,
            v_mean=v_mean,
            p_std=p_std,
            t_std=t_std,
            u_std=u_std,
            v_std=v_std,
        )
        loss_pde = (
            torch.mean(pred_pde1**2)
            + torch.mean(pred_pde2**2)
            + torch.mean(pred_pde3**2)
        )

        # loss des points de data
        pred_data = model(X_train_data)
        loss_data = loss(U_train_data, pred_data)  # (MSE)

        # loss totale
        loss_totale = poids[0] * loss_data + poids[1] * loss_pde

        # Backpropagation
        loss_totale.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        # Pour le test :
        model.eval()

        # loss du pde
        test_pde = model(X_test_pde)
        test_pde1, test_pde2, test_pde3 = pde(
            test_pde,
            X_test_pde,
            Re=Re,
            x_std=x_std,
            y_std=y_std,
            u_mean=u_mean,
            v_mean=v_mean,
            p_std=p_std,
            t_std=t_std,
            u_std=u_std,
            v_std=v_std,
        )
        loss_test_pde = (
            torch.mean(test_pde1**2)
            + torch.mean(test_pde2**2)
            + torch.mean(test_pde3**2)
        )
        # loss de la data
        test_data = model(X_test_data)
        loss_test_data = loss(U_test_data, test_data)  # (MSE)

        # loss totale
        loss_test = poids[0] * loss_test_data + poids[1] * loss_test_pde
        with torch.no_grad():
            train_loss["total"].append(loss_totale.item())
            train_loss["data"].append(loss_data.item())
            train_loss["pde"].append(loss_pde.item())
            test_loss["total"].append(loss_test.item())
            test_loss["data"].append(loss_test_data.item())
            test_loss["pde"].append(loss_test_pde.item())

        if ((epoch + 1) % display == 0) or (epoch + 1 == nb_it_tot):
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :", file=f)
            print(
                f"Train : loss: {np.mean(train_loss['total'][-display:]):.3e}, data: {np.mean(train_loss['data'][-display:]):.3e}, pde: {np.mean(train_loss['pde'][-display:]):.3e}"
            )
            print(
                f"Train : loss: {np.mean(train_loss['total'][-display:]):.3e}, data: {np.mean(train_loss['data'][-display:]):.3e}, pde: {np.mean(train_loss['pde'][-display:]):.3e}",
                file=f,
            )
            print(
                f"Test : loss: {np.mean(test_loss['total'][-display:]):.3e}, data: {np.mean(test_loss['data'][-display:]):.3e}, pde: {np.mean(test_loss['pde'][-display:]):.3e}"
            )
            print(
                f"Test : loss: {np.mean(test_loss['total'][-display:]):.3e}, data: {np.mean(test_loss['data'][-display:]):.3e}, pde: {np.mean(test_loss['pde'][-display:]):.3e}",
                file=f,
            )
            print(f"time: {time.time()-time_start:.0f}s")
            print(f"time: {time.time()-time_start:.0f}s", file=f)

        if epoch <= 4:
            print(
                f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss['total'][-1]:.3e},"
                + f" data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}"
            )
            print(
                f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss['total'][-1]:.3e},"
                + f" data: {train_loss['data'][-1]:.3e}, pde: {train_loss['pde'][-1]:.3e}",
                file=f,
            )

        if (epoch + 1) % resample_rate == 0:
            points_data_train = np.random.choice(len(X), n_data, replace=False)
            X_train_data = (
                torch.from_numpy(X[points_data_train]).requires_grad_().to(device)
            )
            U_train_data = (
                torch.from_numpy(U[points_data_train]).requires_grad_().to(device)
            )
            X_train_pde = rectangle.generate_random(n_pde).to(device)

        if (epoch + 1) % save_rate == 0:
            dossier_midle = Path(folder_result + f"/epoch{len(train_loss['total'])}")
            dossier_midle.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                folder_result
                + f"/epoch{len(train_loss['total'])}"
                + "/model_weights.pth",
            )
            write_csv(
                train_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/train_loss.csv",
            )
            write_csv(
                test_loss,
                folder_result + f"/epoch{len(train_loss['total'])}",
                file_name="/test_loss.csv",
            )
