import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from ThermoGNN.dataset import load_dataset
from ThermoGNN.model import GraphGNN, LogCoshLoss, WeightedMSELoss
from ThermoGNN.training import (EarlyStopping, evaluate, metrics, set_seed, train)


def run_case_study(model, task, graph_dir, weight_dir, fold=5, visualize=False):

    print(f"Task: {task}")

    test_data_list = load_dataset(graph_dir, task)
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    test_direct_loader = DataLoader(
        test_direct_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(
        test_reverse_dataset, batch_size=256, follow_batch=['x_s', 'x_t'], shuffle=False)

    total_pred_dir = []
    total_pred_rev = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(fold):

        model.load_state_dict(torch.load(f"{weight_dir}/model_{i + 1}.pkl"))
        pred_dir, y_dir = evaluate(model, test_direct_loader, device, return_tensor=True)
        pred_rev, y_rev = evaluate(model, test_reverse_loader, device, return_tensor=True)

        corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta = metrics(
            pred_dir, pred_rev, y_dir, y_rev)

        print(f'Fold {i + 1}, Direct PCC: {corr_dir:.3f}, Direct RMSE: {rmse_dir:.3f},'
              f' Reverse PCC: {corr_rev:.3f}, Reverse RMSE: {rmse_rev:.3f},'
              f' Dir-Rev PCC {corr_dir_rev:.3f}, <Delta>: {delta:.3f}')

        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(
        avg_pred_dir, avg_pred_rev, y_dir, y_rev)

    print(f'Avg Direct PCC: {avg_corr_dir:.3f}, Avg Direct RMSE: {avg_rmse_dir:.3f},'
          f' Avg Reverse PCC: {avg_corr_rev:.3f}, Avg Reverse RMSE: {avg_rmse_rev:.3f},'
          f' Avg Dir-Rev PCC {avg_corr_dir_rev:.3f}, Avg <Delta>: {avg_delta:.3f}')

    if visualize:
        wandb.init(project="ThermoGNN", group=os.path.dirname(weight_dir),
                   name=f"{os.path.dirname(weight_dir)}-{task}")

        wandb.run.summary['Avg Direct PCC'] = avg_corr_dir
        wandb.run.summary['Avg Direct RMSE'] = avg_rmse_dir
        wandb.run.summary['Avg Reverse PCC'] = avg_corr_rev
        wandb.run.summary['Avg Reverse RMSE'] = avg_rmse_rev
        wandb.run.summary['Avg Dir-Rev PCC'] = avg_corr_dir_rev
        wandb.run.summary['Avg <Delta>'] = avg_delta

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=300, figsize=(15,5))

        ax1.scatter(y_dir.cpu().numpy(), pred_dir.cpu().numpy(), c=(
            y_dir-pred_dir).cpu().numpy(), cmap="bwr", alpha=0.5, edgecolors="grey", linewidth=0.1, norm=colors.CenteredNorm())
        ax1.plot((-4.5, 6.5), (-4.5, 6.5), ls='--', c='k')
        ax1.set_xlabel(r'Experimental $\Delta \Delta G$ (kcal/mol)')
        ax1.set_ylabel(r'Predicted $\Delta \Delta G$ (kcal/mol)')
        ax1.set_xlim(-4.5, 6.5)
        ax1.set_ylim(-4.5, 6.5)
        ax1.text(0.25, 0.85, 'Direct mutations', horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.75, 0.2, r'$r = {:.2f}$'.format(avg_corr_dir), horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.text(0.75, 0.12, r'$\sigma = {:.2f}$'.format(avg_rmse_dir), horizontalalignment='center',
                 verticalalignment='center', transform=ax1.transAxes)
        ax1.grid(ls='--', alpha=0.5, linewidth=0.5)

        ax2.scatter(y_rev.cpu().numpy(), pred_rev.cpu().numpy(), c=(
            y_rev-pred_rev).cpu().numpy(), cmap="bwr", alpha=0.5, edgecolors="grey", linewidth=0.1, norm=colors.CenteredNorm())
        ax2.plot((-6.5, 4.5), (-6.5, 4.5), ls='--', c='k')
        ax2.set_xlabel(r'Experimental $\Delta \Delta G$ (kcal/mol)')
        ax2.set_ylabel(r'Predicted $\Delta \Delta G$ (kcal/mol)')
        ax2.set_xlim(-6.5, 4.5)
        ax2.set_ylim(-6.5, 4.5)
        ax2.text(0.25, 0.85, 'Reverse mutations', horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.text(0.75, 0.2, r'$r = {:.2f}$'.format(avg_corr_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.text(0.75, 0.12, r'$\sigma = {:.2f}$'.format(avg_rmse_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.grid(ls='--', alpha=0.5, linewidth=0.5)

        ax3.scatter(pred_dir.cpu().numpy(), pred_rev.cpu().numpy(),
                    c='#3944BC', alpha=0.2, edgecolors="grey", linewidth=0.1)
        ax3.plot((-5, 5), (5, -5),  ls='--', c='k')
        ax3.set_xlabel('Prediction for direct mutation')
        ax3.set_ylabel('Prediction for reverse mutation')
        ax3.set_xlim(-5, 5)
        ax3.set_ylim(-5, 5)
        ax3.text(0.3, 0.2, r'$r = {:.2f}$'.format(avg_corr_dir_rev), horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes)
        ax3.text(0.3, 0.12, r'$\delta = {:.2f}$'.format(avg_delta), horizontalalignment='center',
                 verticalalignment='center', transform=ax3.transAxes)
        ax3.grid(ls='--', alpha=0.2, linewidth=0.5)

        plt.tight_layout()
        
        # wandb.log({"chart": plt})
        img = wandb.Image(fig)
        wandb.log({"chart": img})

        wandb.join()


def main():
    parser = argparse.ArgumentParser(description='ThermoGNN: predict thermodynamics stability')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=10,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss function (mse, logcosh, wmse)')
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=200,
                        help='embedding dimensions (default: 200)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.3,
                        help='dropout ratio (default: 0.3)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="mean",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--graph-dir', type=str,dest='graph_dir', default='data/graphs',
                        help='directory storing graphs data')
    parser.add_argument('--logging-dir', type=str, dest='logging_dir', default='./',
                        help='logging directory (default: \'./\')')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gin",
                        help='gnn type (gin, gcn, gat, graphsage)')
    parser.add_argument('--split', type=int, default=10,
                        help="Split k fold in cross validation (default: 10)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting dataset (default 42)")
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize training by wandb")
    args = parser.parse_args()

    set_seed(args.seed)

    weight_dir = os.path.join(args.logging_dir, "weights")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(args.logging_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(args.logging_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    logging.info('Loading Training Dataset')
    data_list = load_dataset(args.graph_dir, "train")
    direct_dataset, reverse_dataset = data_list[::2], data_list[1::2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=args.split, shuffle=True, random_state=args.seed)

    logging.info('Loading Test Dataset')
    test_data_list = load_dataset(args.graph_dir, "test")
    test_direct_dataset, test_reverse_dataset = test_data_list[::2], test_data_list[1::2]
    test_direct_loader = DataLoader(test_direct_dataset, batch_size=args.batch_size,
                                    follow_batch=['x_s', 'x_t'], shuffle=False)
    test_reverse_loader = DataLoader(test_reverse_dataset, batch_size=args.batch_size,
                                     follow_batch=['x_s', 'x_t'], shuffle=False)

    total_pred_dir = []
    total_pred_rev = []

    for i, (train_index, valid_index) in enumerate(kf.split(direct_dataset)):

        train_direct_dataset, valid_direct_dataset = [direct_dataset[i] for i in train_index], \
                                                     [direct_dataset[j] for j in valid_index]
        train_reverse_dataset, valid_reverse_dataset = [reverse_dataset[i] for i in train_index], \
                                                       [reverse_dataset[j] for j in valid_index]

        train_loader = DataLoader(train_direct_dataset + train_reverse_dataset, batch_size=args.batch_size,
                                  follow_batch=['x_s', 'x_t'], shuffle=True)
        valid_loader = DataLoader(valid_direct_dataset + valid_reverse_dataset, batch_size=args.batch_size,
                                  follow_batch=['x_s', 'x_t'], shuffle=False)

        train_direct_loader = DataLoader(train_direct_dataset, batch_size=args.batch_size,
                                         follow_batch=['x_s', 'x_t'], shuffle=False)
        train_reverse_loader = DataLoader(train_reverse_dataset, batch_size=args.batch_size,
                                          follow_batch=['x_s', 'x_t'], shuffle=False)
        valid_direct_loader = DataLoader(valid_direct_dataset, batch_size=args.batch_size,
                                         follow_batch=['x_s', 'x_t'], shuffle=False)
        valid_reverse_loader = DataLoader(valid_reverse_dataset, batch_size=args.batch_size,
                                          follow_batch=['x_s', 'x_t'], shuffle=False)

        model = GraphGNN(num_layer=args.num_layer, input_dim=60, emb_dim=args.emb_dim, out_dim=1, JK="last",
                         drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warm_steps)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_steps, after_scheduler=after_scheduler)

        if args.loss == "logcosh":
            criterion = LogCoshLoss()
        elif args.loss == "wmse":
            criterion = WeightedMSELoss()
        else:
            criterion = nn.MSELoss()

        weights_path = f"{weight_dir}/model_{i + 1}.pkl"
        early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {i + 1}')

        for epoch in range(1, args.epochs + 1):

            train_loss, valid_loss = train(model, train_loader, valid_loader, device, criterion, optimizer)
            scheduler.step(epoch)

            train_dir_pcc, train_dir_rmse = evaluate(model, train_direct_loader, device)
            train_rev_pcc, train_rev_rmse = evaluate(model, train_reverse_loader, device)
            valid_dir_pcc, valid_dir_rmse = evaluate(model, valid_direct_loader, device)
            valid_rev_pcc, valid_rev_rmse = evaluate(model, valid_reverse_loader, device)
            test_dir_pcc, test_dir_rmse = evaluate(model, test_direct_loader, device)
            test_rev_pcc, test_rev_rmse = evaluate(model, test_reverse_loader, device)

            logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}')
            logging.info(f'Train Direct PCC: {train_dir_pcc:.3f}, Train Direct RMSE: {train_dir_rmse:.3f},'
                         f' Train Reverse PCC: {train_rev_pcc:.3f}, Train Reverse RMSE: {train_rev_rmse:.3f}')
            logging.info(f'Valid Direct PCC: {valid_dir_pcc:.3f}, Valid Direct RMSE: {valid_dir_rmse:.3f},'
                         f' Valid Reverse PCC: {valid_rev_pcc:.3f}, Valid Reverse RMSE: {valid_rev_rmse:.3f}')
            logging.info(f'Test Direct PCC: {test_dir_pcc:.3f}, Test Direct RMSE: {test_dir_rmse:.3f},'
                         f' Test Reverse PCC: {test_rev_pcc:.3f}, Test Reverse RMSE: {test_rev_rmse:.3f}')
            
            if args.visualize:
                wandb.init(project="ThermoGNN", group=args.logging_dir, name=f'{args.logging_dir}_fold_{i+1}', config=args)
                wandb.log({'Train/Loss': train_loss, 'Valid/Loss': valid_loss}, step=epoch)
                wandb.log({'Train/Direct PCC': train_dir_pcc, 'Train/Direct RMSE': train_dir_rmse,
                           'Train/Reverse PCC': train_rev_pcc, 'Train/Reverse RMSE': train_rev_rmse}, step=epoch)
                wandb.log({'Valid/Direct PCC': valid_dir_pcc, 'Valid/Direct RMSE': valid_dir_rmse,
                           'Valid/Reverse PCC': valid_rev_pcc, 'Valid/Reverse RMSE': valid_rev_rmse}, step=epoch)
                wandb.log({'Test/Direct PCC': test_dir_pcc, 'Test/Direct RMSE': test_dir_rmse,
                           'Test/Reverse PCC': test_rev_pcc, 'Test/Reverse RMSE': test_rev_rmse}, step=epoch)
                
            early_stopping(valid_loss, model, goal="minimize")

            if early_stopping.early_stop:
                logging.info(f"Early stopping at Epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(weights_path))
        pred_dir, y_dir = evaluate(model, test_direct_loader, device, return_tensor=True)
        pred_rev, y_rev = evaluate(model, test_reverse_loader, device, return_tensor=True)

        corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta = metrics(pred_dir, pred_rev, y_dir, y_rev)

        logging.info(f'Fold {i + 1}, Best Valid Loss: {-early_stopping.best_score:.3f}')
        logging.info(f'Direct PCC: {corr_dir:.3f}, Direct RMSE: {rmse_dir:.3f},'
                     f' Reverse PCC: {corr_rev:.3f}, Reverse RMSE: {rmse_rev:.3f},'
                     f' Dir-Rev PCC {corr_dir_rev:.3f}, <Delta>: {delta:.3f}')

        if args.visualize:
            wandb.run.summary['Best Valid Loss'] = -early_stopping.best_score
            wandb.run.summary['Direct PCC'] = corr_dir
            wandb.run.summary['Direct RMSE'] = rmse_dir
            wandb.run.summary['Reverse PCC'] = corr_rev
            wandb.run.summary['Reverse RMSE'] = rmse_rev
            wandb.run.summary['Dir-Rev PCC'] = corr_dir_rev
            wandb.run.summary['<Delta>'] = delta

            wandb.join()

        total_pred_dir.append(pred_dir.tolist())
        total_pred_rev.append(pred_rev.tolist())

    avg_pred_dir = torch.Tensor(total_pred_dir).mean(dim=0).to(device)
    avg_pred_rev = torch.Tensor(total_pred_rev).mean(dim=0).to(device)
    avg_corr_dir, avg_rmse_dir, avg_corr_rev, avg_rmse_rev, avg_corr_dir_rev, avg_delta = metrics(avg_pred_dir,  avg_pred_rev, y_dir, y_rev)

    logging.info(f'Cross Validation Finished!')
    logging.info(f'Avg Direct PCC: {avg_corr_dir:.3f}, Avg Direct RMSE: {avg_rmse_dir:.3f},'
                 f' Avg Reverse PCC: {avg_corr_rev:.3f}, Avg Reverse RMSE: {avg_rmse_rev:.3f},'
                 f' Avg Dir-Rev PCC {avg_corr_dir_rev:.3f}, Avg <Delta>: {avg_delta:.3f}')
    
    # case studies
    model = GraphGNN(num_layer=args.num_layer, input_dim=60, emb_dim=args.emb_dim, out_dim=1, JK="last",
                     drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    run_case_study(model, "test", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    run_case_study(model, "p53", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)
    run_case_study(model, "myoglobin", args.graph_dir, weight_dir, fold=args.split, visualize=args.visualize)


if __name__ == "__main__":
    main()
