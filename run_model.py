import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from utils.diffusion2 import *

from torch.utils.data import DataLoader,Dataset
from utils.model import *

from utils.tools import *
from utils.EarlyStopping import *

from utils.parsers import parser

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 2022
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)


def main():

    data_start_time = time.time()

    #####load data
    with open(args.input + 'train.pkl', 'rb') as ftrain:
        train_cascade, train_global, train_label = pickle.load(ftrain)
    with open(args.input + 'val.pkl', 'rb') as fval:
        val_cascade, val_global, val_label = pickle.load(fval)
    with open(args.input + 'test.pkl', 'rb') as ftest:
        test_cascade, test_global, test_label = pickle.load(ftest)

    train_generator = MyDataset(train_cascade, train_global, train_label, args.max_seq)
    val_generator = MyDataset(val_cascade, val_global, val_label, args.max_seq)
    test_generator = MyDataset(test_cascade, test_global, test_label, args.max_seq)

    train_loader = DataLoader(train_generator, batch_size=args.b_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_generator, batch_size=args.b_size, num_workers=2)
    test_loader = DataLoader(test_generator, batch_size=args.b_size, num_workers=2)

    data_end_time = time.time()
    print('data loading Finished! Time used: {:.3f}mins.'.format((data_end_time - data_start_time) / 60))

    ## model
    model = ODE_diffusion_LODE(args, device)
    ####diffusuon model
    diff_model = Diffusion(args, device)
    diff_model = diff_model.to('cuda:0')
    diff_optimizer = torch.optim.Adam(diff_model.parameters(), lr=args.diff_lr)

    model = model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)
    ##loss
    MSLE_loss = MSLELoss()
    MSE = torch.nn.MSELoss()
    norm = nn.BatchNorm1d(args.max_seq).to('cuda:0')
    ##early_stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        total = 0
        total_loss = 0
        print('======================')
        print(f'EPOCH[{epoch}/{args.epochs}]')
        for step, (input, labels, time_steps) in enumerate(train_loader):
            cas_time = (1 - time_steps)
            cas_time = cas_time.to(torch.float32)
            cas_time = cas_time.to('cuda:0')

            input = input.to(torch.float32)
            input, labels = input.to('cuda:0'), labels.to('cuda:0')
            new_input = input
            input = norm(input)
            labels = labels.reshape([-1, 1])

            ##### model training
            model.train()
            z_0, z_t, out = model.encoder(input, cas_time)

            for j in range(1):
                diff_loss = diff_model.train_loss(new_input, z_0, z_t)
                diff_optimizer.zero_grad()
                diff_loss.backward(retain_graph=True)
                diff_optimizer.step()

            diff_out = diff_model.sample(new_input, z_0, z_t)
            diff_out = norm(diff_out)
            outputs, gaussian, diff_pred_x, diff_kl_loss = model.decoder(diff_out, out, cas_time)

            #####loss
            mse_loss = MSLE_loss(outputs, labels)
            loss = mse_loss + 0.1 * (diff_kl_loss + MSE(diff_out, input))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = float(total_loss + loss * labels.size(0))
            total = total + labels.size(0)
        train_loss = total_loss / total
        print("train_loss:", train_loss)

        # model validing
        model.eval()
        valid_total_num = 0
        total_valid = 0

        with torch.no_grad():
            for step, (input, labels, time_steps) in enumerate(val_loader):
                cas_time = (1 - time_steps)
                cas_time = cas_time.to(torch.float32)
                cas_time = cas_time.to('cuda:0')

                input = input.to(torch.float32)
                input, labels = input.to('cuda:0'), labels.to('cuda:0')
                new_input = input

                input = norm(input)
                z_0, z_t, out = model.encoder(input, cas_time)

                diff_out = diff_model.sample(new_input, z_0, z_t)
                diff_out = norm(diff_out)
                valid_output, gaussian, diff_pred_x, diff_kl_loss = model.decoder(diff_out, out, cas_time)

                labels = labels.reshape([-1, 1])
                valid_loss = MSLE_loss(valid_output, labels)

                valid_total_num += labels.size(0)
                total_valid += valid_loss * labels.size(0)

        aver_valid = total_valid / valid_total_num
        print("valid_loss:", aver_valid)
        print("========================")
        early_stopping(aver_valid, model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break

    model_dict = torch.load("./checkpoint.pt")
    model.load_state_dict(model_dict)
    # model test
    model.eval()
    total = 0
    MSLE_test = 0
    with torch.no_grad():
        for step, (input, labels,time_steps) in enumerate(test_loader):
            cas_time = (1 - time_steps)
            cas_time = cas_time.to(torch.float32)
            cas_time = cas_time.to('cuda:0')

            input = input.to(torch.float32)
            input, labels = input.to('cuda:0'), labels.to('cuda:0')
            new_input = input.to('cuda:0')
            input = norm(input)

            z_0, z_t, out = model.encoder(input, cas_time)

            diff_out = diff_model.sample(new_input, z_0, z_t)
            diff_out = norm(diff_out)
            test_output, gaussian, diff_pred_x, diff_kl_loss = model.decoder(diff_out, out, cas_time)

            labels = labels.reshape([-1, 1])

            MSLE = MSLE_loss(test_output, labels)

            total += labels.size(0)
            MSLE_test += MSLE * labels.size(0)

    aver_MSLE = MSLE_test / total
    print("========================")
    print("MSLE_loss:", aver_MSLE)
    print('Finished! Time used: {:.3f}mins.'.format((time.time() - start_time) / 60))

if __name__ == '__main__':
    main()
