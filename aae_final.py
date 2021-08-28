import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse

from dataset import DAD
from dataset_test import DAD_Test
from models import Encoder,Discriminator,Decoder,linear
 
def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly detection using Adversarial autoencoders')
    parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
    parser.add_argument('--mode', default='train', type=str, help='train | test(validation)')
    parser.add_argument('--n_train_batch_size', default=200, type=int, help='Batch Size for training data')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch Size for validation data')
    parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--threshold', default=0.37, type=float, help='threshold of anomalies')
        
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = parse_args()
    if(args.mode == 'train'):
        training_normal_data = DAD(root_path=args.root_path,
                                   subset='train',
                                   view='front_IR',
                                   type='normal',
                                   )

       
        # training_normal_size = int(len(training_normal_data)*0.1)
        # training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))
        train_normal_loader = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size = args.n_train_batch_size,
            shuffle=True,
            num_workers= args.n_threads,
            pin_memory=True,
            drop_last=True
        )
    
    

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)

        encoder, decoder, discriminator = Encoder(), Decoder(), Discriminator()

        encoder.to(device)
        decoder.to(device)
        discriminator.to(device)

        # Set optimizators
        optim_enc = optim.Adam(encoder.parameters(), lr=0.0006)
        optim_dec = optim.Adam(decoder.parameters(), lr = 0.0006)
        optim_dis = optim.Adam(discriminator.parameters(), lr= 0.0008)
        optim_gen = optim.Adam(encoder.parameters(), lr = 0.0008)
    
        encoder.train()
        decoder.train()

        # for i in range(args.epochs//10):
        #     for j, batch in enumerate(train_normal_loader):
        #         inputs, targets = batch
        #         inputs = inputs.to(device)

        #         # train encoder-decoder
        #         encoder.zero_grad()
        #         decoder.zero_grad()
        #         z_sample = encoder(inputs)
        #         X_sample = decoder(z_sample)
        #         recon_loss = F.binary_cross_entropy(X_sample, inputs.view(-1, 36864))
        #         recon_loss.backward()
        #         optim_enc.step()
        #         optim_dec.step()

        #     print("[{:d}, recon_loss : {:.3f}]".format(i, recon_loss.data))
        
        # torch.save(encoder.state_dict(), args.root_path + "modelsBeforeDisc\encoder")
        # torch.save(decoder.state_dict(), args.root_path + "modelsBeforeDisc\decoder")    

        discriminator.train()
        torch.autograd.set_detect_anomaly(mode=True)
        for i in range(args.epochs):
            for j, batch in enumerate(train_normal_loader):
                inputs, targets = batch
                inputs = inputs.to(device)

                # train encoder-decoder
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()
                z_sample = encoder(inputs)
                X_sample = decoder(z_sample)
                recon_loss = F.binary_cross_entropy(X_sample, inputs.view(-1, 36864))
                recon_loss.backward()
                optim_enc.step()
                optim_dec.step()

                # train discriminator
                encoder.eval() 
                discriminator.zero_grad()

                u = np.random.rand(2*args.n_train_batch_size)
                sample = np.vectorize(linear)(u)
                z_real = torch.from_numpy(sample.reshape(-1,2)).float()
                z_real = z_real.to(device)
                z_fake = encoder(inputs)

                D_real, D_fake = discriminator(z_real), discriminator(z_fake)
                D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
                D_loss.backward()
                optim_dis.step()

                # train generator
                encoder.train()
                encoder.zero_grad()
                z_fake = encoder(inputs)
                D_fake = discriminator(z_fake)

                G_loss = -torch.mean(torch.log(D_fake))
                G_loss.backward()
                optim_gen.step()
            
       
            torch.save({
                'epoch': i,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optim_enc': optim_enc.state_dict(),
                'optim_dec' : optim_dec.state_dict(),
                'optim_dis' : optim_dis.state_dict(),
                'optim_gen' : optim_gen.state_dict(),
                'recon_loss': recon_loss,
                'G_Loss' : G_loss,
                'D_Loss' : D_loss
            
                }, args.root_path + "models\model{:d}".format(i))
            print("[{:d}, recon_loss : {:.3f}, D_loss : {:.3f}, G_loss : {:.3f}]".format(i, recon_loss.data, D_loss.data, G_loss.data))
    
        torch.save(encoder.state_dict(), args.root_path + "Models\encoder")
        torch.save(decoder.state_dict(), args.root_path + "Models\decoder")
        
        
    if(args.mode == 'test'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        encoder = Encoder()
        decoder = Decoder()
        
        encoder.load_state_dict(torch.load(args.root_path + "pretrained\encoder"))
        decoder.load_state_dict(torch.load(args.root_path + "pretrained\decoder"))
        encoder.eval()
        decoder.eval()

        encoder.to(device)
        decoder.to(device)
        


        test_data_front_ir = DAD_Test(root_path=args.root_path,
                                      subset='validation',
                                      view='front_IR',
                                      sample_duration=1,
                                      type=None
                                      )
    
        test_loader_front_ir = torch.utils.data.DataLoader(
            test_data_front_ir,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True          
        )
        print('Front IR view is done')
        
        all_values_count  = 0
        confirmed_anomaly_count = 0
        confirmed_normal_count = 0
        error_anomaly_count = 0
        error_normal_count = 0
    
    
    
        for j,batch in enumerate(test_loader_front_ir):
            inputs, targets = batch
            inputs = torch.squeeze(inputs,2)
            inputs = inputs.to(device)
        
            Y_Eval = encoder(inputs)
            X_Target = decoder(Y_Eval)
            recon_loss = F.binary_cross_entropy(X_Target, inputs.view(-1, 36864))
            label = 1
            if(recon_loss > args.threshold):
                label = 0
            for k in range(batch[0].size()[0]):
                if(targets[k].item() == label):
                    if(label==0):
                        confirmed_anomaly_count = confirmed_anomaly_count + 1
                    else:
                        confirmed_normal_count = confirmed_normal_count + 1
                else: 
                    if(label==0):
                        error_anomaly_count = error_anomaly_count + 1
                    else:
                        error_normal_count = error_normal_count + 1
                all_values_count = all_values_count + 1
    
        precision = confirmed_normal_count/(confirmed_normal_count + error_normal_count)
        recall = confirmed_normal_count/(confirmed_normal_count + error_anomaly_count)

        print("precision: {:.3f} , recall {:.3f} ".format(precision, recall))
        # precision: 0.665 , recall 0.998