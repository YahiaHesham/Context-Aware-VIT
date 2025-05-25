import os
import os.path as osp
import torch
from torch import nn, optim
from lib.utils.data_utils import build_data_loader
from configs.pie import parse_sgnet_args as parse_args
from lib.models.Trajnet import Trajnet
from lib.losses import rmse_loss
from lib.utils.train_val_test import train, val, test
from torch.utils.tensorboard import SummaryWriter

def main(args):
    this_dir = osp.dirname(__file__)
    logs_dir = osp.join(this_dir,"runs",args.version_name)
    if not osp.isdir(logs_dir):
        os.makedirs(logs_dir)
    writer = SummaryWriter(log_dir=logs_dir)
    val_save_dir = osp.join(this_dir, 'checkpoints', args.version_name, 'val')
    test_save_dir = osp.join(this_dir, 'checkpoints', args.version_name, 'test')
    if not osp.isdir(val_save_dir):
        os.makedirs(val_save_dir)
    if not osp.isdir(test_save_dir):
        os.makedirs(test_save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # FIXED: Create model with much smaller SOA weights
    model = Trajnet(args, device)
   
    model = nn.DataParallel(model)
    model = model.to(device)


    
    # SIMPLIFIED: Standard optimizer - no special SOA handling
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=args.patience,
                                                        min_lr=1e-10, verbose=1)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=args.patience,
                                                        min_lr=1e-10, verbose=1)
    
    # Load checkpoint if specified
    if osp.isfile(args.checkpoint):    
        checkpoint = torch.load(args.checkpoint, map_location=device)
        new_state_dict = {'module.' + k: v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_state_dict)
        args.start_epoch += checkpoint['epoch']      

    criterion = rmse_loss().to(device)
    
    # Build data loaders
    train_gen, scaler_sp = build_data_loader(args, 'train')
    val_gen, _ = build_data_loader(args, 'val', scaler_sp=scaler_sp)
    test_gen, _ = build_data_loader(args, 'test', scaler_sp=scaler_sp)
    
    print(f"Training samples: {len(train_gen)}")
    print(f"Validation samples: {len(val_gen)}")
    print(f"Test samples: {len(test_gen)}")

    min_loss = 1e6
    soa_enabled = False
    baseline_epochs = 1  # Train without SOA for first 2 epochs
    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # PROGRESSIVE SOA TRAINING STRATEGY
        

        # Training
        total_train_loss = train(model, train_gen, criterion, optimizer, device, epoch, writer, args)
        print(f'Train Epoch: {epoch + 1} \t Total Loss: {total_train_loss:.4f}')
        
        # Log SOA statistics during training
        

        # Validation
        val_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, \
        CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05, CFMSE_10 = val(model, val_gen, criterion, device, epoch, writer, args)
        
        lr_scheduler.step(val_loss)
        
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"MSE_05: {MSE_05:.1f};  MSE_10: {MSE_10:.1f};  MSE_15: {MSE_15:.1f}")
        
        # Check for improvement
        if val_loss < min_loss:
            print(f"📈 NEW BEST MODEL! (Previous: {min_loss:.4f} -> Current: {val_loss:.4f})")
            
            # Remove old best model
            try:
                os.remove(best_model_metric)
            except:
                pass

            min_loss = val_loss
            
            # Save metrics
            with open(os.path.join(val_save_dir, 'metric.txt'), "w") as f:
                f.write("%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\n" % 
                       (MSE_05, MSE_10, MSE_15, FMSE_05, FMSE_10, FMSE_15, FIOU_05, FIOU_10, FIOU_15, 
                        CFMSE_05, CFMSE_10, CFMSE_15, CMSE_05, CMSE_10, CMSE_15))

            saved_model_metric_name = f'metric_epoch_{epoch+1:03d}_loss_{min_loss:.4f}.pth'
            print(f"💾 Saving checkpoint: {saved_model_metric_name}")
            
            if not os.path.isdir(val_save_dir):
                os.mkdir(val_save_dir)
                
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'soa_enabled': soa_enabled,
                'min_loss': min_loss
            }
            torch.save(save_dict, os.path.join(val_save_dir, saved_model_metric_name))
            best_model_metric = os.path.join(val_save_dir, saved_model_metric_name)

            # Test on best model
            test_loss, MSE_15, MSE_05, MSE_10, FMSE_15, FMSE_05, FMSE_10, FIOU_15, FIOU_05, FIOU_10, \
            CMSE_15, CMSE_05, CMSE_10, CFMSE_15, CFMSE_05, CFMSE_10 = test(model, test_gen, criterion, device, epoch, writer, args)
            
            print(f"🎯 Test Loss: {test_loss:.4f}")
            print(f"Test MSE_05: {MSE_05:.1f};  MSE_10: {MSE_10:.1f};  MSE_15: {MSE_15:.1f}")
            
            with open(os.path.join(test_save_dir, 'metric.txt'), "w") as f:
                f.write("%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\t%4f\n" % 
                       (MSE_05, MSE_10, MSE_15, FMSE_05, FMSE_10, FMSE_15, FIOU_05, FIOU_10, FIOU_15, 
                        CFMSE_05, CFMSE_10, CFMSE_15, CMSE_05, CMSE_10, CMSE_15))
        else:
            print(f"📉 No improvement (Current: {val_loss:.4f} vs Best: {min_loss:.4f})")

        print("-" * 50)

    writer.close() 
    print(f"\n🏁 Training completed! Best validation loss: {min_loss:.4f}")


if __name__ == '__main__':
    main(parse_args())