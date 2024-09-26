# ==================== #
#    NEURAL NETWORKS   #
# ==================== #

from headers import * # local_module
#from utils import *
from config import * # local_module
import config
#from dataset import *
#from activelearners import *

# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 16)
        self.enc2 = self.conv_block(16, 64)
        #self.enc3 = self.conv_block(32, 64)

        # Bottleneck
        self.bottleneck = self.conv_block(64, 128)

        # Decoder
        #self.upconv3 = self.upconv(128, 64)
        #self.dec3 = self.conv_block(128, 64)

        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.upconv1 = self.upconv(64, 16)
        self.dec1 = self.conv_block(32, 16)

        # Final output layer
        self.final_conv = nn.Conv2d(16, config.num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        #enc3 = self.enc3(F.max_pool2d(enc2, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        # Decoder
        #dec3 = self.upconv3(bottleneck)
        #dec3 = torch.cat((dec3, enc3), dim=1)
        #dec3 = self.dec3(dec3)

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output layer
        return self.final_conv(dec1)


# Early Stopping Class (callable class, behaves like a function)
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            #self.save_checkpoint(val_loss, model)
        elif val_loss > (self.best_score - self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            #self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_score = val_loss