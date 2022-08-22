import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.utils import save_image
import torchvision.transforms as TF
import PIL
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

class CNNBlockSN(nn.Module):
    def __init__(self, in_channels, out_channels,kernel,stride):
        super(CNNBlockSN, self).__init__()
        self.conv=nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
            )),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel,stride):
        super(CNNBlock, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)

class CNNBlock3(nn.Module):
    def __init__(self, in_channels, out_channels,kernel,stride):
        super(CNNBlock3, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
            ),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)


class CNNBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

#Discriminateur 
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            )),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlockSN(in_channels, feature,kernel=4,stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            spectral_norm(nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self,in_channels,img_size,out_channels=256):
        super(SiameseNetwork, self).__init__()
        self.height,self.width=img_size
        self.Conv=nn.Sequential(
            CNNBlock(in_channels=in_channels,out_channels=out_channels,kernel=(self.height,9),stride=1),
            CNNBlock(in_channels=out_channels, out_channels=out_channels, kernel=(1,9), stride=(1,2)),
            CNNBlock(in_channels=out_channels, out_channels=out_channels, kernel=(1,7), stride=(1,2)),
        )
        self.Linear=nn.Linear(53760,128)
    def forward(self,x):
        output=self.Conv(x)
        output1=self.Linear(output.view(-1))
        return output1

class BlockSN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel,stride,pad,down=True):
        super(BlockSN, self).__init__()
        self.Block=nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                bias=False,
            ))if down else

            spectral_norm(nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                bias=False,
            )),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        output=self.Block(x)
        return output

class Generator(nn.Module):
    def __init__(self,in_channels,img_size):
        super(Generator,self).__init__()
        self.height,self.width=img_size
        #Down Parts
        self.Down1=BlockSN(in_channels=in_channels,out_channels=256,kernel=3,stride=2,pad=0)
        self.Down2=BlockSN(in_channels=256,out_channels=256,kernel=3,stride=2,pad=0)
        self.Down3=BlockSN(in_channels=256,out_channels=256,kernel=3,stride=2,pad=0)
        #Up Parts
        self.Up1=BlockSN(in_channels=256,out_channels=256,kernel=3,stride=2,pad=0,down=False)
        self.Up2=BlockSN(in_channels=512,out_channels=256,kernel=3,stride=2,pad=0,down=False)
        self.Up3=nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(
                in_channels=512,
                out_channels=3,
                kernel_size=(2,2),
                stride=2,
                padding=0,
                bias=False)),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        #self.PixelShuffle=nn.PixelShuffle(2)


    def forward(self,x):
       
        Down1=self.Down1(x)
        Down2=self.Down2(Down1)
        Down3=self.Down3(Down2)
        
        Up1=self.Up1(Down3)
        Up1=F.pad(Up1,pad=[Down2.shape[3]-Up1.shape[3],0,Down2.shape[2]-Up1.shape[2],0],mode="replicate")#Adding Pad

        Up2=self.Up2(torch.cat([Up1,Down2],dim=1))
        Up2=F.pad(Up2,pad=[Down1.shape[3]-Up2.shape[3],0,Down1.shape[2]-Up2.shape[2],0],mode="replicate")#Adding Pad

        Up3=self.Up3(torch.cat([Up2,Down1],dim=1))
        Up3=F.pad(Up3,pad=[self.width-Up3.shape[3],0,self.height-Up3.shape[2],0],mode="replicate")#Adding Pad

        return Up3


class MelSpecDataset(Dataset):
    def __init__(self, root_dir,Target_dir):
        self.InputPath = root_dir
        self.TargetPath= Target_dir
        self.InputFiles = os.listdir(self.InputPath)
        self.TargetFiles =os.listdir(self.TargetPath)

    def __len__(self):
        returnlen(self.InputFiles) or len(self.TargetFiles)

    def __getitem__(self, index):
        input_file = self.InputFiles[index]
        target_file= self.TargetFiles[index]

        input_path = self.InputPath+input_file
        input_image=np.array(Image.open(input_path))


        target_path = self.TargetPath+target_file
        target_image = np.array(Image.open(target_path))

        Transform=TF.Compose(
            [
                TF.ToTensor(),
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                #TF.Pad(padding=[1,0,0,0],fill=0,padding_mode="constant"),
            ]
        )


        return Transform(input_image), Transform(target_image)

if __name__=="__main__":
  Path_inputs="inputs path of your datasets"
  Path_Targets="targets path of your datasets"
  device = "cuda" if torch.cuda.is_available() else "cpu"
    
  writer_fake = SummaryWriter(f"/content/logs/fake")
  writer_real = SummaryWriter(f"/content/logs/real")
  writer=SummaryWriter()
  #HyperParameteres
  BATCH_SIZE=3
  NUM_EPOCHS=5
  ALFA=1
  BETA=10
  GAMA=10
  SIGMA=10
  Img_Size=(450,305)


  #DataLoader&Dataset
  Dataset=MelSpecDataset(Path_inputs,Path_Targets)
  DataLoader=DataLoader(Dataset,batch_size=BATCH_SIZE,shuffle=True)

  #Networks
  Generator=Generator(3,Img_Size)
  Discriminator=Discriminator()
  Siamese=SiameseNetwork(3,Img_Size)


  #Optimizers
  Gen_optim=optim.Adam(Generator.parameters(),lr=0.0001)
  Disc_optim=optim.SGD(Discriminator.parameters(),lr=0.0004)
  Siam_optim=optim.Adam(Siamese.parameters(),lr=0.0001)

  #Loss
  H_Loss=nn.HingeEmbeddingLoss()
  Cos_Sim=nn.CosineSimilarity(dim=0)
  L2=nn.MSELoss()

  #Variables
  step=0
  GenStep=0
  ones=None
  minus_ones=None


  Genlosstoal=0
  Discolsstotal=0
  Siamelosstotal=0

  for Epoch in range(NUM_EPOCHS):
    for i,(Input,Target) in enumerate(DataLoader):
      with torch.autograd.set_detect_anomaly(True):
        #Spliting Input Data to two
        Input1,Input2=torch.tensor_split(Input,2,dim=3)
        #Feed The inputs to the generator
        Gen1=Generator(Input1)
        Gen2=Generator(Input2)
        #Concatinate the two result to get the Fake one
        Fake=torch.cat([Gen1,Gen2],dim=3)
        #Train Discrminator
        D_Real=Discriminator(Input,Target)
        D_Fake=Discriminator(Input,Fake)

        if ones==None:
          ones=torch.ones_like(D_Real)
        if minus_ones==None:
          minus_ones=torch.full(D_Fake.shape,-1)
        
        H_loss_Real=H_Loss(D_Real,ones)
        #compare with zeros
        H_loss_Fake=H_Loss(D_Fake,minus_ones)

        #Compare With minus ones
        #H_loss_Fake=H_Loss(D_Fake,torch.full(D_fake.shape,fill_value=-1))
        H_loss_D=(H_loss_Real+H_loss_Real)/2
        Discriminator.zero_grad()
        H_loss_D.backward(retain_graph=True)
        Disc_optim.step()
        #-------Train Generator---------#
        output=Discriminator(Input,Fake)
        Loss_Gen=H_Loss(output,ones)

        #-------------SiamseNetwork--------------#
        Input1,Input2=torch.tensor_split(Input,2,dim=3)
        A1=Siamese(Input1)
        A2=Siamese(Input2)

        t12_input= A1-A2 #T (1,2)

        G1=Siamese(Gen1)
        G2=Siamese(Gen2)

        t12_generated= G1-G2 #T'(1,2)

        Travel_Loss1=Cos_Sim(t12_input,t12_generated)
        Travel_Loss2=L2(t12_input,t12_generated)**2

        Travel_Loss=(Travel_Loss1+Travel_Loss2)/2


        #-----------------------------------------#
        #Margin Loss
        Margin_Loss=max(0,SIGMA-L2(A1,A2)) #max(0,Sigma-||t12||2)


        #Siamese Loss
        Siamese_Loss = ( GAMA * Margin_Loss+ BETA * Travel_Loss )
        Siamese.zero_grad()
        Siamese_Loss.backward(retain_graph=True)




        #Identity Loss
        Bi=Target[:,:,:,0:int(Target.shape[3]/2)]
        Gen_Bi=Generator(Bi)
        Bi=F.pad(Bi,pad=[Gen_Bi.shape[3]-Bi.shape[3],0,0,0],value=0)
        Idendity_Loss=L2(Gen_Bi.reshape(-1),Bi.reshape(-1))**2


        #Generator Loss function
        Generator_Loss=(Loss_Gen+ (ALFA * Idendity_Loss)+ (BETA * Travel_Loss))/3
        Generator.zero_grad()
        Generator_Loss.backward(retain_graph=True)

        Gen_optim.step()
        Siam_optim.step()
        with torch.no_grad():
          print("Discriminator Loss:",H_loss_D," Generator Loss:",Generator_Loss," Siamese Loss:",Siamese_Loss,"Epoch:",Epoch)    
          if GenStep%200==0:
            img_grid_fake = torchvision.utils.make_grid(Fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(Target, normalize=True)
            writer_fake.add_image(
              "Fake Images", img_grid_fake, global_step=step
                )
            writer_real.add_image(
              "Real Images", img_grid_real, global_step=step
                  )
            step+=1 
  
    Genlosstoal=0
    Siamelosstotal=0
    Discolsstotal=0

  


  #Starting tensorboard 
  #%tensorboard --logdir /content/logs
