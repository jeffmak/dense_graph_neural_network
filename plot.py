import matplotlib.pyplot as plt
import numpy as np
import torch

# Figure out which device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert(asort,value):
  return ((asort == torch.ones_like(asort)*value).long()
    * torch.arange(asort.shape[0],dtype=torch.long).unsqueeze(1).cuda()).sum()

def convert_batch(asorts, values):
  if isinstance(values, int):
    values = torch.LongTensor([values]).to(device)
  exp_values = values.unsqueeze(-1).unsqueeze(-1).expand(-1,asorts.shape[1],asorts.shape[2])
  return ((asorts == torch.ones_like(asorts)*exp_values).long().to(device)
    * torch.arange(asorts.shape[1],dtype=torch.long).unsqueeze(1).unsqueeze(0)
           .expand(asorts.shape[0],-1,asorts.shape[2]).to(device)).sum(1)

def new_plot(I, V, out_list, max_size=-1, index=0, avg_name="average",
            sample_name="sample"):
  # Rearrange the vertex order in the output edge tensor
  # in ascending vertex value
  batch_size = V.shape[0] if max_size == -1 else min(max_size, V.shape[0])
  num_v = V.shape[1]
  asort = V.argsort(1)
  grid = torch.zeros(batch_size, num_v, num_v)
  outputs = torch.softmax(out_list[-1]['edge'],-1)[:,:,:,0]
  for i in range(num_v):
    c2 = convert_batch(asort,i)
    for j in range(num_v):
      c1 = convert_batch(asort,I[:,i,j]-1)
      for b in range(batch_size):
        if I[b,i,j] > 0:
          grid[b,c1[b],c2[b]] = outputs[b,i,j]

  # Plot the resulting tensors on a 2D grid by ...
  # 1. Averaging in the batch dimension
  # plt.matshow(grid.mean(dim=0).cpu().detach().numpy())
  plt.axis('off')
  plt.imshow(grid.mean(dim=0).cpu().detach().numpy(), interpolation='nearest')
  plt.savefig('img/'+avg_name+'.png', bbox_inches='tight')
  plt.close()
  # 2. Picking a sample in the batch
  plt.axis('off')
  plt.imshow(grid[index,:].cpu().detach().numpy(), interpolation='nearest')
  plt.savefig('img/'+sample_name+'.png', bbox_inches='tight')
  plt.close()
