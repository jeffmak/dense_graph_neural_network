import torch

def generate_edge_labels(asort, order, I):
  # Generates labels for sorting dataset
  # input: asort - ???, order - ???, I - adjacency matrix
  # output: adjacency matrix
  # example input and output???
  src = asort[order-1]
  src[asort[0]] = -1
  src = src.squeeze(-1).expand(-1,I.shape[-1])
  return torch.eq(src,I-1)


def generate_batch(num_vertices, batch_size, device, delete_edge=False):
  # Function to generate the input and output data for the sorting task dataset
  # with a fixed batch size and number of values to sort.

  # 1. Define batched graph components using PyTorch tensors
  # Context Tensor: one context feature per graph
  C = torch.zeros(batch_size,1)

  # Vertex Tensor - records tensor
  V = torch.rand(batch_size,num_vertices,1)

  # Incoming Tensor - records vertices with incoming edges to a particular vertex for each vertex
  I = torch.arange(1,num_vertices+1).unsqueeze(0).unsqueeze(0).expand(batch_size,num_vertices,-1)

  # Edge Tensor - records vertices with incoming edges to a particular vertex for each vertex
  E = torch.zeros(batch_size,num_vertices,num_vertices,1)

  # use CUDA if possible
  if device == torch.device('cuda'):
    C = C.to(device)
    V = V.to(device)
    I = I.to(device)
    E = E.to(device)

  # 2. Compute labels for the training input data
  asort = torch.argsort(V,dim=1)
  sorted_idx = torch.arange(num_vertices).expand(V.shape[0],-1).unsqueeze(-1).to(device)
  order = torch.zeros_like(asort).scatter(1, asort, sorted_idx)
  least = torch.where(order==0,torch.ones_like(order),torch.zeros_like(order))

  # Labels for vertex tensor
  target_V = torch.cat((least,1-least),dim=-1).float()

  # Labels for edge tensor
  target_E = torch.cat([generate_edge_labels(a,o,i).unsqueeze(0) for a, o, i in zip(asort,order,I)]).float().to(device)
  target_E = target_E.unsqueeze(-1)
  target_E = torch.cat([target_E,1-target_E],dim=-1)

  # Graph structure remains same after passing input graphs
  # through a graph neural network
  target_I = I.to(device)

  # The output context tensor is discarded in the sorting task, so it can be
  # the same as the input
  target_C = C.to(device)

  # Delete non-essential edges randomly in the graph
  # This shouldn't affect the machine learning model
  if delete_edge:
    #randomly delete non-essential edges
    # 1. Identify which edges are non-essential (make mask)
    mask_f = target_E[:,:,:,1] * (torch.rand(target_I.shape, device=device) > 0.4).float()
    mask = mask_f.long()
    # 2. delete such edges
    new_I = I * (1-mask)
    target_E = target_E * (1-mask_f).to(device).unsqueeze(-1).expand(-1,-1,-1,target_E.shape[-1])
  else:
    new_I = I

  return {'num_vertices': num_vertices,
          'batch_size': batch_size,
          'input': {'context': C, 'vertex': V, 'incoming': new_I, 'edge': E},
          'target':{'context': target_C, 'vertex': target_V,
                    'incoming': new_I, 'edge': target_E}
         }
