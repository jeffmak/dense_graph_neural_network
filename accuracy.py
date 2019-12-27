import torch

def acc_vertex(target, output):
  # Computes the accuracy of the output vertex predictions
  prod = torch.sum(torch.softmax(output['vertex'],-1)[:,:,0] * target['vertex'][:,:,0])
  return prod/target['vertex'].shape[0]

def acc_edge(target, output, num_edges_per_graph):
  # Computes the accuracy of the output edge predictions
  prod = (torch.softmax(output['edge'],-1)*target['edge']).sum(-1)
  top = prod.sum((1,2))
  x = top / num_edges_per_graph.squeeze(-1)
  return x.mean()
