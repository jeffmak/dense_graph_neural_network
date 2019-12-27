import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCoreDecoder(nn.Module):
  def net(self, in_features, out_features, hidden_size):
    # gives a generic three layer feedforward neural network
    # with ReLU activations
    return nn.Sequential(nn.Linear(in_features, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, out_features))

  def __init__(self, num_in_v_features, num_out_v_features, v_embed_size,
                     num_in_e_features, num_out_e_features, e_embed_size,
                     num_in_c_features, num_out_c_features, c_embed_size,
                     num_core_iterations, num_hidden, num_vertices):
    super(EncoderCoreDecoder, self).__init__()
    # Figure out which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Encoders for vertex, edge and context tensors
    self.enc_v = self.net(num_in_v_features, v_embed_size, num_hidden).to(device)
    self.enc_e = self.net(num_in_e_features, e_embed_size, num_hidden).to(device)
    self.enc_c = self.net(num_in_c_features, c_embed_size, num_hidden).to(device)

    # Decoders for vertex, edge and context tensors
    self.dec_v = self.net(v_embed_size, num_out_v_features, num_hidden).to(device)
    self.dec_e = self.net(e_embed_size, num_out_e_features, num_hidden).to(device)
    self.dec_c = self.net(c_embed_size, num_out_c_features, num_hidden).to(device)


    # Core networks
    # Compute input sizes
    self.v_dim = e_embed_size + v_embed_size + c_embed_size
    self.e_dim = v_embed_size*2 + e_embed_size + c_embed_size
    self.c_dim = e_embed_size + v_embed_size + c_embed_size

    # Core networks
    self.v_net = self.net(self.v_dim, v_embed_size, num_hidden).to(device)
    self.e_net = self.net(self.e_dim, e_embed_size, num_hidden).to(device)
    self.c_net = self.net(self.c_dim, c_embed_size, num_hidden).to(device)

    # Record number of output edge features and number of core iterations
    self.out_e_features = num_out_e_features
    self.num_core_iterations = num_core_iterations
    self.num_vertices = num_vertices

  def forward(self, graph_batch):
    # Encode data
    emb_V = self.enc_v(graph_batch['vertex'])
    emb_E = self.enc_e(graph_batch['edge'])
    emb_C = self.enc_c(graph_batch['context'])
    I = graph_batch['incoming']

    # Mask for later comparison
    # Moving operations relating to the incoming tensor I out of the loop
    # since I is not encoded
    # Assumes that the entire batch has the same graph topology
    mask_I = I.unsqueeze(-1).expand(-1,-1,-1,emb_V.shape[-1])
    # Padding I for convenient indexing
    pad_I = F.pad(I, (0,0,1,0), "constant", 0)
    upad_I = pad_I.unsqueeze(-1).expand(-1,-1,-1,emb_V.shape[-1])

    # Mask for removing output features after decoding
    emask_I = I.unsqueeze(-1).expand(-1,-1,-1,self.out_e_features)

    # Accumulate outputs from encode - (core) x N - decode architecture
    # for i = 1...N in out_list
    out_list = []

    # Core loop
    for i in range(self.num_core_iterations):
      # Operations relating to E
      # 1. Compute updated edge attributes
      # Padding emb_V for convenient indexing in gather
      pad_V = F.pad(emb_V, (0,0,1,0), "constant", 0).unsqueeze(2)
      rV = pad_V.expand(-1,-1,pad_I.shape[2],-1)
      sendV = rV.gather(1,upad_I)[:,1:,:]
      recV = rV[:,1:,:]

      # Sets embedded values to 0 for senders with id=0
      recV = torch.where(mask_I>0, recV, torch.zeros_like(recV))
      rC = emb_C.unsqueeze(1).unsqueeze(2).expand(-1,recV.shape[1],recV.shape[2],-1)
      e_input = torch.cat((emb_E,recV,sendV,rC),dim=-1)
      emb_E = self.e_net(e_input)

      # Mask emb_E to force edge_attrib values associated with i=0 to 0
      # We need this mask operation in order to compute e->v and e->c aggregations
      masked_emb_E = torch.where(mask_I>0,emb_E,torch.zeros_like(emb_E))

      # Operations relating to V
      # 2. Aggregate edge attributes per node
      edge_sum = masked_emb_E.sum(2) #sums up masked edge features per receiver vertex
      num_send_per_rec = (I > 0).sum(2).unsqueeze(-1).float() #counts the number of sender vertices per receiver vertex
      avgE = edge_sum / num_send_per_rec #computes average
      avgE[avgE!=avgE] = 0 #sets nan to zero since avg=0 iff no incoming edges

      # 3. Compute updated node attributes
      expC = emb_C.unsqueeze(1).expand(-1,emb_V.shape[1],-1)
      v_input = torch.cat((avgE,emb_V,expC),dim=-1)
      emb_V = self.v_net(v_input)

      # Operations relating to C
      # 4. Aggregate edge attributes globally
      num_edges_per_graph = num_send_per_rec.sum(1)
      global_agg_e = masked_emb_E.sum((1,2)) / num_edges_per_graph

      # 5. Aggregate node attributes globally
      global_agg_v = emb_V.sum(1) / self.num_vertices

      # 6. Compute updated global attribute
      c_input = torch.cat((global_agg_e,global_agg_v,emb_C),dim=-1)
      emb_C = self.c_net(c_input)

      # Decode data
      out_V = self.dec_v(emb_V)
      out_E = self.dec_e(emb_E)
      out_C = self.dec_c(emb_C)
      out_E = F.log_softmax(out_E, dim=-1)

      # Masking output for directed edges only present in the graph
      out_E = torch.where(emask_I>0,out_E,torch.zeros_like(out_E))
      out_list.append({'vertex': out_V, 'edge': out_E, 'context': out_C})
    return out_list

  def batch_loss(self, target_batch, out_list):
    # Compute loss function for a batch.
    criterion = nn.CrossEntropyLoss()
    loss = 0.
    for out_graph in out_list:
      v_label = target_batch['vertex'][:,:,1].long()
      e_label = target_batch['edge'][:,:,:,1].long()
      loss += criterion(out_graph['vertex'].view(-1,2), v_label.view(-1))
      loss += F.nll_loss(out_graph['edge'].view(-1,2), e_label.view(-1))
    return loss
