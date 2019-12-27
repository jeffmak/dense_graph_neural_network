import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from generate_data import generate_batch
from encoder_core_decoder import EncoderCoreDecoder
from accuracy import acc_vertex, acc_edge
from plot import new_plot

# Figure out which device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for input tensors
num_vertices = 8
batch_size = 32

# Generate training and test data for sorting task
train_data = generate_batch(num_vertices, batch_size, device, delete_edge=True)
test_data = generate_batch(num_vertices*2, batch_size, device, delete_edge=True)

# Graph neural network parameters
# Input size
num_input_context_features = 1
num_input_vertex_features = 1
num_input_edge_features = 1
# Output sizes
num_output_context_features = 1
num_output_vertex_features = 2
num_output_edge_features = 2
# Embedded sizes
vertex_embed_size = 2
edge_embed_size = 2
context_embed_size = 2
# Other params
num_core_iterations = 5
num_hidden_units = 16

# Construct the graph neural network
graph_net = EncoderCoreDecoder(num_input_vertex_features, num_output_vertex_features,
                       vertex_embed_size, num_input_edge_features,
                       num_output_edge_features, edge_embed_size,
                       num_input_context_features, num_output_context_features,
                       context_embed_size, num_core_iterations, num_hidden_units,
                       num_vertices)

# Setup for training loop
num_epochs = 10001
metrics = {'i': [], 'train_loss': [], 'train_v_acc': [], 'train_e_acc': [],
                      'test_loss': [], 'test_v_acc': [], 'test_e_acc': []}
optim = torch.optim.Adam(lr=1e-3, params=graph_net.parameters())
train_num_edges_per_graph = (train_data['input']['incoming'] > 0).sum(2).unsqueeze(-1).float().sum(1)
test_num_edges_per_graph = (test_data['input']['incoming'] > 0).sum(2).unsqueeze(-1).float().sum(1)

# Run training loop
for i in range(num_epochs):
    graph_net.train()
    # Perform forward pass and compute loss
    out_list = graph_net(train_data['input'])
    loss = graph_net.batch_loss(train_data['target'], out_list)

    # Compute accuracies
    v_acc = acc_vertex(train_data['target'], out_list[-1])
    e_acc = acc_edge(train_data['target'], out_list[-1], train_num_edges_per_graph)

    # Perform backpropagation
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Evaluate graph neural network for tracking accuracy
    graph_net.eval()
    if i % 100 == 0:
      test_out_list = graph_net(test_data['input'])
      test_loss = graph_net.batch_loss(test_data['target'], test_out_list)
      test_v_acc = acc_vertex(test_data['target'], test_out_list[-1])
      test_e_acc = acc_edge(test_data['target'], test_out_list[-1], test_num_edges_per_graph)

      metrics['i'].append(i)
      metrics['train_loss'].append(loss)
      metrics['train_v_acc'].append(v_acc)
      metrics['train_e_acc'].append(e_acc)
      metrics['test_loss'].append(test_loss)
      metrics['test_v_acc'].append(test_v_acc)
      metrics['test_e_acc'].append(test_e_acc)

      print("epoch "+str(i)+": "+str(float(loss)) + " " +str(float(v_acc))+ " " +str(float(e_acc))
             +" "+str(float(test_loss)) + " " +str(float(test_v_acc))+ " " +str(float(test_e_acc)))

plt.plot(metrics['i'],metrics['train_loss'], label='train')
plt.plot(metrics['i'],metrics['test_loss'], label='test')
plt.xlabel('Batch Iterations')
plt.ylabel('Loss')
plt.yscale("log")
plt.legend()
plt.savefig('img/batch.pdf')
plt.close()

plt.plot(metrics['i'],metrics['train_v_acc'], label='train')
plt.plot(metrics['i'],metrics['test_v_acc'], label='test')
plt.xlabel('Batch Iterations')
plt.ylabel('Vertex Accuracy')
plt.yscale("linear")
plt.ylim(0,1)
plt.legend()
plt.savefig('img/vertex.pdf')
plt.close()

plt.plot(metrics['i'],metrics['train_e_acc'], label='train')
plt.plot(metrics['i'],metrics['test_e_acc'], label='test')
plt.xlabel('Batch Iterations')
plt.ylabel('Edge Accuracy')
plt.ylim(0,1)
plt.legend()
plt.savefig('img/edge.pdf')
plt.close()

#Figure out which random batch sample to pick
index = np.random.randint(train_data['batch_size'])
new_plot(train_data['input']['incoming'], train_data['input']['vertex'], out_list,
         index=index, avg_name="train_avg", sample_name="train_sample")
new_plot(test_data['input']['incoming'], test_data['input']['vertex'], test_out_list, index=index, max_size=100, avg_name="test_avg", sample_name="test_sample")
