# Graph Transformer Optimization

> This README file is adapted from the `gtv2` branch of this repository.



## Naming Examples of GTv3

1. GTv3 $[\text{SA}-\text{CA}_\text{e}]$

   ```python
   class attention_head(nn.Module):
       ...
       def forward(self, x, e):
           x_new, _ = self.self_att_node_to_node(x, e)  # x_new: [bs, n, d_head]
           _, e_new = self.cross_att_node_to_edge(x, e) # e_new: [bs, n, n, d_head]
           return x_new, e_new
   ```

2. GTv3 $[\text{CA}_\text{n}-\text{CA}_\text{e}(h_i^{\ell+1})]$

   ```python
   class attention_head(nn.Module):
       ...
       def forward(self, x, e):
           x_new, _ = self.cross_att_edge_to_node(x, e)     # x_new: [bs, n, d_head]
           _, e_new = self.cross_att_node_to_edge(x_new, e) # e_new: [bs, n, n, d_head]
           return x_new, e_new
   ```

3. GTv3 $[\text{SA}-\text{CA}_\text{n}(\hat{h}_i^{\ell+1})-\text{CA}_\text{e}(\hat{h}_i^{\ell+1})]$

   ```python
   class attention_head(nn.Module):
       ...
       def forward(self, x, e):
           x_hat, _ = self.self_att_node_to_node(x, e)      # x_hat: [bs, n, d_head]
           x_new, _ = self.cross_att_edge_to_node(x_hat, e) # h_new: [bs, n, d_head]
           _, e_new = self.cross_att_node_to_edge(x_hat, e) # e_new: [bs, n, n, d_head]
           return x_new, e_new
   ```

4. GTv3 $[\text{SA}-\text{CA}_\text{n}(\hat{h}_i^{\ell+1})-\text{CA}_\text{n}(h_i^{\ell+1})]$

   ```python
   class attention_head(nn.Module):
       ...
       def forward(self, x, e):
           x_hat, _ = self.self_att_node_to_node(x, e)      # x_hat: [bs, n, d_head]
           x_new, _ = self.cross_att_edge_to_node(x_hat, e) # h_new: [bs, n, d_head]
           _, e_new = self.cross_att_node_to_edge(x_new, e) # e_new: [bs, n, n, d_head]
           return x_new, e_new
   ```

   

## Preliminary Results

Regression results on a subset of the ZINC dataset (2000 training samples, 200 testing samples) with batch size 512. Models are trained for 250 epochs, and the loss values reported are the mean and standard deviation from the last 10 epochs.

|             Network             | Train Loss on ZINC | Test Loss on ZINC  | Time (min) |
| :-----------------------------: | :----------------: | :----------------: | :--------: |
|              GTv1              |   0.5629(0.0006)   |  0.5219(0.0024) |   7.9222   |
| GTv2-Weighted ($\alpha$=0.25) |   0.5934(0.0004)   |  0.5240(0.0009) |  14.9815 |
| GTv2-Weighted ($\alpha$=0.5) |   0.5829(0.0003)   |  0.5083(0.0005) |  15.8551 |
| GTv2-Weighted ($\alpha$=0.75) |   0.5661(0.0004)   |  0.4819(0.0008) |  16.4721 |
| GTv2-Weighted ($\alpha$=0.8) | 0.5734(0.0005) | 0.4855(0.0013) | 12.6444 |
| GTv2-Weighted ($\alpha$=0.9) | 0.5739(0.0007) | 0.4852(0.0012) | 16.2364 |
|          GTv2-Gated          |   0.5653(0.0006)   |  0.4947(0.0011) |  16.3939 |
|          GTv2-Mixed          |   0.5920(0.0002)   |  0.5125(0.0005) |  15.8853 |
|          **GTv2-FiLM**          |   **0.5418(0.0002)**   | **0.4614(0.0008)** |  16.5036 |
| GTv3 $[\text{SA}-\text{CA}_\text{e}]$ | 0.6487(0.0003) | 0.6037(0.0009) | 7.0779 |
| GTv3 $[\text{CA}_\text{n}-\text{CA}_\text{e}]$ | *0.5560(0.0005)* | *0.4763(0.0008)* | 10.2855 |
| GTv3 $[\text{SA}-\text{CA}_\text{e}(h_i^{\ell+1})]$ | 0.6438(0.0002) | 0.6006(0.0007) | 7.7025 |
| GTv3 $[\text{CA}_\text{n}-\text{CA}_\text{e}(h_i^{\ell+1})]$ | *0.5629(0.0007)* | *0.4768(0.0024)* | 10.1360 |
| GTv3 $[\text{CA}_\text{e}-\text{CA}_\text{n}(e_{ij}^{\ell+1})]$ | 0.6282(0.0004) | 0.5798(0.0007) | 11.1540 |
| GTv3 $[\text{SA}-\text{CA}_\text{n}(\hat{h}_i^{\ell+1})-\text{CA}_\text{e}(\hat{h}_i^{\ell+1})]$ | 0.5857(0.0003) | 0.5091(0.0009) | 14.6597 |
| GTv3 $[\text{SA}-\text{CA}_\text{n}(\hat{h}_i^{\ell+1})-\text{CA}_\text{n}(h_i^{\ell+1})]$ | *0.5749(0.0003)* | *0.4869(0.0006)* | 14.9075 |
| GTv3 $[\text{SA}-\text{CA}_\text{e}(\hat{h}_i^{\ell+1})-\text{CA}_\text{n}(\hat{h}_i^{\ell+1}, e_{ij}^{\ell+1})]$ | 0.6524(0.0002) | 0.6153(0.0004) | 14.3571 |
| GTv3 $[\text{CA}_\text{n}-\text{SA}(\hat{h}_i^{\ell+1})-\text{CA}_\text{e}(\hat{h}_i^{\ell+1})]$ | *0.5540(0.0004)* | *0.4730(0.0015)* | 15.3171 |
| GTv3 $[\text{CA}_\text{n}-\text{SA}(\hat{h}_i^{\ell+1})-\text{CA}_\text{n}(h_i^{\ell+1})]$ | 0.5879(0.0003) | 0.5077(0.0019) | 14.7363 |



## Attention Mechanisms of GTv2 and GTv3

GTv2 and GTv3 is an optimized version of the GTv1, which aims to enhance the modeling capabilities of graph data by making every component in the network "attentioned." The structure introduces three key types of attention mechanisms to capture interactions between nodes and edges effectively.

1. **Cross-attention: edge-to-node**
   - Update node features $h_i$ using edge features $e_{ij}$.
   
   - Mathematical expression: $h_i \leftarrow \sum\limits_{j \in \mathcal{V}} \text{softmax} \left( q_i^{\top} k_{ij} \right) v_{ij}$.
   
2. **Cross-attention: node-to-edge**
   - Update edge features $e_{ij}$ using information from its connected nodes ($h_i$, $h_j$).
   
   - Mathematical expression: $e_{ij} \leftarrow \frac{\exp \left( q_{ij}^{\top} k_i \right)}{\exp \left( q_{ij}^{\top} k_i \right) + \exp \left( q_{ij}^{\top} k_j \right)} v_i + \frac{\exp \left( q_{ij}^{\top} k_j \right)}{\exp \left( q_{ij}^{\top} k_i \right) + \exp \left( q_{ij}^{\top} k_j \right)} v_j$.
   
3. **Self-attention: node-to-node**
   - Update node features $h_i$ by directly considering the relationships with all other nodes.
   
   - Mathematical expression: $h_i \leftarrow \sum\limits_{j \in \mathcal{V}} \text{softmax} \left( q_i^{\top} k_j \right) v_j$.



## Integration Mechanisms in GTv2

GTv2 introduces three integration mechanisms for combining self-attention and cross-attention.

1. **Weighted integration:** Uses a fixed weight $\alpha$ to balance self-attention and cross-attention.
   
   $h_k = \alpha \cdot \text{CrossAttention}(h^{\ell}) + (1 - \alpha) \cdot \text{SelfAttention}(h^{\ell})$.
   
   > *Three variants tested*: $\alpha = 0.25, 0.5,$ and $0.75$.

2. **Gated integration:** Implements a learnable gating mechanism, and uses a sigmoid function to compute dynamic weights.

   $g = \sigma \left( W_g \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_g \right)$,
   
   $h_k = g \odot \text{CrossAttention}(h^{\ell}) + (1 - g) \odot \text{SelfAttention}(h^{\ell})$.

3. **Mixed integration:** Learns a linear combination of attention outputs.
   
   $h_k = W_m \cdot \text{Concat}[\text{CrossAttention}(h^{\ell}), \text{SelfAttention}(h^{\ell})] + b_m$.


4. **FiLM integration:** Learns leveraging a FiLM layer.
   
   $h_k = W_1 \cdot \text{SelfAttention}(h^{\ell}) + \lbrack W_2 \cdot \text{SelfAttention}(h^{\ell}) \rbrack \odot \text{CrossAttention}(h^{\ell}) + \text{CrossAttention}(h^{\ell})$.



## References

[Vijay Prakash Dwivedi and Xavier Bresson. "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699* (2020).](https://arxiv.org/abs/2012.09699)
