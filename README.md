# Graph Transformer Optimization

> This file is adapted from the generation of GPT-4o and Claude 3.5 Sonnet.

This repository contains an enhanced implementation of the [vanilla Graph Transformer](https://arxiv.org/abs/2012.09699) (GT) network, referred to as GTv1. The primary focus is on improving the model's performance through advancements in the attention layers, resulting in a new architecture called GTv2.



## Environment Setup and Datastes

The core implementation files, datasets, and environment setups are adapted from National University of Singapore CS5284 Graph Machine Learning [tutorial](https://github.com/xbresson/CS5284_2024/tree/main).



## Attention Mechanisms of GTv2

GTv2 is an optimized version of the GTv1, which aims to enhance the modeling capabilities of graph data by making every component in the network "attentioned." The structure introduces three key types of attention mechanisms to capture interactions between nodes and edges effectively.

1. **Cross-attention: node-to-edge**
   - Update node features $h_i$ using edge features $e_{ij}$.
   
   - Mathematical expression: $h_i \leftarrow \sum\limits_{j \in \mathcal{V}} \text{softmax} \left( q_i^{\top} k_{ij} \right) v_{ij}$.
   
2. **Cross-attention: edge-to-node**
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



## Results


> The results of task 2 and 3 are excerpted from the [repository](https://github.com/Klasnov/DiGress), which is forked from original code of the [DiGress](https://github.com/cvignac/DiGress).


### Task 1: Regression on ZINC Dataset

Regression results on a subset of the ZINC dataset (2000 training samples, 200 testing samples) with batch size 512. Models are trained for 250 epochs, and the loss values reported are the mean and standard deviation from the last 10 epochs.

|             Network             | Train Loss on ZINC | Test Loss on ZINC  | Time (min) |
| :-----------------------------: | :----------------: | :----------------: | :--------: |
|              GTv1              |   0.5629(0.0006)   |  0.5219(0.0024)   |   7.9222   |
| GTv2-Weighted ($\alpha$=0.25) |   0.5934(0.0004)   |  0.5240(0.0009)   |  14.9815   |
| GTv2-Weighted ($\alpha$=0.5) |   0.5829(0.0003)   |  0.5083(0.0005)   |  15.8551   |
| GTv2-Weighted ($\alpha$=0.75) |   0.5661(0.0004)   |  0.4819(0.0008)   |  16.4721   |
|          GTv2-Gated          |   0.5653(0.0006)   |  0.4947(0.0011)   |  16.3939   |
|          GTv2-Mixed          |   0.5920(0.0002)   |  0.5125(0.0005)   |  15.8853   |
|          GTv2-FiLM          |   **0.5418(0.0002)**   | **0.4614(0.0008)** |  16.5036   |



### Task 2: Generation on QM9 Dataset with DiGress

Generation results with the DiGress model on a subset of the QM9 dataset (2000 training samples, 200 testing samples) with batch size 512. Models are trained for 500 epochs, generating 1000 samples to evaluate validity, uniqueness, and novelty. The GTv2-Weighted model uses hyperparameter $\alpha = 0.5$.

|     Network      | Test Loss | Valid | Unique | Novelty |
| :--------------: | :---------------: | :--------: | :--------------: | :--------------: |
| *DiGress* | *142.9952* | *89.60%* | *99.33%* | *99.78%* |
| GTv1 | **142.6156** | 78.10% | 99.87% | 99.87% |
| GTv2-Weighted | 146.3584 | 73.10% | 99.61% | 99.61% |
|  GTv2-Gated  | 143.9262 | 72.20% | 99.86% | 99.86% |
|  GTv2-Mixed  | 144.2760 | 82.80% | 99.88% | **99.88%** |
|   GTv2-FiLM   | 143.8812 | **83.40%** | **100.00%** | 99.52% |



### Task 3: Generation on ZINC Dataset with DiGress

Generation results with the DiGress model on a subset of the ZINC dataset (5000 training samples, 200 testing samples) with batch size 512. Models are trained for 1000 epochs, generating 1000 samples to evaluate validity and uniqueness. The GTv2-Weighted model uses hyperparameter $\alpha = 0.5$.

|     Network      | Test Loss | Valid | Unique |
| :--------------: | :---------------: | :--------: | :--------------: |
| *DiGress* | *233.9124* | *60.90%* | *100.00%* |
| GTv1 | 249.4620 | 53.80% | 100.00% |
| GTv2-Weighted | 253.3110 | 53.10% | 100.00% |
|  GTv2-Gated  | 271.0986 | 58.40% | 100.00% |
|  GTv2-Mixed  | **240.4660** | 52.50% | 100.00% |
|   GTv2-FiLM   | 260.0413 | **60.60%** | 100.00% |



## References

1. [Vijay Prakash Dwivedi and Xavier Bresson. "A generalization of transformer networks to graphs." *arXiv preprint arXiv:2012.09699* (2020).](https://arxiv.org/abs/2012.09699)
2. [Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. "Digress: Discrete denoising diffusion for graph generation." *arXiv preprint arXiv:2209.14734* (2022).](https://arxiv.org/abs/2209.14734)
3. [Jonathan Ho, Ajay Jain and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
4. [Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin and Aaron Courville. "Film: Visual reasoning with a general conditioning layer." In Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1. 2018.](https://ojs.aaai.org/index.php/AAAI/article/view/11671)
