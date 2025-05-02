# Attention

## Scale dot product attention
Definition:
```math
\begin{equation}
\text{Attention}(Q,K,V) = \text{Softmax}(\frac{QK^T}{\sqrt{d}}) V,
\end{equation}
```
where
```math
\begin{equation}
Q = x W^Q, ~K = x W^K, ~V = x W^V,
\end{equation}
```
and $\text{Attention}$ has the size of $query\_seq\_length \times d_{model}$

## Multi head attention
In order to compute the attention parallelly, we use the so call 'Multi Head Attention' by devide the original attention into several parts.
Definition:

We devide the attention from size of $query\_seq\_length \times d_{model}$ to size of $query\_seq\_length \times d_{k}$ by applying linear mapping. 
```math
\begin{equation}
Q_i = x W_i^Q, ~K_i = x W_i^K, ~V_i = x W_i^V,
\end{equation}
```
where $W_i^{(Q,K,T)}$ are $query\_seq\_length \times d_k$ matrices, $d_k = d_{model}/h$ and $i = [0, d_k)$. We call the $h$ as 'head'. 

The $i$-th head attention is defined as following:
```math
\begin{align}
A_i &= \text{Attention}(Q_i, K_i, V_i) \nonumber \\
&= \text{Softmax}(\frac{Q_i K_i^T}{\sqrt{d_k}}) V_i.
\end{align}
``` 
where $A_i$ has the size of $query\_seq\_length \times d_k$. 
Finally we concat the heads with matrix $W^O$ with the size of $hd_k \times d_{model}$
```math
\begin{align}
\text{Maluti Head Attention} &= \left( A_0, ~A_1, ~\cdots, ~A_{h-1} \right) 
\left(\begin{matrix}
W_0^O \\
W_1^O \\
\vdots \\
W_{h-1}^O 
\end{matrix}\right) \nonumber \\
&= \sum_{i=0}^{h-1} A_i W_i^O
\end{align}
```

Code sampe:
```python
#As  : List of A_i with size of (n_batch, d_model, d_k)
#WOs : Module list of W^O_i with size of (d_k, d_model)
#i runs from 0 to h-1
Attention = 0
for i in range(h):
    Attention += As[i] @ WOs[i]
```

We can also use the tensor product
```math
(\text{Maluti Head Attention})_{n,d_m,d'_m,} = \sum_{i, d_k} A_{n,d_m,d_k,i} W^O_{i,d_k,d'_m}
```
Code Sample:
```python
#A : Multi head attentions with size of (n_batch, d_model, d_k, h)
#WO : Concat matrix with size of (h, dk, d_model)
Attention = torch.einsum('nmki,ikM->nmM', A, WO)
```