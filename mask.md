# Mask
## Padding mask

Input sentence (source sentence or src in short) with $M$ words and $N$ PADs, where $s_i$ is the token ID (index of vocabulary) that represent to relative one-hot vector.
```math
\begin{align}
&\mathrm{src} : \nonumber\\
&\quad s = (s_0, s_1, s_2, \cdots, s_M, s_{m+1}, \cdots, s_{M+N-1}) \\
&\mathrm{one-hot vectors of src} : \nonumber \\
&\quad s \equiv \left(\begin{matrix}
s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} & \cdots & s_{0,d_{vocab}-1}\\
s_{1,0} & s_{1,1} & s_{1,2} & s_{1,3} & \cdots & s_{1,d_{vocab}-1}\\
s_{2,0} & s_{2,1} & s_{2,2} & s_{2,3} & \cdots & s_{2,d_{vocab}-1}\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
s_{M-1,0} & s_{M-1,1} & s_{M-1,2} & s_{M-1,3} & \cdots & s_{M-1,d_{vocab}-1}\\
s_{M,0} & s_{M,1} & s_{M,2} & s_{M,3} & \cdots & s_{M,d_{vocab}-1}\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
s_{M+N-1,0} & s_{M+N-1,1} & s_{M+N-1,2} & s_{M+N-1,3} & \cdots & s_{M+N-1,d_{vocab}-1}
\end{matrix}\right)
\end{align}
```

In the encoder, we first map the source sentences into vectors with look up table, which is equivalent to left multiplying a transversed matrix $W^{E}_{d_{vocab} \times d_{model}}$ by one-hot vectors. We call this as Embedding".
```math
\begin{align}
x &= s W \\
&= \left(\begin{matrix}
x_{0,0} & x_{0, 1} & \cdots & x_{0, d_{model} - 1} \\
x_{1,0} & x_{1, 1} & \cdots & x_{1, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
x_{M-1,0} & x_{M-1, 1} & \cdots & x_{M-1, d_{model} - 1} \\
x_{M,0} & x_{M, 1} & \cdots & x_{M, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
x_{M+N-1,0} & x_{M+N-1, 1} & \cdots & x_{M+N-1, d_{model} - 1} \\
\end{matrix}\right)
\end{align}
```

Remember that the words from $M$-th to $(M+N-1)$-th are all PADs, that we need to ignore them in the attention calculating process.

```math
\begin{align}
& Q = x W^{Q}, ~ K = x W^{K}, ~ V = x W^{V} \\
& Q = \left(\begin{matrix}
q_{0,0} & q_{0, 1} & \cdots & q_{0, d_{model} - 1} \\
q_{1,0} & q_{1, 1} & \cdots & q_{1, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
q_{M-1,0} & q_{M-1, 1} & \cdots & q_{M-1, d_{model} - 1} \\
q_{M,0} & q_{M, 1} & \cdots & q_{M, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
q_{M+N-1,0} & q_{M+N-1, 1} & \cdots & q_{M+N-1, d_{model} - 1} \\
\end{matrix}\right) \\
\nonumber\\
& K = \left(\begin{matrix}
k_{0,0} & k_{0, 1} & \cdots & k_{0, d_{model} - 1} \\
k_{1,0} & k_{1, 1} & \cdots & k_{1, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
k_{M-1,0} & k_{M-1, 1} & \cdots & k_{M-1, d_{model} - 1} \\
k_{M,0} & k_{M, 1} & \cdots & k_{M, d_{model} - 1} \\
\vdots & \vdots & \ddots & \vdots \\ 
k_{M+N-1,0} & k_{M+N-1, 1} & \cdots & k_{M+N-1, d_{model} - 1} \\
\end{matrix}\right) \\
\nonumber \\
& QK^T = \left(\begin{matrix}
qk^T_{0,0} & qk^T_{0,1} & \cdots & qk^T_{0,M-1} & qk^T_{0,M} & \cdots & qk^T_{0,M+N-1} \\
qk^T_{1,0} & qk^T_{1,1} & \cdots & qk^T_{1,M-1} & qk^T_{1,M} & \cdots & qk^T_{1,M+N-1} \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
qk^T_{M-1,0} & qk^T_{M-1,1} & \cdots & qk^T_{M-1,M-1} & qk^T_{M-1,M} & \cdots & qk^T_{M-1,M+N-1} \\
qk^T_{M,0} & qk^T_{M,1} & \cdots & qk^T_{M,M-1} & qk^T_{M,M} & \cdots & qk^T_{M,M+N-1} \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
qk^T_{M+N-1,0} & qk^T_{M+N-1,1} & \cdots & qk^T_{M+N-1,M-1} & qk^T_{M+N-1,M} & \cdots & qk^T_{M+N-1,M+N-1}
\end{matrix}\right)
\end{align}
```

The attention is given by
```math
\begin{equation}
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left( \frac{QK^T}{\sqrt{d_{model}}} \right) V,
\end{equation}
```
where $\mathrm{Attention}(Q,K,V)$ have a size of $d_{M+N-1} \times d_{model}$. Here we want to ignore the PADs by masking out the PAD elements in $QK^T$ matrix. 

Example of PAD mask:
```math
\begin{align}
v_{pad\_mask} &= (1, 1, \cdots, 1, 0, \cdots, 0) \\
M_{pad\_mask} &= v_{pad\_mask} \otimes v_{pad\_mask} \nonumber \\
&= \left(\begin{matrix}
1 & 1 & \cdots & 1 & 0 & \cdots & 0 \\
1 & 1 & \cdots & 1 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
\end{matrix}\right) \\
&\rightarrow \left(\begin{matrix}
0 & 0 & \cdots & 0 & \infty & \cdots & \infty \\
0 & 0 & \cdots & 0 & \infty & \cdots & \infty \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & \infty & \cdots & \infty \\
\infty & \infty & \cdots & \infty & \infty & \cdots & \infty \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots \\
\infty & \infty & \cdots & \infty & \infty & \cdots & \infty \\
\end{matrix}\right),
\end{align}
```

where the $0$ elements in $v_{pad\_mask}$ are from $M$-th to $(M+N-1)$-th. Then the attention is modified as following
```math
\begin{equation}
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left( \frac{QK^T - M_{pad\_mask}}{\sqrt{d_{model}}} \right) V.
\end{equation}
```



## Casual language modeling (CLM) mask
In the decoder, we will first compute the masked attention to make sure the model will predict the tokens one by one that only depend on the previous words.

Exapmle of CLM mask:
```math
\begin{align}
v_{tgt\_mask} &= (1, 1, 1, 1, 0, 0, 0, 0, 0, 0) \\
M_{tgt\_mask} &= v_{tgt\_mask} \otimes v_{tgt\_mask} \\
&= \left(\begin{matrix}
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{matrix}\right) \\
&\rightarrow \left(\begin{matrix}
0 & 0 & 0 & 0 & \infty & \infty & \infty & \infty & \infty & \infty \\
0 & 0 & 0 & 0 & 0 & \infty & \infty & \infty & \infty & \infty \\
0 & 0 & 0 & 0 & 0 & 0 & \infty & \infty & \infty & \infty \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & \infty & \infty & \infty \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \infty & \infty \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \infty \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{matrix}\right)
\end{align}
```
Code sample of generating casual mask:
```Python
import torch
tgt_length = 3000
mask_begin = 9
tgt_mask = torch.zeros(size = (tgt_length,), dtype = int)
tgt_mask[:mask_begin] = 1

mask_mat = torch.zeros(size  = (tgt_length, tgt_length), 
                       dtype = torch.float)

mask_mat_part = torch.triu(
    torch.full(
        size       = (tgt_length - mask_begin - 1, 
                      tgt_length - mask_begin - 1), 
        fill_value = -torch.inf,
        dtype      = float)
    )

mask_mat[:tgt_length - mask_begin - 1, mask_begin + 1:] = mask_mat_part
mask_mat
```

Attention:
```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T - M_{tgt\_mask}}{\sqrt{d_{model}}}\right) V
```

EXAMPLE:

$p_{i,j}$ : Weight between a given $i$-th query and $j$-th key

```math
\begin{equation}
\bf{x}_i = \sum_{j} p_{ij} \bf{v}_j
\end{equation}
```

```math
\begin{equation}
\left(\begin{matrix}
    \bf{x}_0 \\
    \bf{x}_1 \\
    \bf{x}_2 \\
    \vdots \\
    \bf{x}_M
\end{matrix}\right) = 
\left(\begin{matrix}
    p_{00} & p_{01} & p_{02} & \cdots & p_{0M} \\
    p_{10} & p_{11} & p_{12} & \cdots & p_{1M} \\
    p_{20} & p_{21} & p_{22} & \cdots & p_{2M} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    p_{M0} & p_{M1} & p_{M2} & \cdots & p_{MM}
\end{matrix}\right)
\left(\begin{matrix}
    \bf{v}_0 \\
    \bf{v}_1 \\
    \bf{v}_2 \\
    \vdots \\
    \bf{v}_M
\end{matrix}\right)
\end{equation}
```

For the given target $\mathrm{tgt}_{input} = (\mathrm{[BOS]}, ~\mathrm{YES}, ~\mathrm{,}, ~\mathrm{attention}, ~\mathrm{is}, ~\mathrm{very}, ~\mathrm{powerful}, ~\mathrm{.})$, suppose we want to predict the words behind $\mathrm{[BOS]}$. Hence we will need features from the $0$-th token for $\bf{x}_0$, the $0,1$-st tokens for $\bf{x}_1$, the $0,1,2$-nd tokens for $\bf{x}_2$, and so on. Next, we will compute the source target attention :
```math
\begin{align}
&\mathrm{Attention}(Q,K,V) \nonumber \\
=~& \mathrm{Softmax}({x}_{decoded} {W^Q} {W^V}^T {x}^T_{encoded}) {x}_{encoded} {W^V}
\end{align}
```
The $\mathrm{Softmax}$ computes the weights between query $Q = x_{decoded}W^Q$ which's $x$ from encoder and key $K = x_{encoded} W^K$ from decoder. Remember that the $\bf{x}_i$ only contains the features up to $i$-th tokens, that means the $i$-th output token only depends on the previous.


# Training steps
Example: 
* input: $\mathrm{src} = (\mathrm{Attention}, ~\mathrm{is}, ~\mathrm{all}, ~\mathrm{you}, ~\mathrm{need}, ~\mathrm{.}, ~\mathrm{[EOS]})$ 
* Expected output (or answer): $\mathrm{tgt} = (\mathrm{[BOS]}, ~\mathrm{Yes}, ~\mathrm{,}, ~\mathrm{attention}, ~\mathrm{is}, ~\mathrm{very}, ~\mathrm{powerful}, ~\mathrm{.}, ~\mathrm{[EOS]})$
* Predected output: $\mathrm{out} = (\bf{o}_0, ~\bf{o}_1,\cdots, ~\bf{o}_{m-2}, ~\bf{o}_{m-1})$, where $\bf{o}_i$ is the vector of $\mathrm{logit}$ values that $\bf{o}_i = (o_{i,0}, ~o_{i,1}, ~o_{i,d_{vocab}}) $ and $y_{pred} = \mathrm{softmax}(\mathrm{out}) = (y_0,~y_1, ~y_2, \cdots, ~y_{m-1})$
* Evaluate cross entropy with $\mathrm{out}$ and answer by using $\mathrm{tgt}_{output} \equiv (\mathrm{YES}, ~\mathrm{,}, ~\mathrm{attention}, ~\mathrm{is}, ~\mathrm{very}, ~\mathrm{powerful}, ~\mathrm{.}, ~\mathrm{[EOS]})$

Training steps:
1. $\mathrm{src} = (\mathrm{Attention}, ~\mathrm{is}, ~\mathrm{all}, ~\mathrm{you}, ~\mathrm{need}, ~\mathrm{.}, ~\mathrm{[EOS]})$ 
2. $s_{input} = (s_0, ~s_1, ~s_2, ~s_3, ~s_4, ~s_5, ~s_6) $, $\mathrm{src\_mask} = (1,1,1,1,1,1,1)$.
* In decoder:

    3. Compute $x_{encoded} = \mathrm{Encoder}(s_{input}, ~\mathrm{src\_mask})$

4. Right shift the target $\mathrm{tgt}$ to $\mathrm{tgt}_{input} = (\mathrm{[BOS]}, ~\mathrm{YES}, ~\mathrm{,}, ~\mathrm{attention}, ~\mathrm{is}, ~\mathrm{very}, ~\mathrm{powerful}, ~\mathrm{.})$
5. $\bf{t}_{input} = (t_0, ~t_1, ~t_2, ~t_3, ~t_4, ~t_5, ~t_6, ~t_7)$, $\mathrm{tgt\_mask} = (1,0,0,0,0,0,0,0)$.

* In encoder:

    6. Compute attention of input target $x_{decoded} = \mathrm{Attention}(Q,K,V)$ with target mask (in this case, we use casual mask) and then Add&Norm.
    7. Use the $x_{encoded}$ to construct query $Q$ and value $V$, $x_{decoded}$ to construct key $K$, then compute attention $x_{decoded} = \mathrm{Attention}(Q,K,V)$ and Add&Norm.
    8. Next Compute feed forward and Add&Norm and obtain $x_{decoded}$.
    9. Use the $x_{decoded}$ as input from 8. and repeat 6. to 8. for $N$ times where $N$ is the number of decoding block.
    
10. Compute the output logit values $\mathrm{out} = x_{decoded} W$.
11. Evaluate the cross entropy. 
