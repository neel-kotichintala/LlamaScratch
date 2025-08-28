# LLM from Scratch

---

## 1. Byte Pair Encoding (BPE)

BPE is a **tokenization method** used to split text into manageable units (tokens).  
Instead of splitting by characters or words, it learns a vocabulary of subwords.

### How It Works
1. Start with a text corpus, initially treating each character as a token.  
   Example: `"lower"` → `["l", "o", "w", "e", "r"]`

2. Count all **adjacent symbol pairs** in the dataset.  
   Example pairs: `"lo"`, `"ow"`, `"we"`, `"er"`

3. Merge the **most frequent pair** into a new token.  
   Example: `"er"` → new token `"er"`

4. Repeat until you reach the desired vocabulary size.  

### Advantages
- Handles **rare words** by breaking them into subwords.  
  - `"unhappiness"` → `["un", "happi", "ness"]`
- Efficient for large vocabularies.  
- Reduces unknown tokens compared to word-level tokenization.

---

## 2. Attention

Attention is the mechanism that lets models **decide which parts of the input are relevant** to each token.

### Key Components
- **Query (Q)**: What this token is looking for.  
- **Key (K)**: What each token represents.  
- **Value (V)**: The information carried by the token.

All are computed from embeddings via learned weight matrices:

```
Q = X * W_Q
K = X * W_K
V = X * W_V
```

### How It Works
1. Compute similarity between Query and all Keys (dot product).  
2. Apply softmax to get attention weights.  
3. Take weighted sum of Values → new representation for the token.

```
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V
```

### Intuition
- Query = "What am I asking about?"  
- Key = "What do I have?"  
- Value = "What info do I give if selected?"  

Example:  
For `"The cat sat"`, the word `"sat"` might attend strongly to `"cat"` because they are semantically linked.

### Multi-Head Attention
- Multiple sets of \(Q, K, V\) (heads).  
- Each head captures different relationships (syntax, long-range, semantics).  
- Outputs are concatenated and combined.

---

## 3. Rotary Embeddings (RoPE)

Transformers need **positional information** since attention is order-invariant.  
RoPE is a way to encode token positions **without fixed sinusoidal embeddings**.

### Idea
- Each token embedding is rotated in a **complex plane** by an angle depending on its position.  
- Rotation is applied **inside the attention mechanism**, not added externally.

### Mathematical Form
For each token vector split into pairs \((x_{2i}, x_{2i+1})\):

```
RoPE(x, p) = [
x_even * cos(theta_p) - x_odd * sin(theta_p),
x_even * sin(theta_p) + x_odd * cos(theta_p)
]
```

Where:
- p = position index  
- theta_p = rotation angle, usually decreasing with dimension index  

### Benefits
- Captures **relative positions** between tokens.  
- Extends naturally to longer sequences.  
- More memory-efficient than learned positional embeddings.

---
