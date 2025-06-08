# Recurrent Neural Network (RNN): From Theory to Translation

RNN is a type of artificial neural network designed for processing sequential data, such as time series, natural language, or speech. Unlike traditional feedforward neural networks, RNNs have a "memory" that allows them to use information from previous inputs by passing it through a loop, making them well-suited for tasks where context or order matters.  It comes before Transformers and is used widely in text generation, speech recognition, and time series forecasting (stock price forecast).

## Mathematical Foundation of RNNs

### Core Equations

At each time step $t$, an RNN performs the following operations:

1. **Hidden State Update**:
   $$ h_t = \text{tanh}(W_{hh}h_{t-1} + W_{xh}x_t + b_h) $$
   
   - $h_t$: New hidden state at time $t$ (shape: `[hidden_size]`)
   - $h_{t-1}$: Previous hidden state (shape: `[hidden_size]`)
   - $x_t$: Input at time $t$ (shape: `[input_size]`)
   - $W_{hh}$: Hidden-to-hidden weights (shape: `[hidden_size, hidden_size]`)
   - $W_{xh}$: Input-to-hidden weights (shape: `[hidden_size, input_size]`)
   - $b_h$: Hidden bias term (shape: `[hidden_size]`)
   - $\text{tanh}$: Hyperbolic tangent activation function

2. **Output Calculation**:
   $$o_t = W_{hy}h_t + b_y$$
   
   - $o_t$: Output at time $t$ (shape: `[output_size]`)
   - $W_{hy}$: Hidden-to-output weights (shape: `[output_size, hidden_size]`)
   - $b_y$: Output bias term (shape: `[output_size]`)

### Backpropagation Through Time (BPTT)

RNNs are trained using BPTT, which unrolls the network through time and applies the chain rule:

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^T \frac{\partial L_t}{\partial o_t} \frac{\partial o_t}{\partial h_t} \sum_{k=1}^t \left( \prod_{i=k+1}^t \frac{\partial h_i}{\partial h_{i-1}} \right) \frac{\partial h_k}{\partial W}
$$

This can lead to the vanishing/exploding gradients problem, which is addressed by LSTM and GRU architectures.

## GRU: Gated Recurrent Unit

Before diving into our translation example, let's examine the mathematical foundation of GRUs, which are used in our model. GRUs address the vanishing gradient problem in standard RNNs through gating mechanisms.

### GRU Equations

At each time step $t$, a GRU computes the following:

1. **Update Gate** ($z_t$):
   $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
   
   - $z_t$: Update gate (shape: `[hidden_size]`)
   - $W_z$: Weight matrix for update gate (shape: `[hidden_size, hidden_size + input_size]`)
   - $b_z$: Bias term for update gate (shape: `[hidden_size]`)
   - $h_{t-1}$: Previous hidden state
   - $x_t$: Current input
   - $\sigma$: Sigmoid activation (squashes values between 0 and 1)
   
   The update gate decides how much of the previous hidden state to keep.

2. **Reset Gate** ($r_t$):
   $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
   
   - $r_t$: Reset gate (shape: `[hidden_size]`)
   - $W_r$: Weight matrix for reset gate (shape: `[hidden_size, hidden_size + input_size]`)
   - $b_r$: Bias term for reset gate (shape: `[hidden_size]`)
   
   The reset gate determines how much of the previous hidden state to forget.

3. **Candidate Hidden State** ($\tilde{h}_t$):
   $$\tilde{h}_t = \text{tanh}(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$
   
   - $\tilde{h}_t$: Candidate hidden state (shape: `[hidden_size]`)
   - $W$: Weight matrix for candidate state (shape: `[hidden_size, hidden_size + input_size]`)
   - $b$: Bias term (shape: `[hidden_size]`)
   - $\odot$: Element-wise multiplication (Hadamard product)
   
   This represents the "new" hidden state content that could be used.

4. **Final Hidden State** ($h_t$):
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
   
   - The final hidden state is a combination of the previous hidden state and the candidate state
   - $z_t$ acts as an interpolation factor between old and new information

### Why GRUs Work Well for Translation

1. **Update Gate**
   - In our English-to-Chinese example, this helps decide whether to:
     - Keep the previous context (e.g., maintaining the subject of the sentence)
     - Update with new information (e.g., when encountering a new word)

2. **Reset Gate**
   - Helps forget irrelevant information
   - For example, when translating a new sentence, it can reset the context from the previous sentence

3. **Gradient Flow**
   - The additive update ($+$) in the final hidden state calculation helps preserve gradient flow
   - This is crucial for learning long-range dependencies in translation tasks

##  Toy RNN Example

This simplified example trains an RNN to predict the next character in the word "hello".
1. **Model Definition**: 
   - `nn.RNN` handles the recurrent computation.
   - A fully connected layer (`fc`) maps the hidden state to the output (character predictions).
2. **Data**: 
   - We use "hell" as input and expect "ello" as output (shifting the sequence).
   - Characters are converted to one-hot vectors (e.g., 'h' → [1, 0, 0, 0]).
3. **Training**: 
   - The model learns by minimizing the cross-entropy loss between predicted and target characters.
4. **Prediction**: 
   - After training, the model predicts the next characters.


```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# Hyperparameters
input_size = 4   # Number of unique characters (h, e, l, o)
hidden_size = 8  # Size of the hidden state
output_size = 4  # Same as input_size
learning_rate = 0.01

# Character vocabulary
chars = ['h', 'e', 'l', 'o']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Input data: "hell" to predict "ello"
input_seq = "hell"
target_seq = "ello"

# Convert to one-hot encoding with explicit batch dimension
def to_one_hot(seq):
    tensor = torch.zeros(1, len(seq), input_size)  # [batch_size, seq_len, input_size]
    for t, char in enumerate(seq):
        tensor[0][t][char_to_idx[char]] = 1  # Batch size = 1
    return tensor

# Prepare input and target tensors
input_tensor = to_one_hot(input_seq)  # Shape: [1, 4, 4]
print("Input tensor shape:", input_tensor.shape)
target_tensor = torch.tensor([char_to_idx[ch] for ch in target_seq], dtype=torch.long)  # Shape: [4]

# Initialize the model, loss, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(100):
    hidden = model.init_hidden(1)  # Batch size = 1
    print("Hidden state shape:", hidden.shape)  # Should be [1, 1, 8]
    optimizer.zero_grad()
    output, hidden = model(input_tensor, hidden)  # output: [1, 4, 4], hidden: [1, 1, 8]
    
    loss = criterion(output.squeeze(0), target_tensor)  # output.squeeze(0): [4, 4], target: [4]
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    hidden = model.init_hidden(1)
```

## English-to-Chinese Translation Example

We will build a simple English-to-Chinese translation model using PyTorch's GRU (Gated Recurrent Unit), which is a variant of RNN that handles long-term dependencies better.

### 1. Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample parallel corpus (English -> Chinese)
english_sentences = [
    "hello", "how are you", "i love machine learning",
    "good morning", "artificial intelligence"
]

chinese_sentences = [
    "你好", "你好吗", "我爱机器学习",
    "早上好", "人工智能"
]

# Create vocabulary
eng_chars = sorted(list(set(' '.join(english_sentences))))
zh_chars = sorted(list(set(''.join(chinese_sentences))))

# Add special tokens
SOS_token = 0  # Start of sentence
EOS_token = 1  # End of sentence
eng_chars = ['<SOS>', '<EOS>', '<PAD>'] + eng_chars
zh_chars = ['<SOS>', '<EOS>', '<PAD>'] + zh_chars

# Create word-to-index mappings
eng_to_idx = {ch: i for i, ch in enumerate(eng_chars)}
zh_to_idx = {ch: i for i, ch in enumerate(zh_chars)}

# Convert sentences to tensors
def sentence_to_tensor(sentence, vocab, is_target=False):
    indices = [vocab[ch] for ch in (sentence if not is_target else sentence)]
    if is_target:
        indices.append(EOS_token)  # Add EOS token for target
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)
```

### 2. Model Architecture

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        
        # Encoder (English to hidden)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        # Decoder (hidden to Chinese)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input_seq, hidden=None, max_length=10):
        # Encoder
        embedded = self.embedding(input_seq).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        
        # Decoder
        decoder_input = torch.tensor([[SOS_token]], device=input_seq.device)
        decoder_hidden = hidden
        decoded_words = []
        
        for _ in range(max_length):
            output, decoder_hidden = self.gru(
                self.embedding(decoder_input).view(1, 1, -1),
                decoder_hidden
            )
            output = self.softmax(self.out(output[0]))
            topv, topi = output.topk(1)
            
            if topi.item() == EOS_token:
                break
                
            decoded_words.append(zh_chars[topi.item()])
            decoder_input = topi.detach()
            
        return ''.join(decoded_words), decoder_hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
```

### 3. Training the Model

```python
# Hyperparameters
hidden_size = 256
learning_rate = 0.01
n_epochs = 1000

# Initialize model
model = Seq2Seq(len(eng_chars), hidden_size, len(zh_chars))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    
    for eng_sent, zh_sent in zip(english_sentences, chinese_sentences):
        # Prepare data
        input_tensor = sentence_to_tensor(eng_sent, eng_to_idx)
        target_tensor = sentence_to_tensor(zh_sent, zh_to_idx, is_target=True)
        
        # Forward pass
        model.zero_grad()
        hidden = model.init_hidden()
        
        # Run through encoder
        embedded = model.embedding(input_tensor).view(len(input_tensor), 1, -1)
        _, hidden = model.gru(embedded, hidden)
        
        # Prepare decoder
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = hidden
        loss = 0
        
        # Teacher forcing: use the target as the next input
        for di in range(len(target_tensor)):
            output, decoder_hidden = model.gru(
                model.embedding(decoder_input).view(1, 1, -1),
                decoder_hidden
            )
            output = model.out(output[0])
            loss += criterion(output, target_tensor[di])
            decoder_input = target_tensor[di]
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / len(target_tensor)
    
    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(english_sentences):.4f}')

# Test translation
def translate(sentence):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence.lower(), eng_to_idx)
        output_words, _ = model(input_tensor)
        return output_words

# Example translations
print("\nTranslations:")
print(f"'hello' -> '{translate('hello')}'")
print(f"'how are you' -> '{translate('how are you')}'")
print(f"'i love machine learning' -> '{translate('i love machine learning')}'")
```

### 4. Understanding the Output

After training, the model should be able to translate simple English phrases to Chinese. For example:

- Input: "hello"
  - Output: "你好"
  
- Input: "how are you"
  - Output: "你好吗"
  
- Input: "i love machine learning"
  - Output: "我爱机器学习"

### 5. Key Components Explained

1. **Embedding Layer**:
   - Converts discrete word indices to continuous vectors
   - Captures semantic relationships between words

2. **GRU (Gated Recurrent Unit)**:
   - Controls information flow using update and reset gates
   - Addresses the vanishing gradient problem in standard RNNs

3. **Teacher Forcing**:
   - Uses the target output as the next input during training
   - Helps the model learn the correct translation faster

4. **Beam Search**:
   - Could be implemented for better translation quality
   - Keeps track of multiple possible translations during decoding

### 6. Challenges and Improvements

1. **Handling Variable-Length Sequences**:
   - Use padding and masking
   - Implement attention mechanism for better alignment

2. **Vocabulary Size**:
   - Use subword units (Byte Pair Encoding, WordPiece)
   - Implement pointer-generator networks for rare words

3. **Performance**:
   - Use bidirectional RNNs for better context understanding
   - Implement transformer architecture for parallel processing

This example provides a foundation for sequence-to-sequence learning with RNNs. For production systems, consider using transformer-based models like BART or T5, which have shown superior performance in machine translation tasks.

