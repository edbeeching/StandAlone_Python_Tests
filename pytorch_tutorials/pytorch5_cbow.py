# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:37:22 2017

@author: Edward
"""

from torch.nn import Module, Linear, Embedding, NLLLoss
import torch.nn.functional as F
from torch.optim import SGD
import torch
from torch.functional import stack
from torch import autograd

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word:i for i, word in enumerate(vocab)}

data = []
for i in range(CONTEXT_SIZE, len(raw_text)-CONTEXT_SIZE):
    context = [raw_text[j] for j in range(i-CONTEXT_SIZE, i+CONTEXT_SIZE+1) if j!=i]
    target = raw_text[i]
    data.append((context, target))
    
    
    
class CBOW(Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = Embedding(vocab_size, embedding_dim)
        self.linear1 = Linear(embedding_dim, 128)
        self.linear2 = Linear(128, vocab_size)
        
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        #print(embeds.size())
        sums = torch.sum(embeds,0).view(1,-1)
        #print(sums.size())
        out = F.relu(self.linear1(sums))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs
    
    
model = CBOW(len(vocab), EMBEDDING_DIM)

#example_context = [word_to_ix[word] for word in data[0][0]]
#example_vars = autograd.Variable(torch.LongTensor(example_context))
#
#model.forward(example_vars)
losses = []
loss_function =  NLLLoss()
optimizer =SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = torch.Tensor([0])
    for context, target in data:
        example_context = [word_to_ix[word] for word in context]
        example_vars = autograd.Variable(torch.LongTensor(example_context))

        model.zero_grad()
        log_probs = model(example_vars)
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        
    losses.append(total_loss)

print(losses)
        
        



















                
        
        