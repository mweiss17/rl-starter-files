import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class NAC(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()
    self.W_hat = nn.Parameter(torch.Tensor(n_out, n_in))
    self.M_hat = nn.Parameter(torch.Tensor(n_out, n_in))
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.W_hat)
    nn.init.kaiming_uniform_(self.M_hat)

  def forward(self, input):
    weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
    return F.linear(input, weights)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_number=False, use_nac=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_number = use_number
        self.use_memory = use_memory
        self.use_nac = use_nac
        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.hidden_layer_size = 64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        if self.use_number:
            self.embedding_size += 1


        if self.use_nac:
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, self.hidden_layer_size),
                nn.Tanh(),
                NAC(self.hidden_layer_size, 1)
            )
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, self.hidden_layer_size),
                nn.Tanh(),
                NAC(self.hidden_layer_size, action_space.n)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, self.hidden_layer_size),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_size, action_space.n)
            )
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, self.hidden_layer_size),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_size, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        if self.use_number:
            embedding = torch.cat((embedding, obs.numbers), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class ACMLPModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self.image_embedding_size = n * m
        self.embedding_size = self.semi_memory_size + 1
        self.hidden_layer_size = self.embedding_size

        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_layer_size),
            nn.Tanh(),
        )
        self.critic_nac = NAC(2 * self.hidden_layer_size, 1)

        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_layer_size),
            nn.Tanh(),
        )
        self.actor_nac = NAC(2 * self.hidden_layer_size, action_space.n)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size


    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x[:, -1]
        x = x.reshape(x.shape[0], -1)
        embedding = torch.cat((x, obs.numbers), dim=1)

        x = self.actor(embedding)
        x = torch.cat((embedding, x), dim=1) # skip connection
        x = self.actor_nac(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        x = torch.cat((embedding, x), dim=1) # skip connection
        x = self.critic_nac(x)

        value = x.squeeze(1)

        return dist, value, memory



class ACNACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self.image_embedding_size = n * m
        self.embedding_size = self.semi_memory_size + 1
        self.hidden_layer_size = 64

        self.critic = nn.Sequential(
            NAC(self.embedding_size, 1)
        )
        self.actor = nn.Sequential(
            NAC(self.embedding_size, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size


    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x[:, -1]
        x = x.reshape(x.shape[0], -1)
        embedding = torch.cat((x, obs.numbers), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
