import torch.nn as nn

class NeuralNetClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size1 = 512
        hidden_size2 = 256
        hidden_size3 = 128
        hidden_size4 = 64
        super(NeuralNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size4, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation function

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return self.softmax(x)

