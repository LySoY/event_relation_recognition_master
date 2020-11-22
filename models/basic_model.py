from nn.encoder import BiEncoder as BE
from nn.encoder import DotEncoder as DE
from process_control import label_from_output
from my_loss_functions import *


# 事件关系分类网络定义
class RelClassifyModel(nn.Module):
    def __init__(self, config, args):
        super(RelClassifyModel, self).__init__()
        self.tagset_size = len(args.rel2label)
        self.encoder1 = BE(config.hidden_size,
                           args.max_sent_len,
                           config.num_hidden_layers,
                           config.num_attention_heads,
                           1)
        self.encoder2 = BE(config.hidden_size,
                           args.max_sent_len,
                           config.num_hidden_layers,
                           config.num_attention_heads,
                           1)
        self.encoder = DE(config.hidden_size,
                          config.intermediate_size,
                          config.num_hidden_layers,
                          config.num_attention_heads)
        self.rnn = nn.GRU(config.hidden_size,
                          config.hidden_size//2,
                          batch_first=True,
                          num_layers=1,
                          bidirectional=True)
        self.fc1 = nn.Linear(in_features=config.hidden_size,
                             out_features=1)
        self.fc2 = nn.Linear(in_features=args.max_sent_len,
                             out_features=self.tagset_size)
        self.soft = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(0.5)
        self.loss1 = CrossEntropyLoss()
        self.loss2 = SoftenLoss(self.tagset_size)

    def set_loss_device(self, device):
        self.loss1.to(device)
        self.loss2.to(device)

    def load(self, output_model_file):
        model_state_dict = torch.load(output_model_file)
        self.load_state_dict(model_state_dict)

    def save(self, output_model_file):
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), output_model_file)

    def get_acc(self, x, y):
        is_right = 0
        size = y.size()[0]
        for i in range(size):
            try:
                if y[i] == label_from_output(x[i]):
                    is_right += 1
            except:
                continue
        return is_right / size

    def one_hot(self, y):
        size = y.size()[0]
        label = torch.zeros(size, self.tagset_size).to(y.device)
        for i in range(size):
            for j in range(self.tagset_size):
                try:
                    label[i][int(y[i])] = 1
                except:
                    label[i][self.tagset_size - 1] = 1
        return label

    def test(self, x1, x2, y):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.encoder(x1, x2)
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.fc2(x)
        x = self.soft(x)
        acc = self.get_acc(x, y)
        return acc

    def get_guess(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.encoder(x1, x2)
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.fc2(x)
        x = self.soft(x)
        return x

    def forward(self, x1, x2, y):
        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x = self.encoder(x1, x2)
        x, _ = self.rnn(x)
        x = self.fc1(x)
        x = x.squeeze(-1)
        x = self.fc2(x)
        x = self.soft(x)
        acc = self.get_acc(x, y)
        y = self.one_hot(y)
        return 0.8*self.loss1(x, y) + 0.2*self.loss2(x), acc
