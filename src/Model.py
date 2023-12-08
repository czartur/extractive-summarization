import torch
from copy import deepcopy

class MLP_FT(torch.nn.Module):
    def __init__(self, base_model, params):
        super(MLP_FT, self).__init__()
        self.base_model = deepcopy(base_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.mlp = self.MLP(params)

    def forward(self, sample):
        seq = sample['sequence']
        mask = sample['attention_mask']
        speakers = sample['speaker']
        in_degrees = sample['in_degree'] 
        out_degrees = sample['out_degree']
        lengths = sample['length']

        # language model pass
        outputs = self.base_model(seq, attention_mask=mask)
        hidden_states = outputs.last_hidden_state
        x = hidden_states[:,0,:]

        # MLP pass
        x = self.dropout(x)
        x = torch.cat((x, in_degrees, out_degrees, lengths), dim = 1)
        # x = torch.cat((x, speakers, in_degrees, out_degrees, lengths), dim=1)
        # x = torch.cat((x, speakers, lengths), dim=1)
        x = self.mlp(x) 
        return x
    
    # MLP factory from parameters
    def MLP(self, params : dict) -> torch.nn.Sequential:
        n_layers = params['n_layers']
        layers = []

        in_features = params['input_size']
        for i in range(n_layers):
            out_features = params[f'n_{i}_size']
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            
            # dropout
            p = params['n_p']
            layers.append(torch.nn.Dropout(p))

            # updating next layer size
            in_features = out_features
            
        layers.append(torch.nn.Linear(in_features, params['output_size']))
        model = torch.nn.Sequential(*layers)
        return model