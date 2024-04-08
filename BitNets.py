import torch
import torch.nn as nn
from torch.nn import functional as F

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, activation_bits = 8, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.epsilon = 1e-5
        self.act_range = 2 ** (activation_bits - 1)
        self.norm = nn.LayerNorm(in_features)

        self.register_buffer("quantized_weights", torch.sign(self.weight.data).to(torch.int8))
        del self.weight

    def dequantize_weights(self):
        weight_bf16 = self.quantized_weights.to(torch.float32)
        alpha = weight_bf16.mean()
        return weight_bf16 * alpha

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)

    def sign_binarize(self, x):
        # STE to allow gradient flow through non-differentiable operation
        return torch.sign(x).detach() + x - x.detach()

    
    def forward(self, input):
        # Normalize input
        x = self.norm(input)
        
        # Binarize weights
        weights = self.dequantize_weights()
        alpha = weights.mean()
        binarized_weights = self.sign_binarize(weights - alpha)

        # Perform the linear transformation with binarized weights
        output = F.linear(x, binarized_weights, self.bias)
        
        # Quantize the output activations 
        gamma_g = output.abs().max()
        scale = self.act_range / (gamma_g + self.epsilon)
        output = torch.clamp(output * scale, min= (-(self.act_range)+ self.epsilon), max=(self.act_range + self.epsilon))

        return output

