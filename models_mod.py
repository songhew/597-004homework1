import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CommonSpace(nn.Module):
    def __init__(self, embed_size):
        super(CommonSpace, self).__init__()
        self.linear = nn.Linear(embed_size, embed_size)
    def forward(self, x):
        # TODO
        # =========================================================
        output = None
        # =========================================================
        return output 

    def get_trainable_parameters(self):
        return list(self.parameters())


class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(DummyImageEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer. add # feature
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(resnet.fc.in_features, momentum=0.01)
        self.relu = nn.ReLU()

    def get_trainable_parameters(self):
        return list(self.bn.parameters()) + list(self.linear.parameters())

    def forward(self, image):

        # TODO
        # =========================================================
        # Feed the image to ResNet but requires not grad (torch.no_grad())
        # We are not going to finetune the ResNet cause it takes too much computational resources
        img_ft = None 

        # Apply batch norm to the image features
        norm_img_ft = None

        # Feed the normed features to the output layer
        out = None

        # Embed to the common space
        out = None
        # =========================================================

        return out


class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, embed_size):
        super(DummyCaptionEncoder, self).__init__()
        self.embed_size = embed_size
        self.out_linear = nn.Linear(embed_size, embed_size, bias=False)
        self.rnn = nn.GRU(vocab_embed_size, embed_size) # GRU is selected as default RNN architecture but you can modify and choose your favourite RNN structure. add expected results
        self.embed = nn.Embedding(vocab_size, vocab_embed_size)

    def forward(self, inputs, lengths):

        # TODO
        # =========================================================
        # First, embed the tokens into word emebeddings
        inputs = None 

        # Do some necessary sorting operation; NO need to modify
        lengths = torch.LongTensor(lengths)
        [_, sort_ids] = torch.sort(lengths, descending=True)
        sorted_input = input[sort_ids]
        sorted_length = lengths[sort_ids]
        reverse_sort_ids = sort_ids.clone()
        for i in range(sort_ids.size(0)):
            reverse_sort_ids[sort_ids[i]] = i

        # Pack the padded sequence with torch.nn.utils.rnn.pack_padded_sequence
        # This step is necessary since we don't want to use a for loop to recurrsively 
        packed = None

        # Feed your input into RNN
        output, hidden = None, None

        # Pad packed sequence with torch.nn.utils.rnn.pad_packed_sequence
        padded, output_length = None, None

        # Do some necessary sorting and stack operation; NO need to modify
        output = [padded[output_length[i]-1, i, :] for i in range(len(output_length))]
        output = torch.stack([output[reverse_sort_ids[i]] for i in range(len(output))], dim=0)

        # Feed output into the final linear layer
        output = None

        # Embed to the common space
        output = None

        # =========================================================

        return outputs

    def get_trainable_parameters(self):
        return list(self.parameters())

