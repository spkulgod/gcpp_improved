import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import torch.nn.functional as F
from scipy.stats import multivariate_normal

####################################################
# Seq2Seq LSTM AutoEncoder Model
# 	- predict locations
####################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
        #input [7680, T, 64] , [  N*V, T, 64]  #HIDDEN 2 X 7680 X 60
    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, num_vehicles, dropout=0.5, isCuda=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size*num_vehicles
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size*30, self.output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        # decoded_output = self.tanh(decoded_output)
        # decoded_output = self.sigmoid(decoded_output)
        decoded_output = self.dropout(decoded_output)
        # decoded_output = self.tanh(self.linear(decoded_output))
        decoded_output = self.linear(decoded_output)
        # decoded_output = self.sigmoid(self.linear(decoded_output))
        return decoded_output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_vehicles, dropout=0.5, isCuda=True, num_traj = 5):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        # self.pred_length = pred_length
        print("hidden size",hidden_size)
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size+1, num_layers, num_traj, dropout, isCuda)
        self.num_traj = num_traj

    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size  #2     
        self.pred_length = pred_length
        
        print("output dim value", out_dim)
        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location
        
        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        if self.isCuda:
            outputs = outputs.cuda()
            
        for t in range(self.pred_length):
        # encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:,t:t+1] = now_out #UPDATE FOR TIMESTEPS
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
                # decoder_input = now_out
        return outputs,mean,std_dev,prob_list

###################################################
###################################################

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np 

# ####################################################
# # Seq2Seq LSTM AutoEncoder Model
# # 	- predict locations
# ####################################################
# class EncoderRNN(nn.Module):
# 	def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
# 		super(EncoderRNN, self).__init__()
# 		self.input_size = input_size
# 		self.hidden_size = hidden_size
# 		self.num_layers = num_layers
# 		self.isCuda = isCuda
# 		# self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
# 		self.lstm = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
		
# 	def forward(self, input):
# 		output, hidden = self.lstm(input)
# 		return output, hidden

# class DecoderRNN(nn.Module):
# 	def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
# 		super(DecoderRNN, self).__init__()
# 		self.hidden_size = hidden_size
# 		self.output_size = output_size
# 		self.num_layers = num_layers
# 		self.isCuda = isCuda
# 		# self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
# 		self.lstm = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

# 		#self.relu = nn.ReLU()
# 		self.sigmoid = nn.Sigmoid()
# 		self.dropout = nn.Dropout(p=dropout)
# 		self.linear = nn.Linear(output_size*30, output_size)
# 		self.tanh = nn.Tanh()
	
# 	def forward(self, encoded_input, hidden):
# 		decoded_output, hidden = self.lstm(encoded_input, hidden)
# 		# decoded_output = self.tanh(decoded_output)
# 		# decoded_output = self.sigmoid(decoded_output)
# 		decoded_output = self.dropout(decoded_output)
# 		# decoded_output = self.tanh(self.linear(decoded_output))
# 		decoded_output = self.linear(decoded_output)
# 		# decoded_output = self.sigmoid(self.linear(decoded_output))
# 		return decoded_output, hidden

# class Seq2Seq(nn.Module):
# 	def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=True):
# 		super(Seq2Seq, self).__init__()
# 		self.isCuda = isCuda
# 		# self.pred_length = pred_length
# 		self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
# 		self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)
	
# 	def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
# 		batch_size = in_data.shape[0]
# 		out_dim = self.decoder.output_size
# 		self.pred_length = pred_length

# 		outputs = torch.zeros(batch_size, self.pred_length, out_dim)
# 		if self.isCuda:
# 			outputs = outputs.cuda()

# 		encoded_output, hidden = self.encoder(in_data)
# 		decoder_input = last_location
# 		for t in range(self.pred_length):
# 			# encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
# 			now_out, hidden = self.decoder(decoder_input, hidden)
# 			now_out += decoder_input
# 			outputs[:,t:t+1] = now_out 
# 			teacher_force = np.random.random() < teacher_forcing_ratio
# 			decoder_input = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
# 			# decoder_input = now_out
# 		return outputs

# ####################################################
# ####################################################