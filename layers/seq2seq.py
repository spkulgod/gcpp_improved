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

class VAE(nn.Module):
    def __init__(self,io_size,latent_size,num_vehicles,isCuda=True):
        super(VAE,self).__init__()
        self.latent_size = latent_size
        self.io_size = io_size
        self.isCuda = isCuda
        self.norm = multivariate_normal(mean=np.zeros(latent_size), cov=np.eye(latent_size))
        self.num_vehicles = num_vehicles
        
        self.enc1 = nn.Linear(io_size, 60)
        self.mean = nn.Linear(60, latent_size)
        self.std = nn.Linear(60, latent_size)
        self.dec1 = nn.Linear(latent_size, 60)
        self.dec2 = nn.Linear(60, io_size)
    
    def encode(self, x):
        h1 = F.relu(self.enc1(x))
        return self.mean(h1), self.std(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
#         print("shape of eps",eps.shape)
        prob = self.norm.pdf(eps.detach().cpu().numpy())
#         print("shape of eps",prob.shape)
        return mu + eps*std, prob

    def decode(self, z):
        h3 = F.relu(self.dec1(z))
        h3 = self.dec2(h3).reshape(h3.shape[0],self.hidden_layers,-1)
        return h3
    
    def forward(self,h_input,num_traj):
        self.hidden_layers = h_input.shape[1]
        mu, logvar = self.encode(h_input.reshape(-1,self.io_size))
        multiple_traj = []
        prob_list=[]
        for i in range(num_traj): 
            z ,prob= self.reparameterize(mu, logvar)
            multiple_traj.append(self.decode(z).permute(1,0,2).contiguous())
            prob_list.append(prob)
        prob_list = np.exp(np.array(prob_list))
        prob_list = prob_list/np.sum(prob_list,axis=0)
        prob_list = prob_list.reshape((prob_list.shape[0],-1,self.num_vehicles))
        prob_list = torch.tensor(prob_list).to(torch.device("cuda:0"))
        return multiple_traj, mu, logvar, prob_list
        
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        # self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.lstm = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size*30, output_size)
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
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)
        self.vae = VAE(30*hidden_size*num_layers, 20, num_vehicles, isCuda)
        self.num_traj = num_traj

    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size  #2     
        self.pred_length = pred_length
        
        encoded_output, hidden = self.encoder(in_data)
        hidden_list,mean,std_dev,prob_list=self.vae(hidden.permute(1,0,2),self.num_traj)
#         hidden=hidden.permute(1,0,2).contiguous()
#         mean = 0
#         std_dev = 0
        decoder_input = last_location
        output_trajs = torch.tensor(np.zeros((self.num_traj, batch_size, self.pred_length, out_dim))).to(torch.device("cuda:0"))
        
        for i in range(self.num_traj):
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
            output_trajs[i] = outputs
        return output_trajs,mean,std_dev,prob_list

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