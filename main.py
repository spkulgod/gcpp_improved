import argparse
import os 
import sys
import numpy as np 
import torch
import torch.optim as optim
from model import Model
from xin_feeder_baidu import Feeder
from datetime import datetime
import random
import itertools

CUDA_VISIBLE_DEVICES='0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

max_x = 1. 
max_y = 1. 
history_frames = 6 # 3 second * 2 frame/second
future_frames = 12 # 6 second * 2 frame/second 

batch_size_train = 64
batch_size_val = 4
batch_size_test = 1
total_epoch = 30
base_lr = 0.01
lr_decay_epoch = 5
dev = torch.device("cuda:0")
# dev=device_("cuda" if cuda.is_available() else "cpu")
work_dir = './trained_models_temp'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

criterion = torch.nn.SmoothL1Loss()

min_it = np.infty
min_itk1 = np.infty
min_itk2 = np.infty
min_ade = np.infty
min_adek1 = np.infty
min_adek2 = np.infty

min_fde = np.infty

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

def my_print(pra_content):
    with open(log_file, 'a') as writer:
        print(pra_content)
        writer.write(pra_content+'\n')

def display_result(pra_results, pra_pref='Train_epoch'):
    if(pra_pref == 'miss_rate'):
        all_overall_miss_list,all_overall_fde_num_list= pra_results
        all_overall_miss_sum = np.sum(all_overall_miss_list, axis=0) #t
        overall_fde_num_time = np.sum(all_overall_fde_num_list, axis=0) #
        miss_rate = all_overall_miss_sum/overall_fde_num_time
        my_print("\n overall miss rate:={}".format(miss_rate))
        return miss_rate
    global min_it,min_ade,min_itk1,min_adek1,min_itk2,min_adek2,min_fde
    all_overall_sum_list, all_overall_num_list, all_overall_ade_list,all_overall_fde_list,all_overall_fde_num_list,iteration = pra_results
    overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0) #  t
    overall_num_time = np.sum(all_overall_num_list, axis=0) # t 
    overall_ade_time = np.sum(all_overall_ade_list, axis=0) #t
    overall_fde_time = np.sum(all_overall_fde_list, axis=0) #t
    overall_fde_num_time = np.sum(all_overall_fde_num_list, axis=0) #t

#     overall_loss_time = (overall_sum_time / (overall_num_time+1000))
    print("overall_num_time",overall_num_time)
    print("overall_fde_num",overall_fde_num_time)
    overall_ade = (overall_ade_time / (overall_num_time))
    overall_fde = (overall_fde_time / (overall_fde_num_time))

#     overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
#     my_print(overall_log)
    my_print('ADE={}'.format(overall_ade))
    mean_ade = overall_ade
    if(pra_pref=='car'):
        if(mean_ade<min_ade):
            min_ade = mean_ade
            min_it = iteration
        my_print('ADE mean={}'.format(mean_ade))
        my_print('ADE min={},  Iteration ={}'.format(min_ade,min_it))
    if(pra_pref=='car ade k1'):
        if(mean_ade<min_adek1):
            min_adek1 = mean_ade
            min_itk1 = iteration
        my_print('ADE mean={}'.format(mean_ade))
        my_print('ADE mink1={},  Iteration ={}'.format(min_adek1,min_itk1))
    if(pra_pref=='car ade k2'):
        if(mean_ade<min_adek2):
            min_adek2 = mean_ade
            min_itk2 = iteration
        my_print('ADE mean={}'.format(mean_ade))
        my_print('ADE mink2={},  Iteration ={}'.format(min_adek2,min_itk2))
    if(overall_fde<min_fde):
        min_fde = overall_fde
    my_print('FDE ={}'.format(overall_fde))
    my_print('FDE_min={}'.format(min_fde))
    return overall_ade, overall_fde



# def display_result_multi(pra_results, pra_pref='Train_epoch'):
#     global min_it,min_ade
#     all_overall_sum_list, all_overall_num_list, all_overall_ade_list,iteration = pra_results
#     overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0) #  t
#     overall_num_time = np.sum(all_overall_num_list, axis=0) # t 
#     overall_ade_time =  np.sum(all_overall_ade_list, axis=0) #t
#     overall_ade_batch=np.sum(all_overall_ade_list,

#     overall_loss_time = (overall_sum_time / (overall_num_time+1000))
#     overall_ade_tim = (overall_ade_time / (overall_num_time+1000))

#     overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
#     my_print(overall_log)
#     my_print('ADE={}'.format(overall_ade_tim))
#     mean_ade = np.mean(overall_ade_tim)
#     if(mean_ade<min_ade):
#         min_ade = mean_ade
#         min_it = iteration
#     my_print('ADE mean={}'.format(mean_ade))
#     my_print('ADE min={},  Iteration ={}'.format(min_ade,min_it))
#     my_print('FDE={}'.format(overall_ade_tim[-1]))
#     return overall_loss_time



def my_save_model(pra_model, pra_epoch):
    path = '{}/model_epoch_{:04}.pt'.format(work_dir, pra_epoch)
    torch.save(
        {
            'xin_graph_seq2seq_model': pra_model.state_dict(),
        }, 
        path)
    print('Successfull saved to {}'.format(path))


def my_load_model(pra_model, pra_path):
    checkpoint = torch.load(pra_path)
    pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
    print('Successfull loaded from {}'.format(pra_path))
    return pra_model


def data_loader(pra_path, pra_img_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
    feeder = Feeder(data_path=pra_path, img_path = pra_img_path, graph_args=graph_args, train_val_test=train_val_test)
    loader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=pra_batch_size,
        shuffle=pra_shuffle,
        drop_last=pra_drop_last, 
        num_workers=5,
        )
    return loader

def preprocess_data(pra_data, pra_rescale_xy):
    # pra_data: (N, C, T, V)
    # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]	
    feature_id = [3, 4, 9, 10] # [x, y, heading, mask]
    ori_data = pra_data[:,feature_id].detach()
    data = ori_data.detach().clone() 
# 	print('data shape',data[:, :2, 1:]!=0)

    new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) #when x and y exist for the 
# 	print('newmask',new_mask.shape)
    data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float() #computing velocity and applying mask
# 	print(data[0,:2,1],'sanity for 2 dimensions of data')
    data[:, :2, 0] = 0	  

    # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
    object_type = pra_data[:,2:3]

    data = data.float().to(dev)
    ori_data = ori_data.float().to(dev)
    object_type = object_type.to(dev) #type
    data[:,:2] = data[:,:2] / pra_rescale_xy

    return data, ori_data, object_type

def compute_RMSE(pra_pred, pra_GT, pra_mask, pra_error_order=2):
    pred = pra_pred * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    GT = pra_GT * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
    x2y2 = torch.sum(torch.abs(pred - GT)**pra_error_order, dim=1) # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 6, 120)

    overall_sum_time = x2y2.sum(dim=-1) # (N, T, V) -> (N, T)=(N, 6)
    overall_mask = pra_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 6)
    overall_num = overall_mask 

    ade_sum=torch.sqrt(x2y2) # (N,T,V)
    ade_sum=torch.sum(ade_sum,dim=-1) # (N,6)

    return overall_sum_time, overall_num, x2y2,ade_sum

def compute_RMSE_multi(pra_pred, pra_GT, pra_mask,probabilities,pra_error_order=2, train=True):
    GT = pra_GT * pra_mask # (N, C, T, V)=(N, 2, 6, 120) 
    min_rmse = np.inf
    min_prob = 1
    k1 = 5
    k2 = 10
            
    overall_mask = pra_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 6)
    overall_num = torch.max(torch.sum(overall_mask), torch.ones(1,).to(dev)) 
    prob_mat = probabilities*pra_mask[:,0,0,:] #( Traj, batch, vehicles) ( 5, 128, 50)
    prob_max=torch.argsort(prob_mat, dim=0, descending=True) # 5 x 128 x 50
#     print('argsorted',)
#     print('prob max shape',prob_max.shape)
#     print('prob all traj one batch one veh',prob_mat[:,1,0])
    
    pred = pra_pred*pra_mask[:,:,:,:] # (M,N, C, T, V)=(5,N, 2, 12, 120)
#     print('mask-shape',pra_mask.shape)  #128 1 12 50
#     print('pred-shape',pred.shape)  # 5 128 2 12 50
    x2y2 = torch.sum(torch.abs(pred - GT)**pra_error_order, dim=2) # x^2+y^2, (M,N, C, T, V)->(M,N, T, V)=(5,N, 12, 120)
    rmse_mat = x2y2.sum(dim=-2) # (M,N, T, V) -> (M,N,V)
    
    if not train:
        ade_sqrt=torch.sqrt(x2y2)  # (5,N, 12, 50)
        
        dist_max,_ = torch.max(ade_sqrt,dim=2) #(5,N,50)
        miss_num = torch.sum((dist_max[0]>0).float())
        miss_bool = (dist_max>2).float()
        miss_avg = torch.mean(miss_bool,dim=0) #(N,50)
        miss_sum = torch.sum(miss_avg)
        
        fde_mat_full = ade_sqrt[:,:,-1,:] #5,N,50
        fde_min = fde_mat_full.gather(0,prob_max[0].view(-1,prob_max[0].shape[0],prob_max[0].shape[1]))
#         fde_min,_ = torch.min(fde_mat_full,dim=0) # N 50
#         print("fde min shape",fde_min.shape)
        fde_batch_sum = fde_min.sum()

    #     ade_sqrt_prob=ade_sqrt.permute(2,0,1,3)*prob_mat #(12,5,N,50)
        ade_sum_time=ade_sqrt.sum(dim=2) # (5 N 50)
        
        ade_sorted_time = torch.ones_like(ade_sum_time)
        for i in range(ade_sum_time.shape[0]):
            ade_sorted_time[i] = ade_sum_time.gather(0,prob_max[i].view(-1,prob_max[i].shape[0],prob_max[i].shape[1]))
            
        ade_sorted_topk1=ade_sorted_time[:k1,:,:]
        ade_sorted_topk2=ade_sorted_time[:k2,:,:]
#         print('topk1',ade_sorted_topk1.shape)
#         print('topk2',ade_sorted_topk2.shape)
        
        ade_sum_min,_ = torch.min(ade_sum_time,dim=0) #(N,50)
        ade_batch_sum = ade_sum_min.sum()
        
        ade_sum_min_k1,_ = torch.min(ade_sorted_topk1,dim=0) #(N,50)
        ade_batch_sum_k1 = ade_sum_min_k1.sum()
        
        ade_sum_min_k2,_ = torch.min(ade_sorted_topk2,dim=0) #(N,50)
        ade_batch_sum_k2 = ade_sum_min_k2.sum()
        
        return x2y2,  ade_batch_sum.unsqueeze(0), ade_batch_sum_k1.unsqueeze(0),  ade_batch_sum_k2.unsqueeze(0), fde_batch_sum.unsqueeze(0), overall_num.unsqueeze(0), torch.sum(overall_mask[:,-1]).unsqueeze(0), miss_sum.unsqueeze(0), miss_num.unsqueeze(0)
    
#     ade_sum_traj=ade_sqrt_prob.sum(dim=1) # (12,N,50)
#     ade_sum_vehicles=ade_sum_traj.sum(dim=2).permute(1,0) # (12,N) -> (N,12)
                             
    
#     weighted_ade=ade_sum_time*prob_mat  # ( 5 N 50)
#     sum_traj_ade=weighted_ade.sum(dim=0) #( N 50)
#     sum_ade= sum_traj_ade.sum() # one value # mean shd be taken later for n*50 elements
#     print(sum_ade)
#     print('mean ade',sum_ade)
    
    min_args = torch.argmin(rmse_mat,dim=0)  #  (N,V)
#     print("min args shape",min_args.shape)
    rmse_mat = rmse_mat.gather(0,min_args.view(-1,min_args.shape[0],min_args.shape[1]))
    prob_mat = prob_mat.gather(0,min_args.view(-1,min_args.shape[0],min_args.shape[1]))
    min_rmse = torch.sum(rmse_mat)/overall_num
    min_prob = torch.sum(prob_mat)/overall_num
    
#     min_prob = 1
    return min_rmse, min_prob

def train_model(pra_model, pra_data_loader, pra_optimizer, pra_epoch_log):
    # pra_model.to(dev)
    my_print('{}'.format(pra_epoch_log))
    pra_model.train()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y

    # train model using training data
    for iteration, (ori_data, A, _, img) in enumerate(pra_data_loader):
        # print(iteration, ori_data.shape, A.shape)
        # ori_data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)
        mid = int(data.shape[-2]/2)
        #print("A shape1 ::",A.shape)
        for now_history_frames in range(1,data.shape[-2]):  ### put just a no.
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)  #future-gt
            output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev) # shape(N,3,120,120)
            img = img.float().to(dev)

            #print("A shape::",A.shape,"input shape",input_data.shape)
            predicted, prob_list = pra_model(pra_x=input_data, pra_A=A, pra_img = img, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
#             print(prob_list[:,0,0])
#             print("does a nan exist in predicted",torch.any(torch.isnan(predicted)))

            ########################################################
            # Compute loss for training
            ########################################################
            # We use abs to compute loss to backward update weights
            # (N, T), (N, T)
#             min_rmse, min_prob, x2y2, ade_batch_sum, fde_batch_sum, overall_num ,torch.sum(overall_mask[:,-1])
#             print('predicted',predicted[:,0,:,0,0])
            min_rmse,prob = compute_RMSE_multi(predicted, output_loc_GT, output_mask, prob_list,pra_error_order=1)
#             KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            p_loss = -torch.log(prob+0.0001) 
            # overall_loss
            total_loss = min_rmse + p_loss#(1,)
#             print(["min_rmse:{},p_loss:", p_loss,"  total_loss:", total_loss)
            
            if(torch.isnan(total_loss)):
                sys.exit()
            
            
            now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
            print('|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), pra_epoch_log, iteration, total_loss.data.item(),now_lr))

            pra_optimizer.zero_grad()
            total_loss.backward()
            pra_optimizer.step()

def val_model(pra_model, pra_data_loader,train_it):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    all_overall_ade_list = []
    all_overall_fde_list=[]
    all_overall_fde_num_list=[]
    all_overall_miss_list = []


    all_car_sum_list = []
    all_car_ade_list = []
    all_car_adek1_list = []
    all_car_adek2_list = []
    all_car_num_list = []
    all_car_fde_list=[]
    all_car_fde_num_list=[]
    all_car_miss_list = []
    all_car_miss_num = []

    # train model using training data
    for iteration, (ori_data, A, _, img) in enumerate(pra_data_loader):
        # data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

        for now_history_frames in range(6, 7):
            input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
            output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

            ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
            ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:]

            # for category
            cat_mask = ori_data[:,2:3, now_history_frames:, :] # (N, C, T, V)=(N, 1, 6, 120)

            A = A.float().to(dev)
            img = img.float().to(dev)
            
            predicted, prob_list = pra_model(pra_x=input_data, pra_A=A, pra_img = img, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
#             print("does a nan exist in predicted",torch.any(torch.isnan(predicted)))
#             torch.where(predicted==nan,0,predicted)
            ########################################################
            # Compute details for training
            ########################################################
            
            for i in range(predicted.shape[0]):
                predicted[i] = predicted[i]*rescale_xy
                # output_loc_GT = output_loc_GT*rescale_xy
                for ind in range(1, predicted[i].shape[-2]):
                    predicted[i][:,:,ind] = torch.sum(predicted[i][:,:,ind-1:ind+1], dim=-2)
                predicted[i] += ori_output_last_loc  #traj
#             print("predicted.shape",predicted.shape)
 
            ### overall dist
            
# min_rmse, min_prob, x2y2, ade_batch_sum, fde_batch_sum, overall_num ,torch.sum(overall_mask[:,-1])
            x2y2,overall_ade,_,_,overall_fde,overall_num,fde_num,miss_sum,_ = compute_RMSE_multi(predicted, ori_output_loc_GT, output_mask,prob_list,2,False)		
            
            # all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
            all_overall_num_list.extend(overall_num.detach().cpu().numpy())
            all_overall_ade_list.extend(overall_ade.detach().cpu().numpy())
            all_overall_fde_list.extend(overall_fde.detach().cpu().numpy())
            all_overall_fde_num_list.extend(fde_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            now_x2y2 = x2y2.detach().cpu().numpy()
            now_x2y2 = now_x2y2.sum(axis=-1)
            all_overall_sum_list.extend(now_x2y2)
#             all_overall_ade_list.extend(ade_one_val.detach().cpu().numpy())

            ### car dist
            car_mask = (((cat_mask==1)+(cat_mask==2))>0).float().to(dev)
            car_mask = output_mask * car_mask
            
            
            # min_rmse, min_prob, x2y2, ade_batch_sum, fde_batch_sum, overall_num ,torch.sum(overall_mask[:,-1])

            car_x2y2, car_ade,car_adek1,car_adek2,car_fde,car_num,car_fde_mask,car_miss_sum,car_miss_num = compute_RMSE_multi(predicted, ori_output_loc_GT, car_mask,prob_list,2,False)		
            all_car_num_list.extend(car_num.detach().cpu().numpy())
            all_car_ade_list.extend(car_ade.detach().cpu().numpy())
            all_car_adek1_list.extend(car_adek1.detach().cpu().numpy())
            all_car_adek2_list.extend(car_adek2.detach().cpu().numpy())
            all_car_fde_list.extend(car_fde.detach().cpu().numpy())
            all_car_fde_num_list.extend(car_fde_mask.detach().cpu().numpy())
            all_car_miss_list.extend(car_miss_sum.detach().cpu().numpy())
            all_car_miss_num.extend(car_miss_num.detach().cpu().numpy())
            # x2y2 (N, 6, 39)
            car_x2y2 = car_x2y2.detach().cpu().numpy()
            car_x2y2 = car_x2y2.sum(axis=-1)
            all_car_sum_list.extend(car_x2y2)


    result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list) , np.array(all_car_ade_list),np.array(all_car_fde_list),np.array(all_car_fde_num_list), train_it ], pra_pref='car')
    result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list) , np.array(all_car_adek1_list),np.array(all_car_fde_list),np.array(all_car_fde_num_list), train_it ], pra_pref='car ade k1')
    result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list) , np.array(all_car_adek2_list),np.array(all_car_fde_list),np.array(all_car_fde_num_list), train_it ], pra_pref='car ade k2')
    result_car = display_result([np.array(all_car_miss_list),np.array(all_car_miss_num)], pra_pref='miss_rate')


    all_overall_sum_list = np.array(all_overall_sum_list)
    all_overall_num_list = np.array(all_overall_num_list)
    all_overall_ade_list = np.array(all_overall_ade_list)
#     all_overall_adek1_list = np.array(all_overall_adek2_list)
#     all_overall_adek2_list = np.array(all_overall_adek1_list)
    all_overall_fde_list = np.array(all_overall_fde_list)
    all_overall_fde_num_list = np.array(all_overall_fde_num_list)


    return all_overall_sum_list, all_overall_num_list, all_overall_ade_list, all_overall_fde_list, all_overall_fde_num_list, train_it



def test_model(pra_model, pra_data_loader):
    # pra_model.to(dev)
    pra_model.eval()
    rescale_xy = torch.ones((1,2,1,1)).to(dev)
    rescale_xy[:,0] = max_x
    rescale_xy[:,1] = max_y
    all_overall_sum_list = []
    all_overall_num_list = []
    with open(test_result_file, 'w') as writer:
        # train model using training data
        for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
            # data: (N, C, T, V)
            # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
            data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
            input_data = data[:,:,:history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
            output_mask = data[:,-1,-1,:] # (N, V)=(N, 120)
            # print(data.shape, A.shape, mean_xy.shape, input_data.shape)

            ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:]

            A = A.float().to(dev)
            predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
            predicted = predicted *rescale_xy 

            for ind in range(1, predicted.shape[-2]):
                predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
            predicted += ori_output_last_loc

            now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
            now_mean_xy = mean_xy.detach().cpu().numpy() # (N, 2)
            now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
            now_mask = now_ori_data[:, -1, -1, :] # (N, V)

            now_pred = np.transpose(now_pred, (0, 2, 3, 1)) # (N, T, V, 2)
            now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1)) # (N, T, V, 11)

            # print(now_pred.shape, now_mean_xy.shape, now_ori_data.shape, now_mask.shape)

            for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
                # (6, 120, 2), (2,), (6, 120, 11), (120, )
                num_object = np.sum(n_mask).astype(int)
                # only use the last time of original data for ids (frame_id, object_id, object_type)
                # (6, 120, 11) -> (num_object, 3)
                n_dat = n_data[-1, :num_object, :3].astype(int)
                for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
                    # (120, 2) -> (n, 2)
                    # print(n_dat.shape, n_pre.shape)
                    for info, pred in zip(n_dat, n_pre+n_mean_xy):
                        information = info.copy()
                        information[0] = information[0] + time_ind
                        result = ' '.join(information.astype(str)) + ' ' + ' '.join(pred.astype(str)) + '\n'
                        # print(result)
                        writer.write(result)


def run_trainval(pra_model, pra_traindata_path,pra_trainimg_path, pra_testdata_path, pra_testimg_path):
    loader_train = data_loader(pra_traindata_path, pra_trainimg_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True, train_val_test='train')
    loader_val = data_loader(pra_testdata_path, pra_testimg_path, pra_batch_size=batch_size_val, pra_shuffle=True, pra_drop_last=True, train_val_test='val')

    optimizer = optim.Adam(
        [{'params':model.parameters()},],) # lr = 0.0001)

    for now_epoch in range(total_epoch):
        all_loader_train = loader_train

        my_print('#######################################Train')
        train_model(pra_model, all_loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))

        my_save_model(pra_model, now_epoch)

        my_print('#######################################Test')
        display_result(
            val_model(pra_model, loader_val,now_epoch),
            pra_pref='{}_Epoch{}'.format('Test', now_epoch),
        )


def run_test(pra_model, pra_data_path):
    loader_test = data_loader(pra_data_path, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
    test_model(pra_model, loader_test)



if __name__ == '__main__':
#     graph_args={'max_hop':2, 'num_node':120} #120 apolo
    graph_args={'max_hop':2, 'num_node':50}
    model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
    model.to(dev)

    # train and evaluate model
    run_trainval(model, pra_traindata_path='./nuscenes_pkl_map/train_data.pkl', 
                        pra_trainimg_path = './nuscenes_pkl_map/train_data.txt', 
                        pra_testdata_path='./nuscenes_pkl_map/test_data.pkl',
                        pra_testimg_path = './nuscenes_pkl_map/test_data.txt')
#     run_trainval(model, pra_traindata_path='./Dataset2/train_data.pkl', pra_testdata_path='./Dataset2/test_data.pkl')
    
    # pretrained_model_path = './trained_models/model_epoch_0016.pt'
    # model = my_load_model(model, pretrained_model_path)
    # run_test(model, './test_data.pkl')




