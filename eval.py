# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import torch.optim as optim
import os
import shutil
from models import *
from pruner import *
from config import *
from ptflops import get_model_complexity_info
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="architecture model to be analyzed", default='vgg', choices=['vgg', 'mobilenet', 'resnet', 'resnet-56'])
parser.add_argument("--model_path", help="path where the model to be analyzed is stored", default='0')
parser.add_argument("--data_path", help="path to dataset", default='CIFAR100')
parser.add_argument("--pruned", help="is the model to be analyzed a pruned model?", default='False', choices=['True', 'False'])
parser.add_argument("--train_acc", help="evaluate train accuracy", default='False', choices=['True', 'False'])
parser.add_argument("--test_acc", help="evaluate test accuracy", default='False', choices=['True', 'False'])
parser.add_argument("--flops", help="calculate flops in a model", default='False', choices=['True', 'False'])
parser.add_argument("--compression", help="calculate compression ratio for model", default='False', choices=['True', 'False'])
parser.add_argument("--eval_ortho", help="evaluate how orthogonal a model is", default='False', choices=['True', 'False'])
parser.add_argument("--finetune", help="fine-tune a model", default='False', choices=['True', 'False'])
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
	
######### Functions to evaluate different properties #########
# Accuracy
def cal_acc(net, use_loader):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(use_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	return correct / total

# FLOPs 
def cal_flops(net):
	with torch.cuda.device(0):
		flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
		print('	FLOPs: {:<8}'.format(flops))

# Compression Ratio
def cal_compression_ratio(net_path, model):
	temp_path = "./temp_models/"
	base_model = create_model(name=model, is_pruned=False)
	if os.path.exists(temp_path):
		shutil.rmtree(temp_path)
	os.mkdir(temp_path)
	state = {'net': base_model.state_dict()}
	torch.save(state, temp_path+'temp_base.pth')
	base_size = os.path.getsize(temp_path+'temp_base.pth')
	model_size = os.path.getsize(net_path)
	print("	Compression ratio: {:.3}".format(base_size / model_size))
	shutil.rmtree(temp_path)

# Fine-tune
def finetune(net):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Orthogonality evaluator
def eval_ortho():
	if(args.model == 'vgg'):
		w_mat = net.module.features[0].weight
		w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
		b_mat = net.module.features[0].bias
		b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
		params = torch.cat((w_mat1, b_mat1), dim=1)
		angle_mat = torch.matmul(torch.t(params), params)
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("	Conv_{ind}: {num:.2}".format(ind=0, num=(L_diag.cpu()/L_angle.cpu()).item()))
		for conv_ind in [3, 7, 10, 14, 17, 21, 24, 28, 31]:
			w_mat = net.module.features[conv_ind].weight
			w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
			b_mat = net.module.features[conv_ind].bias
			b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
			params = torch.cat((w_mat1, b_mat1), dim=1)
			angle_mat = torch.matmul(params, torch.t(params))
			L_diag = (angle_mat.diag().norm(1))
			L_angle = (angle_mat.norm(1))
			print("	Conv_{ind}: {num:.2}".format(ind=conv_ind, num=(L_diag.cpu()/L_angle.cpu()).item()))

	elif(args.model == 'mobilenet'):
		w_mat = net.module.conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(torch.t(params), params) 
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("	Conv_base: {num:.2}".format(num=(L_diag.cpu()/L_angle.cpu()).item()))
		for lnum in range(13):
			w_mat = net.module.layers[lnum].conv1.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			angle_mat = torch.matmul(params.t(), params)
			L_diag = (angle_mat.diag().norm(1))
			L_angle = (angle_mat.norm(1))
			print("	Conv_{ind} -depthwise: {num:.2}".format(ind=lnum, num=(L_diag.cpu()/L_angle.cpu()).item()))
			w_mat = net.module.layers[lnum].conv2.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			angle_mat = torch.matmul(params.t(), params)
			L_diag = (angle_mat.diag().norm(1))
			L_angle = (angle_mat.norm(1))
			print("	Conv_{ind} -pointwise: {num:.2}".format(ind=lnum, num=(L_diag.cpu()/L_angle.cpu()).item()))

	elif(args.model == 'resnet'):
		num_blocks = [3,4,6,3]
		w_mat = net.module.conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(torch.t(params), params)
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("	base layer:", (L_diag.cpu()/L_angle.cpu()).item())
		mod_id = 0
		for module_id in [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]:
			for b_id in range(num_blocks[mod_id]):
				w_mat = module_id[b_id].conv1.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('	layer_'+str(mod_id)+', block', str(b_id)+'_1: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				w_mat = module_id[b_id].conv2.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('	layer_'+str(mod_id)+', block', str(b_id)+'_2: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				try:
					w_mat = module_id[b_id].shortcut[0].weight
					params = (w_mat.reshape(w_mat.shape[0],-1))
					if(params.shape[1] < params.shape[0]):
						params = params.t()
					angle_mat = torch.matmul(params, torch.t(params))
					L_diag = (angle_mat.diag().norm(1))
					L_angle = (angle_mat.norm(1))
					print('	layer_'+str(mod_id) + ', shortcut: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				except:
					pass
			mod_id += 1

	elif(args.model == 'resnet-56'):
		num_blocks = [9,9,9]
		w_mat = net.module.conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(torch.t(params), params)
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("	base layer:", (L_diag.cpu()/L_angle.cpu()).item())
		mod_id = 0
		for module_id in [net.module.layer1, net.module.layer2, net.module.layer3]:
			for b_id in range(num_blocks[mod_id]):
				w_mat = module_id[b_id].conv1.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('	layer_'+str(mod_id)+', block', str(b_id)+'_1: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				w_mat = module_id[b_id].conv2.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('	layer_'+str(mod_id)+', block', str(b_id)+'_2: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				try:
					w_mat = module_id[b_id].shortcut[0].weight
					params = (w_mat.reshape(w_mat.shape[0],-1))
					if(params.shape[1] < params.shape[0]):
						params = params.t()
					angle_mat = torch.matmul(params, torch.t(params))
					L_diag = (angle_mat.diag().norm(1))
					L_angle = (angle_mat.norm(1))
					print('	layer_'+str(mod_id) + ', shortcut: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
				except:
					pass
			mod_id += 1

# Create model for evaluation#net = torch.nn.DataParallel(VGG())
def create_model(name, is_pruned):
	if(name == 'vgg'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(VGG_p(cfg_p))
		else:
			net = torch.nn.DataParallel(VGG())

	elif(name == 'mobilenet'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(MobileNet_p(cfg_p[0], cfg_p[1:]))
		else:
			net = torch.nn.DataParallel(MobileNet())

	elif(name == 'resnet'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(ResPruned(cfg_p))
		else:
			net = torch.nn.DataParallel(ResNet34())

	elif(name == 'resnet-56'):
		if(is_pruned == True):
			cfg_p = net_dict['cfg']
			net = torch.nn.DataParallel(ResPruned_cifar(cfg_p))
		else:
			net = torch.nn.DataParallel(ResNet56())
	return net


######### Print model name #########
print((args.model).upper())

######### Dataloader #########
if(args.train_acc == 'True' or args.test_acc == 'True' or args.finetune == 'True'):
	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	transform_test = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	if(args.data_path=='CIFAR100'):
		trainset = torchvision.datasets.CIFAR100(root='./../data', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
		testset = torchvision.datasets.CIFAR100(root='./../data', train=False, download=True, transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
	else:
		trainset = datasets.ImageFolder(root=args.data_path+'/train', transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
		testset = datasets.ImageFolder(root=args.data_path+'/test', transform=transform_test)
		testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Testing
def test(net):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
	acc = 100. * correct/total
	if acc > best_acc:
		print("best accuracy:", acc)
		best_acc = acc

######### Load network or create new #########
if(args.train_acc == 'True' or args.test_acc == 'True' or args.flops=='True' or args.eval_ortho=='True'):
	net_dict = torch.load(args.model_path)
	net = create_model(name=args.model, is_pruned=(args.pruned=='True'))
	net.load_state_dict(net_dict['net'])
	
######### FLOPs evaluation #########
if(args.flops == 'True'):
	cal_flops(net)

######### Compression ratio evaluation #########
if(args.compression == 'True'):
	cal_compression_ratio(net_path=args.model_path, model=args.model)

######### Train accuracy evaluation #########
if(args.train_acc == 'True'):
	acc = cal_acc(net, use_loader=trainloader)
	print("	Train accuracy: {:.2%}".format(acc))

######### Test accuracy evaluation #########
if(args.test_acc == 'True'):
	acc = cal_acc(net, use_loader=testloader)
	print("	Test accuracy: {:.2%}".format(acc))

######### Orthogonality evaluation #########
if(args.eval_ortho == 'True'):
	eval_ortho()

if(args.finetune == 'True'):
	net_dict = torch.load(args.model_path)
	net = create_model(name=args.model, is_pruned=(args.pruned=='True'))
	net.load_state_dict(net_dict['net'])
	base_sched, base_epochs, wd = pruned_sched_iter, pruned_epochs_iter, wd_iter
	best_acc = 0
	lr_ind = 0
	epoch = 0
	optimizer = optim.SGD(net.parameters(), lr=base_sched[lr_ind], momentum=0.9, weight_decay=wd)
	while(lr_ind < len(base_sched)):
		optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
		for n in range(base_epochs[lr_ind]):
			print('\nEpoch: {}'.format(epoch))
			finetune(net)
			test(net)
			epoch += 1
		lr_ind += 1
