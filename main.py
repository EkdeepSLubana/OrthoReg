import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import *
from pruner import *
from config import *
from imp_estimator import cal_importance
from ptflops import get_model_complexity_info
import copy
import numpy as np
import os
import argparse

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model to be pruned", default='vgg', choices=['vgg', 'mobilenet', 'resnet', 'resnet-56'])
parser.add_argument("--ebt", help="extract early-bird tickets", default='False', choices=['True', 'False'])
parser.add_argument("--seed", help="create a new backbone model by setting a different seed", default='0')
parser.add_argument("--pretrained", help="use pretrained model", default='False', choices=['True', 'False'])
parser.add_argument("--data_path", help="path to dataset", default='CIFAR100')
parser.add_argument("--pruning_type", help="train nets with which pruning method", choices=['bn', 'fisher', 'orthoreg', 'tfo', 'l1', 'sfp', 'rdt', 'grasp'])
parser.add_argument("--prune_percent", help="percentage to prune")
parser.add_argument("--n_rounds", help="number of rounds to perform pruning in")
parser.add_argument("--only_train", help="choose to just train a model and not prune", default='False', choices=['base', 'ortho', 'False'])
parser.add_argument("--thresholds", help="define manual thresholds for pruning rounds", default='default')
parser.add_argument("--num_classes", help="number of classes in the dataset", default='100')
parser.add_argument("--imp_samples", help="number of samples for importance estimation", default='10000')
parser.add_argument("--GraSP_T", help="temperature for GraSP", default='200')
args = parser.parse_args()

######### Setup #########
torch.manual_seed(int(args.seed))
cudnn.deterministic = True
cudnn.benchmark = False
device='cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if(device == 'cuda'):
		print("Backend:", device)
else:
	raise Exception("Please use a cuda-enabled GPU.")

if not os.path.isdir('pretrained'):
	os.mkdir('pretrained')
if not os.path.isdir('pretrained/ebt'):
	os.mkdir('pretrained/ebt')
if not os.path.isdir('pretrained/iterative'):
	os.mkdir('pretrained/iterative')
if not os.path.isdir('pruned_nets'):
	os.mkdir('pruned_nets')
if not os.path.isdir('pruned_nets/ebt'):
	os.mkdir('pruned_nets/ebt')
if not os.path.isdir('pruned_nets/iterative'):
	os.mkdir('pruned_nets/iterative')

imp_samples = int(args.imp_samples)
GraSP_T = int(args.GraSP_T)

if(args.ebt == 'True'):
	pretrained_root = 'pretrained/ebt/'
	pruned_root = 'pruned_nets/ebt/'
	args.n_rounds = '1'
	base_sched, base_epochs, ortho_sched, ortho_epochs, wd = base_sched_ebt, base_epochs_ebt, ortho_sched_ebt, ortho_epochs_ebt, wd_ebt
	pruned_sched, pruned_epochs = pruned_sched_ebt, pruned_epochs_ebt
else:
	pretrained_root = 'pretrained/iterative/'
	pruned_root = 'pruned_nets/iterative/'
	base_sched, base_epochs, ortho_sched, ortho_epochs, wd = base_sched_iter, base_epochs_iter, ortho_sched_iter, ortho_epochs_iter, wd_iter
	pruned_sched, pruned_epochs = pruned_sched_iter, pruned_epochs_iter
	ortho_pruned_sched_finetune, ortho_pruned_epochs_finetune = ortho_pruned_sched_finetune, ortho_pruned_epochs_finetune

######### Dataloaders #########
transform = transforms.Compose(
	[transforms.RandomCrop(32, padding=4),
	 transforms.RandomHorizontalFlip(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])
transform_test = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])

if(args.data_path=='CIFAR100'):
	trainset = torchvision.datasets.CIFAR100(root='./../data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR100(root='./../data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
elif(args.data_path=='CIFAR10'):
	args.num_classes = '10'
	trainset = torchvision.datasets.CIFAR10(root='./../cifar10', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./../cifar10', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
else:
	trainset = datasets.ImageFolder(root=args.data_path+'/train', transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testset = datasets.ImageFolder(root=args.data_path+'/test', transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Training functions #########
# Training
def train(net):
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

# Testing
def test(net, pruned=False, suff=''):
	global cfg_state
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

	# Save checkpoint.
	if(pruned):
		global best_p_acc
		acc = 100.*correct/total
		if acc > best_p_acc:
			print('Saving..')
			state = {'net': net.state_dict(), 'cfg': cfg_state}
			if(suff == None):
				suff = ''
			torch.save(state, pruned_root+'{mod_name}_{type}_{num}'.format(mod_name=args.model, type=args.pruning_type, num=str(100-p_cumulative)) + suff + '_' + args.seed + '.pth')
			best_p_acc = acc

	else:
		global best_acc
		acc = 100.*correct/total
		if acc > best_acc:
			print('Saving..')
			state = {'net': net.state_dict()}
			torch.save(state, pretrained_root+'{mod_name}'.format(mod_name=args.model) + suff + '.pth')
			best_acc = acc


### Ortho regularized training ###
# VGG
def vgg_train_ortho(net):
	net.train()
	correct = 0
	total = 0
	running_loss = 0.0

	### layer weights for ortho loss ###
	l_imp = {}
	for conv_ind in [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]:
		l_imp.update({conv_ind: net.module.features[conv_ind].bias.shape[0]**(1/2)})
	normalizer = 0
	for key, val in l_imp.items():
		normalizer += val
	for key, val in l_imp.items():
		l_imp[key] = val / normalizer

	for batch_idx, (inputs, labels) in enumerate(trainloader):
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		L_angle = 0
		
		### Conv_ind == 0 ###
		w_mat = net.module.features[0].weight
		w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
		b_mat = net.module.features[0].bias
		b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
		params = torch.cat((w_mat1, b_mat1), dim=1)
		angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
		L_angle += (l_imp[0])*(angle_mat).norm(1)
		
		### Conv_ind != 0 ###
		for conv_ind in [3, 7, 10, 14, 17, 21, 24, 28, 31]:
			w_mat = net.module.features[conv_ind].weight
			w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
			b_mat = net.module.features[conv_ind].bias
			b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
			params = torch.cat((w_mat1, b_mat1), dim=1)
			angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(w_mat.shape[0]).to(device)
			L_angle += (l_imp[conv_ind])*(angle_mat).norm(1) 
	
		Lc = criterion(outputs, labels)
		loss = 1e-2 * L_angle + Lc
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (running_loss/(batch_idx+1), 100.*correct/total, correct, total))

# MobileNet
def mobile_train_ortho(net):
	net.train()
	running_loss = 0
	correct = 0
	total = 0
	### layer weights for ortho loss ###
	l_imp = {}
	l_imp.update({'conv1': net.module.conv1.weight.shape[0]**(1/2)})
	for conv_ind in range(13):
		l_imp.update({conv_ind: {'a':net.module.layers[conv_ind].conv1.weight.shape[0]**(1/2),
			'b': net.module.layers[conv_ind].conv2.weight.shape[1]**(1/2)}})		
	normalizer = l_imp['conv1']
	for key, val in l_imp.items():
		if(isinstance(val, dict)):
			for key1, val1 in val.items():
				normalizer += val1
	l_imp['conv1'] = l_imp['conv1'] / normalizer
	for key, val in l_imp.items():
		if(isinstance(val, dict)):
			for key1, val1 in val.items():
				l_imp[key][key1] = val1 / normalizer

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, labels = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		L_angle = 0
		### Conv_ind == 0 ###
		w_mat = net.module.conv1.weight
		params = w_mat.reshape(w_mat.shape[0],-1)
		angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
		L_angle += l_imp['conv1'] * (angle_mat).norm(1) 
		### Conv_ind != 0 ###		
		for lnum in range(13):
			w_mat = net.module.layers[lnum].conv1.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			angle_mat = torch.matmul(params.t(), params) - torch.eye(params.shape[1]).to(device)
			L_angle += l_imp[lnum]['a'] * (angle_mat).norm(1)

			w_mat = net.module.layers[lnum].conv2.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			angle_mat = torch.matmul(params.t(), params) - torch.eye(params.shape[1]).to(device)
			L_angle += l_imp[lnum]['b'] * (angle_mat).norm(1)
		Lc = criterion(outputs, labels)
		loss = 1e-3*L_angle + Lc
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (running_loss/(batch_idx+1), 100.*correct/total, correct, total))

# ResNet
def res_train_ortho(net):
	net.train()
	running_loss = 0
	correct = 0
	total = 0
	num_blocks = [3,4,6,3]
	l_imp = {-1:{'conv1': net.module.bn1.bias.shape[0]**(1/2)}, 0:{}, 1:{}, 2:{}, 3:{}}
	mod_id = 0
	for module_id in [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]:
		for b_id in range(num_blocks[mod_id]):
			x = module_id[b_id].conv1.weight.reshape(module_id[b_id].conv1.weight.shape[0],-1).shape
			l_imp[mod_id].update({2*b_id: min(x[0], x[1])**(1/2)})
			x = module_id[b_id].conv2.weight.reshape(module_id[b_id].conv2.weight.shape[0],-1).shape
			l_imp[mod_id].update({2*b_id+1: min(x[0], x[1])**(1/2)})
			try:
				x1 = module_id[b_id].shortcut[0].weight.reshape(module_id[b_id].shortcut[0].weight.shape[0],-1).shape
				l_imp[mod_id].update({'s': min(x1[0], x1[1])**(1/2)})
			except:
				pass
		mod_id += 1
	normalizer = 0
	for key, val in l_imp.items():
		for key1, val1 in val.items():
			normalizer += val1
	for key, val in l_imp.items():
		for key1, val1 in val.items():
			l_imp[key][key1] /= normalizer

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, labels = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		L_angle = 0
		### Conv_ind == 0 ###
		w_mat = net.module.conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
		L_angle += l_imp[-1]['conv1']*(angle_mat).norm(1)		
		### Conv_ind != 0 ###
		mod_id = 0
		for module_id in [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]:
			for b_id in range(num_blocks[mod_id]):
				w_mat = module_id[b_id].conv1.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(params.shape[0]).to(device)
				L_angle += l_imp[mod_id][2*b_id]*(angle_mat).norm(1)
				w_mat = module_id[b_id].conv2.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(params.shape[0]).to(device)
				L_angle += l_imp[mod_id][2*b_id+1]*(angle_mat).norm(1)
				try:
					w_mat = module_id[b_id].shortcut[0].weight
					params = (w_mat.reshape(w_mat.shape[0],-1))
					angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
					L_angle += l_imp[mod_id][2*b_id]*(angle_mat).norm(1)
				except:
					pass
			mod_id += 1
		Lc = criterion(outputs, labels)
		loss = (1e-2)*(L_angle) + Lc
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (running_loss/(batch_idx+1), 100.*correct/total, correct, total))

# ResNet
def res56_train_ortho(net):
	net.train()
	running_loss = 0
	correct = 0
	total = 0
	num_blocks = [9,9,9]
	l_imp = {-1:{'conv1': net.module.bn1.bias.shape[0]**(1/2)}, 0:{}, 1:{}, 2:{}}
	mod_id = 0
	for module_id in [net.module.layer1, net.module.layer2, net.module.layer3]:
		for b_id in range(num_blocks[mod_id]):
			x = module_id[b_id].conv1.weight.reshape(module_id[b_id].conv1.weight.shape[0],-1).shape
			l_imp[mod_id].update({2*b_id: min(x[0], x[1])**(1/2)})
			x = module_id[b_id].conv2.weight.reshape(module_id[b_id].conv2.weight.shape[0],-1).shape
			l_imp[mod_id].update({2*b_id+1: min(x[0], x[1])**(1/2)})
			try:
				x1 = module_id[b_id].shortcut[0].weight.reshape(module_id[b_id].shortcut[0].weight.shape[0],-1).shape
				l_imp[mod_id].update({'s': min(x1[0], x1[1])**(1/2)})
			except:
				pass
		mod_id += 1
	normalizer = 0
	for key, val in l_imp.items():
		for key1, val1 in val.items():
			normalizer += val1
	for key, val in l_imp.items():
		for key1, val1 in val.items():
			l_imp[key][key1] /= normalizer

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, labels = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		L_angle = 0
		### Conv_ind == 0 ###
		w_mat = net.module.conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
		L_angle += l_imp[-1]['conv1']*(angle_mat).norm(1)		
		### Conv_ind != 0 ###
		mod_id = 0
		for module_id in [net.module.layer1, net.module.layer2, net.module.layer3]:
			for b_id in range(num_blocks[mod_id]):
				w_mat = module_id[b_id].conv1.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(params.shape[0]).to(device)
				L_angle += l_imp[mod_id][2*b_id]*(angle_mat).norm(1)
				w_mat = module_id[b_id].conv2.weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(params.shape[0]).to(device)
				L_angle += l_imp[mod_id][2*b_id+1]*(angle_mat).norm(1)
				try:
					w_mat = module_id[b_id].shortcut[0].weight
					params = (w_mat.reshape(w_mat.shape[0],-1))
					angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).to(device)
					L_angle += l_imp[mod_id][2*b_id]*(angle_mat).norm(1)
				except:
					pass
			mod_id += 1
		Lc = criterion(outputs, labels)
		loss = (1e-2)*(L_angle) + Lc
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (running_loss/(batch_idx+1), 100.*correct/total, correct, total))

######### Evaluator for estimating how orthogonal layers are #########
# VGG
def vgg_diag(net):
	### conv_ind == 0 ###
	w_mat = net.module.features[0].weight
	w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
	b_mat = net.module.features[0].bias
	b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
	params = torch.cat((w_mat1, b_mat1), dim=1)
	angle_mat = torch.matmul(torch.t(params), params)
	L_diag = (angle_mat.diag().norm(1))
	L_angle = (angle_mat.norm(1))
	print("Conv_{ind}: {num:.2}".format(ind=0, num=(L_diag.cpu()/L_angle.cpu()).item()))
	### conv_ind != 0 ###
	for conv_ind in [3, 7, 10, 14, 17, 21, 24, 28, 31]:
		w_mat = net.module.features[conv_ind].weight
		w_mat1 = (w_mat.reshape(w_mat.shape[0],-1))
		b_mat = net.module.features[conv_ind].bias
		b_mat1 = (b_mat.reshape(b_mat.shape[0],-1))
		params = torch.cat((w_mat1, b_mat1), dim=1)
		angle_mat = torch.matmul(params, torch.t(params))
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("Conv_{ind}: {num:.2}".format(ind=conv_ind, num=(L_diag.cpu()/L_angle.cpu()).item()))

# MobileNet
def mobile_diag(net):
	w_mat = net.module.conv1.weight
	params = (w_mat.reshape(w_mat.shape[0],-1))
	angle_mat = torch.matmul(torch.t(params), params) 
	L_diag = (angle_mat.diag().norm(1))
	L_angle = (angle_mat.norm(1))
	print("Conv_base: {num:.2}".format(num=(L_diag.cpu()/L_angle.cpu()).item()))
	for lnum in range(13):
		w_mat = net.module.layers[lnum].conv1.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(params.t(), params)
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("Conv_{ind} -depthwise: {num:.2}".format(ind=lnum, num=(L_diag.cpu()/L_angle.cpu()).item()))
		w_mat = net.module.layers[lnum].conv2.weight
		params = (w_mat.reshape(w_mat.shape[0],-1))
		angle_mat = torch.matmul(params.t(), params)
		L_diag = (angle_mat.diag().norm(1))
		L_angle = (angle_mat.norm(1))
		print("Conv_{ind} -pointwise: {num:.2}".format(ind=lnum, num=(L_diag.cpu()/L_angle.cpu()).item()))

# ResNet-34
def res_diag(net):
	### Conv_ind == 0 ###
	num_blocks = [3,4,6,3]
	w_mat = net.module.conv1.weight
	params = (w_mat.reshape(w_mat.shape[0],-1))
	angle_mat = torch.matmul(torch.t(params), params)
	L_diag = (angle_mat.diag().norm(1))
	L_angle = (angle_mat.norm(1))
	print(L_diag.cpu()/L_angle.cpu())
	### Conv_ind != 0 ###
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
			print('layer_'+str(mod_id)+'--->', str(b_id)+'_1: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			w_mat = module_id[b_id].conv2.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			if(params.shape[1] < params.shape[0]):
				params = params.t()
			angle_mat = torch.matmul(params, torch.t(params))
			L_diag = (angle_mat.diag().norm(1))
			L_angle = (angle_mat.norm(1))
			print('layer_'+str(mod_id)+'--->', str(b_id)+'_2: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			try:
				w_mat = module_id[b_id].shortcut[0].weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('layer_'+str(mod_id)+'--->', 'shortcut: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			except:
				pass
		mod_id += 1

# ResNet-34
def res56_diag(net):
	### Conv_ind == 0 ###
	num_blocks = [9,9,9]
	w_mat = net.module.conv1.weight
	params = (w_mat.reshape(w_mat.shape[0],-1))
	angle_mat = torch.matmul(torch.t(params), params)
	L_diag = (angle_mat.diag().norm(1))
	L_angle = (angle_mat.norm(1))
	print(L_diag.cpu()/L_angle.cpu())
	### Conv_ind != 0 ###
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
			print('layer_'+str(mod_id)+'--->', str(b_id)+'_1: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			w_mat = module_id[b_id].conv2.weight
			params = (w_mat.reshape(w_mat.shape[0],-1))
			if(params.shape[1] < params.shape[0]):
				params = params.t()
			angle_mat = torch.matmul(params, torch.t(params))
			L_diag = (angle_mat.diag().norm(1))
			L_angle = (angle_mat.norm(1))
			print('layer_'+str(mod_id)+'--->', str(b_id)+'_2: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			try:
				w_mat = module_id[b_id].shortcut[0].weight
				params = (w_mat.reshape(w_mat.shape[0],-1))
				if(params.shape[1] < params.shape[0]):
					params = params.t()
				angle_mat = torch.matmul(params, torch.t(params))
				L_diag = (angle_mat.diag().norm(1))
				L_angle = (angle_mat.norm(1))
				print('layer_'+str(mod_id)+'--->', 'shortcut: %.2f' % (L_diag.cpu()/L_angle.cpu()).item())
			except:
				pass
		mod_id += 1

######### Determine model, load, and train #########
# VGG 
if(args.model == 'vgg'):
	### Use pretrained model ###
	if(args.pretrained == 'True'):
		print("\nLoading pretrained model...", end="")	
		net = torch.nn.DataParallel(VGG().to(device))
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'vgg_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'vgg.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

	### Train new model ###
	elif(args.pretrained=='False' or args.only_train=='ortho' or args.only_train=='base'):
		print("\nTraining base model...")	
		net = torch.nn.DataParallel(VGG().to(device))
		best_acc = 0
		if(args.pruning_type=='orthoreg' or args.only_train=='ortho'):
			### Train Ortho Network ###
			#if(args.ebt=='False'):
				#net_dict = torch.load(pretrained_root+'vgg.pth')
				#net.load_state_dict(net_dict['net'])
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=ortho_sched[lr_ind], momentum=0.9, weight_decay=0)
			while(lr_ind < len(ortho_sched)):
				optimizer.param_groups[0]['lr'] = ortho_sched[lr_ind]
				for n in range(ortho_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					vgg_train_ortho(net)
					test(net, suff='_ortho')
					vgg_diag(net)
					epoch += 1
				lr_ind += 1
		else:
			### Train correlated network ###
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=base_sched[lr_ind], momentum=0.9, weight_decay=wd)
			while(lr_ind < len(base_sched)):
				optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
				for n in range(base_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					train(net)
					test(net)
					epoch += 1
				lr_ind += 1

		print("\nLoading best checkpoint...", end="")	
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'vgg_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'vgg.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

# MobileNet 
elif(args.model == 'mobilenet'):

	### Use pretrained model ###
	if(args.pretrained == 'True'):
		print("\nLoading pretrained model...", end="")	
		net = torch.nn.DataParallel(MobileNet().to(device))
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'mobilenet_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'mobilenet.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

	### Train new model ###
	elif(args.pretrained=='False' or args.only_train=='ortho' or args.only_train=='base'):
		print("\nTraining base model...")	
		net = torch.nn.DataParallel(MobileNet().to(device))
		best_acc = 0
		if(args.pruning_type=='orthoreg' or args.only_train=='ortho'):
			### Train Ortho Network ###
			#if(args.ebt=='False'):
				#net_dict = torch.load(pretrained_root+'mobilenet.pth')
				#net.load_state_dict(net_dict['net'])
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=ortho_sched[lr_ind], momentum=0.9, weight_decay=0)
			while(lr_ind < len(ortho_sched)):
				optimizer.param_groups[0]['lr'] = ortho_sched[lr_ind]
				for n in range(ortho_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					mobile_train_ortho(net)
					test(net, suff='_ortho')
					mobile_diag(net)
					epoch += 1
				lr_ind += 1
		else:
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=base_sched[lr_ind], momentum=0.9, weight_decay=wd)
			while(lr_ind < len(base_sched)):
				optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
				for n in range(base_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					train(net)
					test(net)
					epoch += 1
				lr_ind += 1

		print("\nLoading best checkpoint...", end="")	
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'mobilenet_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'mobilenet.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")


# ResNet
elif(args.model == 'resnet'):
	# Use pretrained model 
	if(args.pretrained == 'True'):
		print("\nLoading pretrained model...", end="")	
		net = torch.nn.DataParallel(ResNet34().to(device))
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'resnet_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'resnet.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

	# Train new model 
	elif(args.pretrained=='False' or args.only_train=='ortho' or args.only_train=='base'):
		print("\nTraining base model...")	
		net = torch.nn.DataParallel(ResNet34().to(device))
		best_acc = 0

		### Train with Ortho regularizer ###
		if(args.pruning_type=='orthoreg' or args.only_train=='ortho'):
			#if(args.ebt=='False'):
				#net_dict = torch.load(pretrained_root+'resnet.pth')
				#net.load_state_dict(net_dict['net'])
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=ortho_sched[lr_ind], momentum=0.9, weight_decay=0)
			while(lr_ind < len(ortho_sched)):
				optimizer.param_groups[0]['lr'] = ortho_sched[lr_ind]
				for n in range(ortho_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					res_train_ortho(net)
					test(net, suff='_ortho')
					res_diag(net)
					epoch += 1
				lr_ind += 1
		else:
			### Train without regularizer ###
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=base_sched[lr_ind], momentum=0.9, weight_decay=wd)
			while(lr_ind < len(base_sched)):
				optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
				for n in range(base_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					train(net)
					test(net)
					epoch += 1
				lr_ind += 1

		print("\nLoading best checkpoint...", end="")	
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'resnet_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'resnet.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

# ResNet-56
elif(args.model == 'resnet-56'):

	# Use pretrained model 
	if(args.pretrained == 'True'):
		print("\nLoading pretrained model...", end="")	
		net = torch.nn.DataParallel(ResNet56().to(device))
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'resnet-56_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'resnet-56.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")

	# Train new model 
	elif(args.pretrained=='False' or args.only_train=='ortho' or args.only_train=='base'):
		print("\nTraining base model...")	
		net = torch.nn.DataParallel(ResNet56(num_classes=int(args.num_classes)).to(device))
		best_acc = 0

		### Train with Ortho regularizer ###
		if(args.pruning_type=='orthoreg' or args.only_train=='ortho'):
			#if(args.ebt=='False'):
				#net_dict = torch.load(pretrained_root+'resnet-56.pth')
				#net.load_state_dict(net_dict['net'])
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=ortho_sched[lr_ind], momentum=0.9, weight_decay=0)
			while(lr_ind < len(ortho_sched)):
				optimizer.param_groups[0]['lr'] = ortho_sched[lr_ind]
				for n in range(ortho_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					res56_train_ortho(net)
					test(net, suff='_ortho')
					res56_diag(net)
					epoch += 1
				lr_ind += 1
		else:
			### Train without regularizer ###
			lr_ind = 0
			epoch = 0
			optimizer = optim.SGD(net.parameters(), lr=base_sched[lr_ind], momentum=0.9, weight_decay=wd)
			while(lr_ind < len(base_sched)):
				optimizer.param_groups[0]['lr'] = base_sched[lr_ind]
				for n in range(base_epochs[lr_ind]):
					print('\nEpoch: {}'.format(epoch))
					train(net)
					test(net)
					epoch += 1
				lr_ind += 1

		print("\nLoading best checkpoint...", end="")	
		if(args.pruning_type == 'orthoreg'):
			net_dict = torch.load(pretrained_root+'resnet-56_ortho.pth')
			net.load_state_dict(net_dict['net'])
		else:
			net_dict = torch.load(pretrained_root+'resnet-56.pth')
			net.load_state_dict(net_dict['net'])
		print("  Loaded\n")


# Print FLOPs in base model 
with torch.cuda.device(0):
	flops, params = get_model_complexity_info(net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
	print('{:<30}  {:<8}\n'.format('FLOPs in base model: ', flops))

######### Pruning happens beyond this #########
if(args.only_train == 'False'):

	if(args.thresholds == 'default'):
		n_rounds = int(args.n_rounds)
		p_total = int(args.prune_percent) / 100
		p_ratios = p_total / (np.arange(1, n_rounds+1, 1) * (p_total) +  n_rounds * (1 - p_total))
		#p_ratios = np.flip(p_total / (np.arange(1, n_rounds+1, 1) * (p_total) +  n_rounds * (1 - p_total)))
	else:
		p_list = args.thresholds
		for char_remov in ['[', ' ', ']']:
			p_list = p_list.replace(char_remov, '')
		p_list = p_list.split(',')
		for ind, thresh in enumerate(p_list):
			p_list[ind] = (float(thresh) / 100)
		p_list = np.array(p_list)
		p_ratios = [p_list[0]]
		for i in range(1, p_list.shape[0]):
			p_ratios.append((p_list[i] - p_list[i-1]) / (1 - p_list[i-1]))
		p_ratios = np.array(p_ratios)
		n_rounds = p_ratios.shape[0]

	net_p = copy.deepcopy(net)
	p_cumulative = 100

	for n_iter in range(n_rounds):
		prune_iter = p_ratios[n_iter] * 100                
		p_cumulative = int(p_cumulative * (1 - p_ratios[n_iter]))
		print("\n------------------ Round: {iter} ------------------\n".format(iter=n_iter+1))
		if(args.pruning_type=='orthoreg' and n_iter == n_rounds-1 and args.ebt == 'False'):
			print("Last round for OrthoReg; The regularizer won't be used at all\n")

		######### Estimate Importance #########
		print("Calculating importance for {} pruning".format(args.pruning_type))
		# VGG
		if(args.model=='vgg'):
			imp_order = np.array([[],[],[]]).transpose()                        
			list_imp = cal_importance(net_p, args.model, args.pruning_type, trainloader, num_stop=imp_samples, T=GraSP_T)
			for ind, l_index in enumerate([2, 5, 9, 12, 16, 19, 23, 26, 30, 33]):
				nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
				imp_order = np.concatenate((imp_order,np.array([np.repeat([l_index],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
												nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
		# MobileNet
		elif(args.model=='mobilenet'):
			imp_order = np.array([[],[],[]]).transpose()
			list_imp = cal_importance(net_p, args.model, args.pruning_type, trainloader, num_stop=imp_samples, T=GraSP_T)
			ind = 0
			nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
			imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
												nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
			ind+=1
			for l_index in range(13):
				imp_prev = imp_order[imp_order[:,0] == ind-1, 2]
				nlist = [np.linspace(0, imp_prev.shape[0]-1, imp_prev.shape[0]), imp_prev]
				imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(),
												nlist[1].tolist()]).transpose()), 0)
				ind+=1
				nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
				imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
												nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
				ind+=1

		# ResNet
		elif(args.model=='resnet'):
			imp_order = np.array([[],[],[]]).transpose()
			list_imp = cal_importance(net_p, args.model, args.pruning_type, trainloader, num_stop=imp_samples, T=GraSP_T)
			for ind in range(36):
				nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
				imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
												nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
				imp_order[imp_order[:,0] == 0, 2] = 1e7

		# ResNet
		elif(args.model=='resnet-56'):
			imp_order = np.array([[],[],[]]).transpose()
			list_imp = cal_importance(net_p, args.model, args.pruning_type, trainloader, num_stop=imp_samples, T=GraSP_T)
			for ind in range(57):
				nlist = [np.linspace(0, list_imp[ind].shape[0]-1, list_imp[ind].shape[0]), list_imp[ind]]
				imp_order = np.concatenate((imp_order,np.array([np.repeat([ind],nlist[1].shape[0]).tolist(), nlist[0].tolist(), 
												nlist[1].detach().cpu().numpy().tolist()]).transpose()), 0)
				imp_order[imp_order[:,0] == 0, 2] = 1e7

		######### Prune using estimated importance #########
		# VGG
		if(args.model=='vgg'):
			if os.path.exists(pretrained_root+"vgg.pth"):	
				base_size = float(os.path.getsize(pretrained_root+"vgg.pth"))
			else:
				base_size = float(os.path.getsize(pretrained_root+"vgg_ortho.pth"))			

			### Prune network ###
			imp_order = constrain_ratios(imp_order, vgg_size(net))
			orig_size = vgg_size(net_p)
			order_layerwise, prune_ratio = vgg_order_and_ratios(imp_order, prune_iter / 100)

			# Print pruned architecture
			print("Pruned Architecture:", orig_size - prune_ratio)

			net_p, cfg_state = vgg_pruner(net_p, order_layerwise, prune_ratio, orig_size)	
			state = {'net': net_p.state_dict()}
			torch.save(state, pruned_root+'vgg_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))
			
			# Characterize pruned model
			print("\n##################\nPruned: {:.2%}\n##################\n".format(1 - vgg_size(net_p).sum() / vgg_size(net).sum()))
			# FLOPs 
			with torch.cuda.device(0):
				flops, params = get_model_complexity_info(net_p, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
				print('{:<30}  {:<8}'.format('FLOPs in pruned model: ', flops))
			# Compression Ratio 
			print("Compression ratio of pruned model: {:.3}".format(base_size / os.path.getsize(pruned_root+'vgg_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))))
			                        
			### Retrain ###
			if(args.pruning_type=='orthoreg' and n_iter < n_rounds-1):
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_iter[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_iter[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_iter[lr_ind]
					for n in range(ortho_pruned_epochs_iter[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True, suff='_ortho')
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%} \n".format(best_p_acc / 100))

				### Fine-tune with the regularizer ###
				print("-----Fine-tuning with the regularizer-----")
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_finetune[lr_ind], momentum=0.9, weight_decay=0)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_finetune[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_finetune[lr_ind]
					for n in range(ortho_pruned_epochs_finetune[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						vgg_train_ortho(net_p)
						test(net_p, pruned=True, suff='_ortho')
						vgg_diag(net_p)
						epoch += 1
					lr_ind += 1
				print("Accuracy of pruned model after fine-tuning (best checkpoint): {:.2%}".format(best_p_acc / 100))

			else:
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=pruned_sched[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(pruned_sched[lr_ind]))
					optimizer.param_groups[0]['lr'] = pruned_sched[lr_ind]
					for n in range(pruned_epochs[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True)
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%}".format(best_p_acc / 100))


		# MobileNet
		elif(args.model=='mobilenet'):
			if os.path.exists(pretrained_root+"mobilenet.pth"):	
				base_size = float(os.path.getsize(pretrained_root+"mobilenet.pth"))
			else:
				base_size = float(os.path.getsize(pretrained_root+"mobilenet_ortho.pth"))			

			### Prune network ###
			imp_order = constrain_ratios(imp_order, mobile_size(net))
			orig_size = mobile_size(net_p)
			order_layerwise, prune_ratio = mobile_order_and_ratios(imp_order, prune_iter / 100)

			# Print pruned architecture
			print("Pruned Architecture:", orig_size - prune_ratio)

			net_p, cfg_state = mobile_pruner(net_p, order_layerwise, prune_ratio, orig_size)	
			state = {'net': net_p.state_dict()}
			torch.save(state, pruned_root+'mobilenet_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))
			
			# Characterize pruned model
			print("\n##################\nPruned: {:.2%}\n##################\n".format(1 - mobile_size(net_p).sum() / mobile_size(net).sum()))
			# FLOPs 
			with torch.cuda.device(0):
				flops, params = get_model_complexity_info(net_p, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
				print('{:<30}  {:<8}'.format('FLOPs in pruned model: ', flops))
			# Compression Ratio 
			print("Compression ratio of pruned model: {:.3}".format(base_size / os.path.getsize(pruned_root+'mobilenet_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))))

			### Retrain ###
			if(args.pruning_type=='orthoreg' and n_iter < n_rounds-1):
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_iter[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_iter[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_iter[lr_ind]
					for n in range(ortho_pruned_epochs_iter[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True, suff='_ortho')
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%} \n".format(best_p_acc / 100))

				### Fine-tune with the regularizer ###
				print("-----Fine-tuning with the regularizer-----")
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_finetune[lr_ind], momentum=0.9, weight_decay=0)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_finetune[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_finetune[lr_ind]
					for n in range(ortho_pruned_epochs_finetune[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						vgg_train_ortho(net_p)
						test(net_p, pruned=True, suff='_ortho')
						vgg_diag(net_p)
						epoch += 1
					lr_ind += 1
				print("Accuracy of pruned model after fine-tuning (best checkpoint): {:.2%}".format(best_p_acc / 100))

			else:
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=pruned_sched[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(pruned_sched[lr_ind]))
					optimizer.param_groups[0]['lr'] = pruned_sched[lr_ind]
					for n in range(pruned_epochs[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True)
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%}".format(best_p_acc / 100))

		# ResNet
		elif(args.model=='resnet'):
			if os.path.exists(pretrained_root+"resnet.pth"):	
				base_size = float(os.path.getsize(pretrained_root+"resnet.pth"))
			else:
				base_size = float(os.path.getsize(pretrained_root+"resnet_ortho.pth"))			

			### Prune network ###
			orig_size = res_size(net_p)
			res_iter = 1 - (p_cumulative / 100) * (res_size(net).sum()) / orig_size.sum()
			order_layerwise, prune_ratio = res_order_and_ratios(imp_order, res_iter)
			prune_ratio[0] = 0
			for i, n in enumerate(orig_size - np.array(prune_ratio)):
				if(n == 0):
					prune_ratio[i] -= 1

			# Print pruned architecture
			print("Pruned Architecture:", orig_size - prune_ratio)

			net_p, order_zeros = res_pruner(net_p, order_layerwise, prune_ratio, orig_size)
			net_p, cfg_state = skip_or_prune(net_p, order_zeros, prune_ratio, orig_size)
			cfg_zero = cfg_res_zero(prune_ratio, orig_size)
			net_zero = torch.nn.DataParallel(ResPruned(cfg_zero))
			state = {'net': net_p.state_dict()}
			torch.save(state, pruned_root+'res_zero_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))

			# Characterize pruned model
			print("\n##################\nPruned: {:.2%}\n##################\n".format(1 - res_size(net_zero).sum() / res_size(net).sum()))
			# FLOPs 
			with torch.cuda.device(0):
				flops, params = get_model_complexity_info(net_p, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
				print('{:<30}  {:<8}'.format('FLOPs in pruned model: ', flops))
			# Compression Ratio 
			print("Compression ratio of pruned model: {:.3}".format(base_size / os.path.getsize(pruned_root+'res_zero_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))))

			### Retrain ###
			if(args.pruning_type=='orthoreg' and n_iter < n_rounds-1):
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_iter[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_iter[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_iter[lr_ind]
					for n in range(ortho_pruned_epochs_iter[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True, suff='_ortho')
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%} \n".format(best_p_acc / 100))

				### Fine-tune with the regularizer ###
				print("-----Fine-tuning with the regularizer-----")
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_finetune[lr_ind], momentum=0.9, weight_decay=0)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_finetune[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_finetune[lr_ind]
					for n in range(ortho_pruned_epochs_finetune[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						vgg_train_ortho(net_p)
						test(net_p, pruned=True, suff='_ortho')
						vgg_diag(net_p)
						epoch += 1
					lr_ind += 1
				print("Accuracy of pruned model after fine-tuning (best checkpoint): {:.2%}".format(best_p_acc / 100))

			else:
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=pruned_sched[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(pruned_sched[lr_ind]))
					optimizer.param_groups[0]['lr'] = pruned_sched[lr_ind]
					for n in range(pruned_epochs[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True)
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%}".format(best_p_acc / 100))

		# ResNet-56
		elif(args.model=='resnet-56'):
			if os.path.exists(pretrained_root+"resnet-56.pth"):	
				base_size = float(os.path.getsize(pretrained_root+"resnet-56.pth"))
			else:
				base_size = float(os.path.getsize(pretrained_root+"resnet-56_ortho.pth"))			

			### Prune network ###
			orig_size = res_size_cifar(net_p)
			res_iter = 1 - (p_cumulative / 100) * (res_size_cifar(net).sum()) / orig_size.sum()
			order_layerwise, prune_ratio = res_order_and_ratios_cifar(imp_order, res_iter)
			prune_ratio[0] = 0
			for i, n in enumerate(orig_size - np.array(prune_ratio)):
				if(n == 0):
					prune_ratio[i] -= 1

			# Print pruned architecture
			print("Pruned Architecture:", orig_size - prune_ratio)

			net_p, order_zeros = res_pruner_cifar(net_p, order_layerwise, prune_ratio, orig_size, num_classes=int(args.num_classes))
			net_p, cfg_state = skip_or_prune_cifar(net_p, order_zeros, prune_ratio, orig_size)
			cfg_zero = cfg_res_zero_cifar(prune_ratio, orig_size)
			net_zero = torch.nn.DataParallel(ResPruned_cifar(cfg_zero, num_classes=int(args.num_classes)))
			state = {'net': net_p.state_dict()}
			torch.save(state, pruned_root+'res56_zero_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))

			# Characterize pruned model
			print("\n##################\nPruned: {:.2%}\n##################\n".format(1 - res_size_cifar(net_zero).sum() / res_size_cifar(net).sum()))
			# FLOPs 
			with torch.cuda.device(0):
				flops, params = get_model_complexity_info(net_p, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
				print('{:<30}  {:<8}'.format('FLOPs in pruned model: ', flops))
			# Compression Ratio 
			print("Compression ratio of pruned model: {:.3}".format(base_size / os.path.getsize(pruned_root+'res56_zero_{type}_{num}.pth'.format(type=args.pruning_type, num=str(100-p_cumulative)))))

			### Retrain ###
			if(args.pruning_type=='orthoreg' and n_iter < n_rounds-1):
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_iter[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_iter[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_iter[lr_ind]
					for n in range(ortho_pruned_epochs_iter[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True, suff='_ortho')
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%} \n".format(best_p_acc / 100))

				### Fine-tune with the regularizer ###
				print("-----Fine-tuning with the regularizer-----")
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=ortho_pruned_sched_finetune[lr_ind], momentum=0.9, weight_decay=0)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(ortho_pruned_sched_finetune[lr_ind]))
					optimizer.param_groups[0]['lr'] = ortho_pruned_sched_finetune[lr_ind]
					for n in range(ortho_pruned_epochs_finetune[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						vgg_train_ortho(net_p)
						test(net_p, pruned=True, suff='_ortho')
						vgg_diag(net_p)
						epoch += 1
					lr_ind += 1
				print("Accuracy of pruned model after fine-tuning (best checkpoint): {:.2%}".format(best_p_acc / 100))

			else:
				### Train pruned model ###
				best_p_acc = 0
				lr_ind = 0
				epoch = 0
				optimizer = optim.SGD(net_p.parameters(), lr=pruned_sched[lr_ind], momentum=0.9, weight_decay=wd)
				while(lr_ind < len(ortho_pruned_sched_iter)):
					print("\n--learning rate is {}".format(pruned_sched[lr_ind]))
					optimizer.param_groups[0]['lr'] = pruned_sched[lr_ind]
					for n in range(pruned_epochs[lr_ind]):
						print('\nEpoch: {}'.format(epoch))
						train(net_p)
						test(net_p, pruned=True)
						epoch += 1
					lr_ind += 1		
				print("Accuracy of pruned model (best checkpoint): {:.2%}".format(best_p_acc / 100))
