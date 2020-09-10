import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torchvision
import torchvision.transforms as transforms
import numpy as np

criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_params(model):
	params = []
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		params.append(param)
	return params

def model_grads(model):
	grads = []
	for param in model.parameters():
		if not param.requires_grad:
			continue
		grads.append(0. if param.grad is None else param.grad + 0.)
	return grads


def grasp_data(dataloader, n_classes, n_samples):
	datas = [[] for _ in range(n_classes)]
	labels = [[] for _ in range(n_classes)]
	mark = dict()
	dataloader_iter = iter(dataloader)
	while True:
		inputs, targets = next(dataloader_iter)
		for idx in range(inputs.shape[0]):
			x, y = inputs[idx:idx+1], targets[idx:idx+1]
			category = y.item()
			if len(datas[category]) == n_samples:
				mark[category] = True
				continue
			datas[category].append(x)
			labels[category].append(y)
		if len(mark) == n_classes:
			break

	X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
	return X, y

### Gradient ###
def cal_grad(net, trainloader, num_stop=5000):
	net.eval()
	num_data = 0  # count the number of datum points in the dataloader
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	for inputs, targets in trainloader:
		if(num_data > num_stop):
			break
		net.zero_grad()
		tmp_num_data = inputs.size(0)
		outputs = net(inputs.to(device))
		loss = criterion(outputs, targets.to(device))
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.detach().clone() * float(tmp_num_data) for gbase1, g1 in zip(gbase, gradsH) ]
		num_data += float(tmp_num_data)

	gbase = [gbase1 / num_data for gbase1 in gbase]
	gbase = [gbase1.reshape(gbase1.shape[0], -1) for gbase1 in gbase]

	return gbase

### Hg ###
def cal_hg(net, trainloader, T=200, n_samples=100, n_classes=100):
	net.eval()
	d_in, d_out = grasp_data(trainloader, n_classes, n_samples)
	base_params = model_params(net)
	gbase = [torch.zeros(p.size()).to(device) for p in base_params]
	hgbase = [torch.zeros(p.size()).to(device) for p in base_params]  
	### gbase
	tot_samples = 0
	for num_class in range(n_classes):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs) / T
		loss = criterion(outputs, targets)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=False)
		### update
		gbase = [ gbase1 + g1.detach().clone() for gbase1, g1 in zip(gbase, gradsH) ]
	gbase = [gbase1 / n_classes for gbase1 in gbase]

	### Hg
	for num_class in range(n_classes):
		net.zero_grad()
		inputs, targets = (d_in[n_samples * num_class: n_samples * (num_class+1)]).to(device), (d_out[n_samples * num_class: n_samples * (num_class+1)]).to(device)
		outputs = net(inputs) / T 
		loss = criterion(outputs, targets)
		gradsH = torch.autograd.grad(loss, base_params, create_graph=True)
		gnorm = 0
		for i in range(len(gbase)):
			gnorm += (gbase[i] * gradsH[i]).sum()
		gnorm.backward()
		Hg = model_grads(net)
		### update
		hgbase = [hgbase1 + hg1.detach().clone() for hgbase1, hg1 in zip(hgbase, Hg)]

	hgbase = [hgbase1.reshape(hgbase1.shape[0], -1) / n_classes for hgbase1 in hgbase]

	return hgbase

### Fisher importance ###
def cal_importance_fisher(net, arch, trainloader, num_stop=5000):
	gvec = cal_grad(net, trainloader, num_stop=num_stop)
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [(np.array(l_params[4*ind:4*ind+2]) * (np.array(gvec[4*ind:4*ind+2]))) for ind in range(int((len(l_params)-2)/4))]
		list_imp = [(list_imp[ind][0].sum(dim=1) + list_imp[ind][1].sum(dim=1)).detach().pow(2) for ind in range(len(list_imp))]
	else:
		list_imp = [((np.array(l_params[3*ind:3*ind+1]) * (np.array(gvec[3*ind:3*ind+1]))))[0].sum(dim=1).detach().pow(2) for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### BN-scale based importance ###
def cal_importance_bn(net, arch):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [l_params[4*ind+2].abs().detach().squeeze() / l_params[4*ind].shape[1] for ind in range(int((len(l_params)-2)/4))]
	else:
		list_imp = [l_params[3*ind+1].abs().detach().squeeze() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### GraSP importance ###
def cal_importance_grasp(net, arch, trainloader, n_samples=100, n_classes=100, T=200):
	hgvec = cal_hg(net, trainloader, T=T, n_samples=n_samples, n_classes=n_classes)
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [(np.array(l_params[4*ind:4*ind+2]) * (np.array(hgvec[4*ind:4*ind+2]))) for ind in range(int((len(l_params)-2)/4))]
		list_imp = [(list_imp[ind][0].sum(dim=1) + list_imp[ind][1].sum(dim=1)).detach() for ind in range(len(list_imp))]
	else:
		list_imp = [((np.array(l_params[3*ind:3*ind+1]) * (np.array(hgvec[3*ind:3*ind+1]))))[0].sum(dim=1).detach() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### TFO based importance (aka NVIDIA's Fisher implementation) ###
def cal_importance_tfo(net, arch, trainloader, num_stop=5000):
	gvec = cal_grad(net, trainloader, num_stop=num_stop)
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [((np.array(l_params[4*ind+2:4*ind+4]) * (np.array(gvec[4*ind+2:4*ind+4]))))[0].sum(dim=1).detach().pow(2) for ind in range(int((len(l_params)-2)/4))]
	else:
		list_imp = [((np.array(l_params[3*ind+1:3*ind+3]) * (np.array(gvec[3*ind+1:3*ind+3]))))[0].sum(dim=1).detach().pow(2) for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### L1-norm based importance ###
def cal_importance_l1(net, arch):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [l_params[4*ind].norm(1,1).detach() for ind in range(int((len(l_params)-2)/4))]
	else:
		list_imp = [l_params[3*ind].norm(1,1).detach() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### SFP importance ###
def cal_importance_sfp(net, arch):
	l_params = model_params(net)
	l_params = [l_params1.reshape(l_params1.shape[0], -1) for l_params1 in l_params]
	if(arch=='vgg'):
		list_imp = [l_params[4*ind].norm(2,1).detach() for ind in range(int((len(l_params)-2)/4))]
	else:
		list_imp = [l_params[3*ind].norm(2,1).detach() for ind in range(int((len(l_params)-2)/3))]
	return list_imp

### RDT importance (KL version) ###
def cal_importance_rd(net, arch, trainloader, num_stop=1000, num_of_classes=100):
	net.eval()
	sm = nn.Softmax(dim=1)
	lsm = nn.LogSoftmax(dim=1)
	num_data = 0
	base_params = model_params(net)
	imp_vec = [torch.zeros(p.size()).to(device) for p in base_params]
	for inputs, targets in trainloader:
		tmp_num_data = inputs.shape[0]
		net.zero_grad()
		outputs = net(inputs.to(device))
		prob, log_prob = sm(outputs).mean(dim=0), lsm(outputs).mean(dim=0)
		for j in range(num_of_classes):
			gradsH = torch.autograd.grad(log_prob[j], base_params, create_graph=True)
			imp_vec = [imp_vec1 + prob[j].item()*g1.detach().clone().pow(2) * float(tmp_num_data) for imp_vec1, g1 in zip(imp_vec, gradsH)]
		num_data += float(tmp_num_data)
		if(num_data > num_stop):
			break	
	imp_vec = [(imp_vec1 / imp_vec1.mean()).reshape(imp_vec1.shape[0], -1) for imp_vec1 in imp_vec]
	if(arch=='vgg'):
		list_imp = [np.array(imp_vec[4*ind:4*ind+2]) for ind in range(int((len(imp_vec)-2)/4))]
		list_imp = [(list_imp[ind][0].sum(dim=1) + list_imp[ind][1].sum(dim=1)).detach() for ind in range(len(list_imp))]
	else:
		list_imp = [imp_vec[3*ind].sum(dim=1).detach() for ind in range(int((len(imp_vec)-2)/3))]
	return list_imp

def cal_importance(net, arch, pruning_type, trainloader, num_stop=5000, num_of_classes=100, T=200):
	if(pruning_type=='orthoreg'):
		nlist = cal_importance_fisher(net, arch, trainloader, num_stop=num_stop)
	elif(pruning_type=='fisher'):
		nlist = cal_importance_fisher(net, arch, trainloader, num_stop=num_stop)
	elif(pruning_type=='tfo'):
		nlist = cal_importance_tfo(net, arch, trainloader, num_stop=num_stop)
	elif(pruning_type=='bn'):
		nlist = cal_importance_bn(net, arch)
	elif(pruning_type=='grasp'):
		nlist = cal_importance_grasp(net, arch, trainloader, n_samples=int(num_stop/num_of_classes), n_classes=num_of_classes, T=T)
	elif(pruning_type=='rdt'):
		nlist = cal_importance_rd(net, arch, trainloader, num_stop=num_stop)
	elif(pruning_type=='l1'):
		nlist = cal_importance_l1(net, arch)
	elif(pruning_type=='sfp'):
		nlist = cal_importance_sfp(net, arch)
	return nlist