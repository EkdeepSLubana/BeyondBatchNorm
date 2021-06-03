import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import get_model
from layer_defs import Conv_prober, Activs_prober
from config import *
import numpy as np
import os
import argparse
import pickle as pkl

######### Parser #########
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arch", help="model architecture", default='VGG', choices=['VGG', 'ResNet-56'])
parser.add_argument("--norm_type", help="Normalization layer to be used", default='BatchNorm', choices=['Plain', 'BatchNorm', 'LayerNorm', 'Instance Normalization', 'GroupNorm', 'Filter Response Normalization', 'Weight Normalization', 'Scaled Weight Standardization', 'EvoNormSO', 'EvoNormBO', 'Variance Normalization', 'Mean Centering'])
parser.add_argument("--p_grouping", help="Number of channels per group for GroupNorm", default='32', choices=['1', '0.5', '0.25', '0.125', '0.0625', '0.03125', '0.0000001', '8', '16', '32', '64'])
parser.add_argument("--conv_type", help="Convolutional layer to be used", default='Plain', choices=['Plain', 'sWS', 'WeightNormalized', 'WeightCentered'])
parser.add_argument("--probe_layers", help="Probe activations/gradients?", default='True', choices=['True', 'False'])
parser.add_argument("--cfg", help="Model configuration", default='cfg_10')
parser.add_argument("--skipinit", help="Use skipinit initialization?", default='False', choices=['True', 'False'])
parser.add_argument("--preact", help="Use preactivation variants for ResNet?", default='False', choices=['True', 'False'])
parser.add_argument("--dataset", help="CIFAR-10 or CIFAR-100", default='CIFAR-100', choices=['CIFAR-10', 'CIFAR-100'])
parser.add_argument("--batch_size", help="Batch size for DataLoader", default='256')
parser.add_argument("--init_lr", help="Initial learning rate", default='1')
parser.add_argument("--lr_warmup", help="Use a learning rate warmup?", default='False', choices=['True', 'False'])
parser.add_argument("--opt_type", help="Optimizer", default='SGD', choices=['SGD'])
parser.add_argument("--seed", help="set random generator seed", default='0')
parser.add_argument("--download", help="download CIFAR-10/-100?", default='False')
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

if not os.path.isdir('trained'):
	os.mkdir('trained')
trained_root = 'trained/'

base_sched, base_epochs, wd = base_sched_iter, base_epochs_iter, wd_base # Training configuration
total_epochs = np.sum(base_epochs)

arch = args.arch # Model architecture
norm_type = args.norm_type # Normalization layer 
probe_layers = (args.probe_layers=='True')
p_grouping = float(args.p_grouping) # Amount of grouping for GroupNorm (<1 will define a group size, e.g., 0.5 = group size of 2; >1 defines number of groups)
conv_type = args.conv_type # Convolutional layer 
cfg_use = cfg_dict[args.cfg] # Model configuration
skipinit = (args.skipinit=='True') # Use skipinit initialization
preact = (args.preact=='True') # Use pre-activation ResNet architecture
use_data = args.dataset # Dataset
bsize = int(args.batch_size) # BatchSize
init_lr = float(args.init_lr) # Initial learning rate
lr_warmup = (args.lr_warmup=='True') # Use learning rate warmup (used for stabilizing training in Filter Response Normalization by Singh and Krishnan, 2019)
opt_type = args.opt_type # Optimizer

base_sched = [1e-1 * 256 / bsize, 1e-2 * 256 / bsize] # Learning rate is linearly scaled according to batch-size
if (bsize<32):
	base_epochs = [8, 2] # 10 epochs at batch-size of 16 have 2x number of iterations as 60 epochs of batch-size 256 training

######### Print Setup #########
print("Architecture:", arch)
print("Normalization layer:", norm_type)
print("Probing On:", probe_layers)
if(norm_type=="GroupNorm"):
	print("Grouping amount:", p_grouping)
print("Convolutional layer:", conv_type)
if(arch=="VGG"):
	print("Model config:", args.cfg)
if(arch=="ResNet-56"):
	print("Skipinit:", skipinit)
print("Dataset:", use_data)
print("Batch Size:", bsize)
print("LR Warmup:", lr_warmup)
print("Optimizer:", opt_type)


######### Dataloaders #########
transform = transforms.Compose(
	[transforms.RandomHorizontalFlip(),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])
transform_test = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	 ])

# d_path = "../../early_pruning/"
d_path = "./" 
if(use_data=="CIFAR-10"):
	n_classes = 10 
	trainset = torchvision.datasets.CIFAR10(root=d_path+'datasets/cifar10/', train=True, download=(args.download=='True'), transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root=d_path+'datasets/cifar10/', train=False, download=(args.download=='True'), transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)
elif(use_data=="CIFAR-100"):
	n_classes = 100
	trainset = torchvision.datasets.CIFAR100(root=d_path+'datasets/cifar100/', train=True, download=(args.download=='True'), transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR100(root=d_path+'datasets/cifar100/', train=False, download=(args.download=='True'), transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=False, num_workers=2)
else:
	raise Exception("Not CIFAR-10/CIFAR-100")


######### Loss #########
criterion = nn.CrossEntropyLoss()

######### Optimizers #########
def get_optimizer(net, lr, wd, opt_type="SGD", total_epochs=200):
	if(opt_type=="SGD"):
		optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
	return optimizer

######### Training functions #########
# Training
def train(net):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	stop_train = False
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
		scheduler.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if(np.isnan(train_loss)):
			stop_train=True
			break
		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%.5f)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, optimizer.param_groups[0]['lr']))
	return 100. * (correct / total), stop_train

# Testing
def test(net):
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
	return 100. * (correct / total)

# Saving
def net_save(net, accs_dict):
	print('Saving Model...')
	state = {'net': net.state_dict(), 'Train_acc': accs_dict['Train'], 'Test_acc': accs_dict['Test']}
	if(lr_warmup):
			torch.save(state, trained_root + '{layer}'.format(layer=args.norm_type) + '_conv_{layer}'.format(layer=args.conv_type) + 
		'_arch_{arch_name}'.format(arch_name=args.arch) + '_cfg_' + str(len(cfg_use)) + '_probed_' + args.probe_layers + 
		'_bsize_' + args.batch_size + '_init_lr_' + args.init_lr + '_skipinit_' + args.skipinit + '_grouping_' + args.p_grouping + '_warmup_'
		+ '_seed_' + args.seed +'.pth')
	else:
		torch.save(state, trained_root + '{layer}'.format(layer=args.norm_type) + '_conv_{layer}'.format(layer=args.conv_type) + 
			'_arch_{arch_name}'.format(arch_name=args.arch) + '_cfg_' + str(len(cfg_use)) + '_probed_' + args.probe_layers + 
			'_bsize_' + args.batch_size + '_init_lr_' + args.init_lr + '_skipinit_' + args.skipinit + '_grouping_' + args.p_grouping 
			+ '_seed_' + args.seed +'.pth')

	if(probe_layers):
		props_dict = {"params_list": net.params_list, 
				  		"grads_list": net.grads_list,
						"activs_norms": [],
						"activs_corr": [],
						"activs_ranks": [],
						"std_list": [],
						"grads_norms": [],
						}

		for mod in net.modules():
			if(isinstance(mod, Activs_prober)):
				props_dict["activs_norms"].append(mod.activs_norms)
				props_dict["activs_corr"].append(mod.activs_corr)
				props_dict["activs_ranks"].append(mod.activs_ranks)
			if(isinstance(mod, Conv_prober)):
				props_dict["std_list"].append(mod.std_list)
				props_dict["grads_norms"].append(mod.grads_norms)

		print('Saving properties...')
		with open(trained_root + "properties_" + '{layer}'.format(layer=args.norm_type) + '_conv_{layer}'.format(layer=args.conv_type) + 
			'_arch_{arch_name}'.format(arch_name=args.arch) + '_cfg_' + str(len(cfg_use)) + '_probed_' + args.probe_layers + 
			'_bsize_' + args.batch_size + '_init_lr_' + args.init_lr + '_skipinit_' + args.skipinit + '_grouping_' + args.p_grouping 
			+ '_seed_' + args.seed +'.pkl', 'wb') as f:
			pkl.dump(props_dict, f)


######### Determine model, load, and train #########
net = get_model(arch, cfg_use, conv_type=conv_type, norm_type=norm_type, p_grouping=p_grouping, n_classes=n_classes, probe=probe_layers, skipinit=skipinit, preact=preact).to(device)
accs_dict = {'Train': [], 'Test': []}

# Train 
print("\n------------------ Training ------------------\n")
best_acc = 0
lr_ind = 0
epoch = 0
base_lr = init_lr * base_sched[0]
final_lr = init_lr * base_sched[-1]

if(lr_warmup):
	warmup_lr = 0 
	warmup_epochs = 1 if bsize==16 else 5
else:
	warmup_lr = base_lr
	warmup_epochs = 0

optimizer = get_optimizer(net, opt_type=opt_type, lr=warmup_lr, wd=wd, total_epochs=total_epochs)
scheduler = LR_Scheduler(optimizer, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr, num_epochs=total_epochs, base_lr=base_lr, final_lr=final_lr, iter_per_epoch=len(trainloader))


stop_train = False
while(lr_ind < len(base_sched)):
	if(stop_train):
		break
	print("\n--learning rate is {}".format(optimizer.param_groups[0]['lr']))
	for n in range(base_epochs[lr_ind]):
		print('\nEpoch: {}'.format(epoch))
		train_acc, stop_train = train(net)
		if(stop_train):
			break
		test_acc = test(net)
		accs_dict['Train'].append(train_acc)
		accs_dict['Test'].append(test_acc)
		epoch += 1
		if((bsize==256 and epoch%5==0) or (bsize<32)):
			net_save(net, accs_dict)
	lr_ind += 1