from wave_setups import Dataset
from spline_models import superres_2d_wave,get_Net,interpolate_wave_states_2
from operators import vector2HSV
from get_param import params,toCuda,toCpu,get_hyperparam_wave
from Logger import Logger
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time,os
import torch.nn.functional as F
#from numpy2vtk import imageToVTK
from mpl_toolkits.axes_grid1 import make_axes_locatable

#### UPDATE THESE VALUES TO CHANGE POSITION OF INTERFACE AND ASSOCIATED STIFFNESS CONST. ####
interface_location = 36
z_stiffness = torch.ones((1,1,99)).to(torch.device("cuda"))
z_stiffness[:,:,:interface_location] = 0.01
z_stiffness[:,:,interface_location] = 0.5
z_stiffness[:,:,interface_location+1:] = 0.99



torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

n_iterations_per_visualization = 1 # this value can be set to a higher integer if the cv2 visualizations impose a bottleneck on your computer and you want to speed up the simulation
save_movie=False#True#
movie_FPS = 20 # ... choose FPS as provided in visualization
params.width = 100 if params.width is None else params.width
# params.height = 200 if params.height is None else params.height
resolution_factor = params.resolution_factor
orders_z = [params.orders_z]
z_size = np.prod([i+1 for i in orders_z])
types = ["oscillator"]# further types: "box","simple","oscillator"

# initialize dataset
dataset = Dataset(params.width,hidden_size=2*z_size,interactive=True,batch_size=1,n_samples=params.n_samples,dataset_size=1,average_sequence_length=params.average_sequence_length,types=types,dt=params.dt,resolution_factor=resolution_factor)

# initialize windows / movies / mouse handler
# cv2.namedWindow('z',cv2.WINDOW_NORMAL)
# cv2.namedWindow('v',cv2.WINDOW_NORMAL)
# cv2.namedWindow('a',cv2.WINDOW_NORMAL)

if save_movie:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	movie_z = cv2.VideoWriter(f'plots/z_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.width*resolution_factor))
	movie_v = cv2.VideoWriter(f'plots/v_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.height*resolution_factor,params.width*resolution_factor))
	movie_a = cv2.VideoWriter(f'plots/a_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.height*resolution_factor,params.width*resolution_factor))

def mousePosition(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y/resolution_factor
		dataset.mousey = x/resolution_factor

# cv2.setMouseCallback("z",mousePosition)
# cv2.setMouseCallback("v",mousePosition)
# cv2.setMouseCallback("a",mousePosition)

# load fluid model
model = toCuda(get_Net(params))
logger = Logger(get_hyperparam_wave(params),use_csv=False,use_tensorboard=False)
date_time,index = logger.load_state(model,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, index: {index}")
model.eval()

FPS = 0
last_FPS = 0
last_time = time.time()

v_old = None

# simulation loop
exit_loop = False


def compute_loss(old_hidden_state, new_hidden_state, z_mask, z_cond):
	''' Returns the boundary and wave losses for a full time step at a random spacial offset'''
	offset = toCuda(torch.randn(2))
	offset[1] = params.dt # always use a time offset of dt for final loss
	z,grad_z,laplace_z,dz_dt,v,a = interpolate_wave_states_2(toCuda(old_hidden_state),toCuda(new_hidden_state),offset,dt=params.dt,orders_z=orders_z)
	loss_boundary = torch.mean(z_mask[:,:,1:-1]*((z-z_cond[:,:,1:-1])**2),dim=(1,2))
	laplace = F.interpolate(z_stiffness[:,:,:], z_stiffness.shape[2] - 1)*laplace_z
	loss_wave = torch.mean((a-laplace+params.damping*v)**2,dim=(1,2))
	return loss_boundary, loss_wave
	
while not exit_loop:
	
	# reset environment (choose new random environment from types-list and reset z / v_z field to 0)
	dataset.reset_env(0)
	
	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot(111)

	x = np.linspace(0,100,800)
	y = np.linspace(-2,2,800)
	line1, = ax.plot(x, y, 'r-')
	plt.axvline(x=interface_location, color='black', linestyle='dotted')

	boundary_loss = []
	wave_loss = []
	num_frames_between_loss_save = 10
	for i in range(params.average_sequence_length):
		
		# obtain boundary conditions / mask as well as spline coefficients of previous timestep from dataset
		z_cond,z_mask,old_hidden_state,_,_,_,_,_ = toCuda(dataset.ask())
	
		# apply wave model to obtain spline coefficients of next timestep
		new_hidden_state = model(old_hidden_state,z_cond,z_mask,z_stiffness)
		
		# feed new spline coefficients back to the dataset
		dataset.tell(toCpu(new_hidden_state))
		
		# visualize fields
			
		print(f"env_info: {dataset.env_info[0]}")
		
		# obtain interpolated field values for z,grad_z,laplace_z,v,a from spline coefficients
		z,grad_z,laplace_z,v = superres_2d_wave(new_hidden_state[0:1],orders_z,resolution_factor)
		
		# visualize field values
		# clamp values into range of [0,1]
		image = torch.clamp(0.5*z[0,0]+0.5, min=0, max=1).cpu().detach().clone()
		
		line1.set_ydata(image.numpy())
		fig.canvas.draw()
		fig.canvas.flush_events()
		
		if params.show_test_loss and i % num_frames_between_loss_save == 0:
			boundary_loss_value, wave_loss_value = compute_loss(old_hidden_state[0:1], new_hidden_state[0:1],
			 z_mask, z_cond)
			boundary_loss.append(np.log10(toCpu(boundary_loss_value).numpy()))
			wave_loss.append(np.log10(toCpu(wave_loss_value).numpy()))
		print(f"FPS: {last_FPS}")
		FPS += 1
		if time.time()-last_time>=1:
			last_time = time.time()
			last_FPS=FPS
			FPS = 0
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot(boundary_loss, '.', label='Boundary Loss')
	ax2.plot(wave_loss, '.', label='Wave Loss')
	ax2.legend()
	plt.show()

