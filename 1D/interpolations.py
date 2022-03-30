import torch
import math
import torch.nn.functional as F
from const import TORCH_DTYPE

def monomial_derivative(order, x, derivative = 0):
    """
    Returns the value of the derivative of a monomial at a point x.
    If the derivative = 0, returns the value of the monomial itself at x.
    """
    if derivative > order:
        return 0
    resulting_order = order - derivative
    return math.factorial(order) / math.factorial(resulting_order) * x** resulting_order

def get_monomial_values(highest_order, x, derivative):
    """
    Returns the values of the nth derivative of monomials of order i, where i is an integer ranging
    from 0 to highest_order (inclusive).
    
    Values are returned in a numpy array in order of increasing order.
    When derivative = 0, this corresponds to the values of the monomials themselves.
    """
    values = torch.zeros(highest_order + 1)
    for order in range(len(values)):
        values[order] = monomial_derivative(order, x, derivative)
    return values

def create_monomial_val_matrix(num_monomials, endpoints):
    
    if num_monomials % 2 != 0:
        raise ValueError("num_monomials must be even as the functions are evaluated at two endpoints")
        
    # sort endpoints so that they are in order of increasing absolute value
    # this is needed since we compute coefficients to the right, and to the left, of 0.
    # wish to be able to reuse a single function
    endpoints.sort(key=abs)

    shape = (num_monomials, num_monomials)

    monomial_vals = torch.empty(shape, dtype=TORCH_DTYPE)
    
    def populate_monomial_vals_matrix(support_point_index):
        num_derivative_evals = num_monomials // 2
        for i in range(num_derivative_evals):
            monomial_vals[i + support_point_index * num_derivative_evals,:] = get_monomial_values(num_monomials - 1, 
                                                                                                  endpoints[support_point_index], 
                                                                                                  i)

    
    populate_monomial_vals_matrix(support_point_index=0)
    populate_monomial_vals_matrix(support_point_index=1)
    
    return monomial_vals

def determine_coefficients(v, endpoints):
    '''
    v: a vector containing the values of the summed function and its derivatives at the support points.
        The first half of the vector contains the values and derivatives of the summed function at x = endpoints[0]
        The second half of the vector contains the values and derivatives of the summed function at x = endpoints[1]
        For instance, if len(v) == 4, then v = [f(endpoints[0]), f'(endpoints[0]), f(endpoints[1]), f'(endpoints[1])]
    endpoints: An array of length 2 containing the endpoint values (the domain of the function ranges from 0 
            to +-endpoint[1] usually)
    returns: a numpy array containing the coefficients of each monomial degree that is used to construct
            the polynomial satisfying the value and derivative constraints at support points.
            For instance, array([ 5.,  0.,  3., -2.]) corresponds to f(x) = 5 + 3x^2 - 2x^3
    '''
    MM = create_monomial_val_matrix(len(v), endpoints)
    return torch.linalg.solve(MM, v)

def sign(x):
    s = torch.sign(x)
    s[s==0] = 1
    return s

def heaviside(x):
    return (torch.sign(x) + 1) / 2

def polynomial(weights, x):
    y = torch.zeros_like(x, dtype=TORCH_DTYPE)
    for order in range(len(weights)):
        y += weights[order] * x ** order
    return y

def kernel(v, xvals):

    left_weights = determine_coefficients(v, [-1,0])
    right_weights = determine_coefficients(v, [0,1])
    
    left_polynomial = polynomial(left_weights, xvals)
    right_polynomial = polynomial(right_weights, xvals)
    
    return heaviside(-xvals) * left_polynomial + heaviside(xvals) * right_polynomial 

def kernels(spline_order, xvals):
    num_boundary_conditions = (spline_order + 1)
    num_monomials = num_boundary_conditions * 2
    hermite_values = torch.eye(num_monomials, dtype=TORCH_DTYPE)
    
    # for hermite, we only ever alter the value / derivative at x=0
    # all other derivatives and values are 0 (i.e. at 1 and -1)
    
    hermite_values = hermite_values[:num_boundary_conditions]

    yvals = torch.empty((num_boundary_conditions, xvals.shape[0]), dtype=TORCH_DTYPE)
    for index, v in enumerate(hermite_values):
        y = kernel(v,xvals)
        # note that kernels are not normalized - range does not necessarily lie in [-1, 1]
        yvals[index,:] = y
    return yvals

def total_num_kernels(spline_order):
    return (spline_order + 1) * (spline_order + 2) // 2

def generate_kernels(num_points, highest_spline_order):
    if num_points % 2 == 0:
        raise ValueError("num_points must be odd!")
    xvals = torch.linspace(-1, 1, num_points, dtype=TORCH_DTYPE, requires_grad=True)
    num_kernels = total_num_kernels(highest_spline_order)
    all_kernels = torch.empty((num_kernels, num_points), dtype=TORCH_DTYPE)
    for i in range(0, highest_spline_order + 1):
        num_kernels_ith_order = total_num_kernels(i)
        num_kernels_prev_order = total_num_kernels(i-1)
        
        all_kernels[num_kernels_prev_order:num_kernels_ith_order,:] = kernels(i , xvals)
    return xvals, all_kernels.unsqueeze(1) # array with shape (num_kernels, num_points). Also
                                              # insert new rank into tensor to correspond to convolution API

def interpolate_kernels(coefficients, kernels):
    '''
    Interpolates a wave function by convolving the coefficients with the hermite spline kernels
    coefficients: Tensor of size (minibatch, num_kernel_types, width)
    kernels: Tensor of size (num_kernel_types, output_waves (usually 1), width)
    '''
    stride = kernels.shape[2] // 2
    waves = F.conv_transpose1d(coefficients, kernels, 
         padding = 0, stride = stride, groups=1)
    return waves[0:1], waves[1:2]

def first_order_time_interpolation(y1, y2, t):
    return (1 - t) * y1 + t * y2

def interpolate_wave_in_time(old_coefficients_z, new_coefficients_z,
                        old_coefficients_v, new_coefficients_v, zero_deriv_kernels, laplace_kernels, time_step):
    old_z = interpolate_kernels(old_coefficients_z, zero_deriv_kernels)[0]
    old_laplace_z = interpolate_kernels(old_coefficients_z, laplace_kernels)[0]
    new_z= interpolate_kernels(new_coefficients_z, zero_deriv_kernels)[0]
    new_laplace_z= interpolate_kernels(new_coefficients_z, laplace_kernels)[0]

    old_v = interpolate_kernels(old_coefficients_v, zero_deriv_kernels)[0] 
    new_v = interpolate_kernels(new_coefficients_v, zero_deriv_kernels)[0] 
    half_t = time_step / 2

    # first order interpolation at middle of time step
    z = first_order_time_interpolation(old_z, new_z, half_t)
    laplace_z = first_order_time_interpolation(old_laplace_z, new_laplace_z, half_t)
    dz_dt = (new_z - old_z) / time_step
    v = first_order_time_interpolation(old_v, new_v, half_t)
    a = (new_v - old_v) / time_step 

    return z, laplace_z, dz_dt, v, a

class KernelValuesHolder():
  def __init__(self, num_kernel_support_points, order, device=torch.device('cpu')):
    if num_kernel_support_points % 2 == 0:
      raise ValueError("number of kernel support points must be odd")
    self.num_kernel_support_points = num_kernel_support_points
    self.xvals, kernels = generate_kernels(num_kernel_support_points, order)
    num_kernels = kernels.shape[0]
    gradients = self.__get_gradient(self.xvals, kernels, num_kernels, num_kernel_support_points)
    laplacian = self.__get_gradient(self.xvals, gradients, num_kernels, num_kernel_support_points)
    self.kernel_values_and_derivs = torch.cat([kernels, laplacian], dim=0).to(device) # gradients not needed for wave equation
  
  def __get_gradient(self, xvals, kernels, num_kernels, num_kernel_support_points):
    gradients = torch.empty(num_kernels, 1, num_kernel_support_points, dtype = TORCH_DTYPE)
    for i in range(num_kernels):
      gradients[i,0,:] = torch.autograd.grad(torch.sum(kernels[i]), 
                          inputs=xvals, 
                          retain_graph=True,
                          create_graph=True)[0]
    return gradients

    
