import torch
import math
import torch.nn.functional as F

def monomial_derivative(order, x, derivative = 0):
    """
    Returns the value of the derivative of a monomial at a point x.
    If the derivative = 0, returns the value of the monomial itself at x.
    """
    if derivative > order:
        return 0
    else:
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
        raise Exception("num_monomials must be even as the functions are evaluated at two endpoints")
        
    # sort endpoints so that they are in order of increasing absolute value
    # this is needed since we compute coefficients to the right, and to the left, of 0.
    # wish to be able to reuse a single function
    endpoints.sort(key=abs)

    shape = (num_monomials, num_monomials)

    monomial_vals = torch.empty(shape, dtype=torch.float64)
    
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
    y = torch.zeros_like(x, dtype=torch.float64)
    for order in range(len(weights)):
        y += weights[order] * x ** order
    return y

def kernel(v, xvals):

    left_weights = determine_coefficients(v, [-1,0])
    right_weights = determine_coefficients(v, [0,1])
    
    left_polynomial = polynomial(left_weights, xvals)
    right_polynomial = polynomial(right_weights, xvals)
    
    return heaviside(-xvals) * left_polynomial + heaviside(xvals) * right_polynomial 

def kernels(highest_order, xvals):
    hermite_values = torch.eye(highest_order + 1, dtype=torch.float64)
    
    # for hermite, we only ever alter the value / derivative at x=0
    # all other derivatives and values are 0 (i.e. at 1 and -1)
    
    num_conditions = (highest_order + 1) // 2
    hermite_values = hermite_values[:num_conditions]

    yvals = torch.empty((num_conditions, xvals.shape[0]), dtype=torch.float64)
    for index, v in enumerate(hermite_values):
        y = kernel(v,xvals)
        yvals[index,:] = y
    return yvals


def generate_kernels(num_points, order):
    if num_points % 2 == 0:
        raise Exception("num_points must be odd!")
    xvals = torch.linspace(-1, 1, num_points, dtype=torch.float64, requires_grad=True)
    return xvals, kernels(order, xvals).unsqueeze(1) # array with shape (num_kernels, num_points). Also
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
                        old_coefficients_v, new_coefficients_v, kernels, time_step):
    old_z, old_laplace_z = interpolate_kernels(old_coefficients_z, kernels)
    new_z, new_laplace_z = interpolate_kernels(new_coefficients_z, kernels)

    old_v = interpolate_kernels(old_coefficients_v, kernels)[0] 
    new_v = interpolate_kernels(new_coefficients_v, kernels)[0] 
    half_t = time_step / 2

    # first order interpolation at middle of time step
    z = first_order_time_interpolation(old_z, new_z, half_t)
    laplace_z = first_order_time_interpolation(old_laplace_z, new_laplace_z, half_t)
    dz_dt = (new_z - old_z) / time_step
    v = first_order_time_interpolation(old_v, new_v, half_t)
    a = (new_v - old_v) / time_step 

    return z, laplace_z, dz_dt, v, a

class KernelValuesHolder():
  def __init__(self, num_kernel_support_points, order):
    if num_kernel_support_points % 2 == 0:
      raise ValueError("number of kernel support points must be odd")
    self.num_kernel_support_points = num_kernel_support_points
    self.num_kernels = order -1
    self.xvals, kernels = generate_kernels(num_kernel_support_points, order)
    gradients = self.__get_gradient(self.xvals, kernels, self.num_kernels, self.num_kernel_support_points)
    laplacian = self.__get_gradient(self.xvals, gradients, self.num_kernels, self.num_kernel_support_points)
    self.kernel_values_and_derivs = torch.cat(kernels, laplacian) # gradients not needed for wave equation
  
  def __get_gradient(self, xvals, kernels, num_kernels, num_kernel_support_points):
    gradients = torch.empty(num_kernels, 1, num_kernel_support_points, dtype = torch.float64)
    for i in range(self.num_kernels):
      gradients[i,0,:] = torch.autograd.grad(torch.sum(kernels[i]), 
                          inputs=xvals, 
                          retain_graph=True,
                          create_graph=True)[0]
    return gradients

    
