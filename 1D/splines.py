import torch

def sign(x):
  s = torch.sign(x)
  s[s == 0] = 1
  return s

def heaviside(x):
  return (torch.sign(x)+1)/2

# 1st order splines

def p1_1(offsets):
  offsets = offsets * sign(offsets)
  return 1 - offsets


p1 = [p1_1]  # list of 1st order basis splines

# 2nd order splines


def p2_1(offsets):
  offsets = offsets * sign(offsets)
  return (1 - offsets) ** 2 * (1 + 2 * offsets)


def p2_2(offsets):
  abs_offsets = offsets * sign(offsets)
  return sign(offsets) * (1 - abs_offsets) ** 2 * (abs_offsets)


p2 = [p2_1, p2_2]  # list of 2nd order basis splines

# 3rd order splines


def p3_1(offsets):
  offsets = offsets * sign(offsets)
  return (1 - offsets) ** 3 * (1 + 3 * offsets + 6 * offsets ** 2)


def p3_2(offsets):
  abs_offsets = offsets * sign(offsets)
  return (
      sign(offsets)
      * (1 - abs_offsets) ** 3
      * (abs_offsets + 3 * abs_offsets ** 2)
      * 2
  )


def p3_3(offsets):
  offsets = offsets * sign(offsets)
  return (1 - offsets) ** 3 * (0.5 * offsets ** 2) * 16


p3 = [p3_1, p3_2, p3_3]  # list of 3rd order basis splines

# 4th order splines


def p4_1(offsets):
  return (offsets - 1) ** 4 * (
      1 + 4 * offsets + 10 * offsets ** 2 + 20 * offsets ** 3
  ) * heaviside(offsets) + (-offsets - 1) ** 4 * (
      1 - 4 * offsets + 10 * offsets ** 2 - 20 * offsets ** 3
  ) * heaviside(
      -offsets
  )


def p4_2(offsets):
  return (
      (offsets - 1) ** 4
      * (1 * offsets + 4 * offsets ** 2 + 10 * offsets ** 3)
      * heaviside(offsets)
      + (-offsets - 1) ** 4
      * (1 * offsets - 4 * offsets ** 2 + 10 * offsets ** 3)
      * heaviside(-offsets)
  ) * 4


def p4_3(offsets):
  return (
      (offsets - 1) ** 4
      * (0.5 * offsets ** 2 + 2 * offsets ** 3)
      * heaviside(offsets)
      + (-offsets - 1) ** 4
      * (0.5 * offsets ** 2 - 2 * offsets ** 3)
      * heaviside(-offsets)
  ) * 32


def p4_4(offsets):
  return (
      (offsets - 1) ** 4 * (1.0 / 6.0 * offsets ** 3) * heaviside(offsets)
      + (-offsets - 1) ** 4 * (1.0 / 6.0 * offsets ** 3) * heaviside(-offsets)
  ) * 512


p4 = [p4_1, p4_2, p4_3, p4_4]

# 5th order splines


def p5_1(offsets):
  return (offsets - 1) ** 5 * (
      -1 - 5 * offsets - 15 * offsets ** 2 - 35 * offsets ** 3 - 70 * offsets ** 4
  ) * heaviside(offsets) + (-offsets - 1) ** 5 * (
      -1 + 5 * offsets - 15 * offsets ** 2 + 35 * offsets ** 3 - 70 * offsets ** 4
  ) * heaviside(
      -offsets
  )


def p5_2(offsets):
  return (
      (offsets - 1) ** 5
      * (-1 * offsets - 5 * offsets ** 2 - 15 * offsets ** 3 - 35 * offsets ** 4)
      * heaviside(offsets)
      + (-offsets - 1) ** 5
      * (-1 * offsets + 5 * offsets ** 2 - 15 * offsets ** 3 + 35 * offsets ** 4)
      * heaviside(-offsets)
  ) * 4


def p5_3(offsets):
  return (
      (offsets - 1) ** 5
      * (-0.5 * offsets ** 2 - 2.5 * offsets ** 3 - 7.5 * offsets ** 4)
      * heaviside(offsets)
      + (-offsets - 1) ** 5
      * (-0.5 * offsets ** 2 + 2.5 * offsets ** 3 - 7.5 * offsets ** 4)
      * heaviside(-offsets)
  ) * 32


def p5_4(offsets):
  return (
      (offsets - 1) ** 5
      * (-0.5 / 3.0 * offsets ** 3 - 2.5 / 3.0 * offsets ** 4)
      * heaviside(offsets)
      + (-offsets - 1) ** 5
      * (-0.5 / 3.0 * offsets ** 3 + 2.5 / 3.0 * offsets ** 4)
      * heaviside(-offsets)
  ) * 512


def p5_5(offsets):
  return (
      (offsets - 1) ** 5 * (-2.5 / 6.0 * offsets ** 4) * heaviside(offsets)
      + (-offsets - 1) ** 5 * (-2.5 / 6.0 * offsets ** 4) * heaviside(-offsets)
  ) * 1024


p5 = [p5_1, p5_2, p5_3, p5_4, p5_5]
