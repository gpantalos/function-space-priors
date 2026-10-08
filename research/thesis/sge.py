import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
import altair as alt

alt.data_transformers.enable('default', max_rows=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_density_chart(P, d=7.0, step=0.1):
  xv, yv = torch.meshgrid([
      torch.arange(-d, d, step), 
      torch.arange(-d, d, step)
  ])
  pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1)
  p_xy = P.log_prob(pos_xy.to(device)).exp().unsqueeze(-1).cpu()
  
  df = torch.cat([pos_xy, p_xy], dim=-1).numpy()
  df = pd.DataFrame({
      'x': df[:, :, 0].ravel(),
      'y': df[:, :, 1].ravel(),
      'p': df[:, :, 2].ravel(),
  })
  
  chart = alt.Chart(df).mark_point().encode(
    x='x:Q',
    y='y:Q',
    color=alt.Color('p:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=['x','y','p']
  )
  
  return chart


def get_particles_chart(X):
  df = pd.DataFrame({
      'x': X[:, 0],
      'y': X[:, 1],
  })

  chart = alt.Chart(df).mark_circle(color='red').encode(
    x='x:Q',
    y='y:Q'
  )
  
  return chart



class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY
  
# Let us initialize a reusable instance right away.
K = RBF()