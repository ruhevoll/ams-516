import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y):
        inputs = torch.stack([x, y], dim=-1)
        return self.net(inputs)

# Compute the PDE residual: ∇²u = u_xx + u_yy = 0
def pde_residual(model, x, y):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    u = model(x, y)
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    return u_xx + u_yy

# Generate training data
def generate_data(n_samples=5000):
    r = np.random.uniform(0.1, 1.0, n_samples)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    u_true = np.log(np.sqrt(x**2 + y**2))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(u_true, dtype=torch.float32)

# Training the PINN
def train_pinn(model, n_epochs=10000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x, y, u_true = generate_data()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # PDE loss
        pde_loss = torch.mean(pde_residual(model, x, y)**2)
        
        # Boundary/data loss
        u_pred = model(x, y).squeeze()
        data_loss = torch.mean((u_pred - u_true)**2)
        
        # Total loss
        loss = 10*pde_loss + data_loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')

# Plotting functions
def plot_results(model):
    # Generate grid for plotting
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    U_true = np.log(R)
    U_true[R < 0.1] = np.nan  # Avoid singularity at r=0
    
    X_torch = torch.tensor(X, dtype=torch.float32).flatten()
    Y_torch = torch.tensor(Y, dtype=torch.float32).flatten()
    U_pred = model(X_torch, Y_torch).detach().numpy().reshape(100, 100)
    
    # 2D Color Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(U_pred, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title('2D Color Plot of PINN Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('pinn_2d_color.png')
    plt.close()
    
    # 2D Contour Plot
    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, U_pred, levels=20, cmap='viridis')
    plt.colorbar(label='u(x,y)')
    plt.title('2D Contour Plot of PINN Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('pinn_2d_contour.png')
    plt.close()
    
    # 3D Surface Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, U_pred, cmap='viridis')
    ax.set_title('3D Surface Plot of PINN Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    plt.savefig('pinn_3d_surface.png')
    plt.close()
    
    # 2D L2 Error Plot
    l2_error = (U_pred - U_true)**2
    l2_error[R < 0.1] = np.nan
    plt.figure(figsize=(6, 5))
    plt.imshow(l2_error, extent=[-1, 1, -1, 1], origin='lower', cmap='inferno')
    plt.colorbar(label='L2 Error')
    plt.title('2D L2 Error of PINN Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('pinn_l2_error.png')
    plt.close()

# Main execution
if __name__ == '__main__':
    model = PINN()
    train_pinn(model)
    plot_results(model)