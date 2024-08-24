import torch

def local_square_matrix_torch(grid_width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create grid coordinates
    y, x = torch.meshgrid(torch.arange(grid_width, device=device), torch.arange(grid_width, device=device))
    positions = torch.stack((x.flatten(), y.flatten()), dim=1)
    
    # Compute pairwise L1 distances
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    distances = torch.abs(diff).sum(dim=2)
    
    # Create adjacency matrix based on the distance criterion
    adjacency = (distances > 0) & (distances < 2)
    
    return adjacency.float()

