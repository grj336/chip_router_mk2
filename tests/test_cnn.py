import torch

from models.cnn_encoder import CNNEncoder


def test_cnn_encoder():
    """
    Test the CNN encoder.
    """
    batch_size = 4
    height = 32
    width = 32
    in_channels = 2
    hidden_channels = (32, 64, 64)
    kernel_sizes = (3, 3, 3)

    cnn = CNNEncoder(in_channels, hidden_channels, kernel_sizes)
    grid = torch.randn(batch_size, in_channels, height, width)
    output = cnn(grid)

    assert output.shape == (batch_size, hidden_channels[-1], height, width)

    print("Creating CNN encoder...")
    cnn = CNNEncoder(in_channels=2, hidden_channels=(32, 64, 64), kernel_sizes=(3, 3, 3))

    print("CNN architecture:")
    print(cnn)
    print(f"\nOutput channels: {cnn.get_output_channels()}")

    # Create test input
    batch_size = 4
    height = 16
    width = 16

    print(f"\nCreating test input: ({batch_size}, 2, {height}, {width})")
    test_grid = torch.randn(batch_size, 2, height, width)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = cnn(test_grid)
        print(f"Output shape: {output.shape}")
        print(f"Expected: ({batch_size}, {cnn.get_output_channels()}, {height}, {width})")

    # Verify spatial dimensions preserved
    assert (
        output.shape[2] == height and output.shape[3] == width
    ), "Spatial dimensions should be preserved!"

    print("\n✅ CNN encoder working!")


if __name__ == "__main__":
    test_cnn_encoder()
