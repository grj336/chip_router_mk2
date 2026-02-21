"""
Conv spatial encoder for the grid
Processes placement canvas for spatial pattterns
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    CNN encoder for processing placement canvas.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: [int, ...] = (32, 64, 64),
        kernel_sizes: [int, ...] = (3, 3, 3),
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        """ """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes

        assert len(hidden_channels) == len(
            kernel_sizes
        ), "hidden_channels and kernel_sizes must have the same length"

        # Build conv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        channels = [in_channels] + list(hidden_channels)

        for i in range(len(hidden_channels)):
            kernel_size = kernel_sizes[i]
            padding = kernel_size // 2

            self.convs.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=not batch_norm,
                )
            )

            if batch_norm:
                self.batch_norms.append(nn.BatchNorm2d(channels[i + 1]))

        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.ReLU()

        self.output_channels = hidden_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, hidden_dim, height, width)
        """
        for conv in self.convs:
            x = conv(x)
        return x

    def get_output_dim(self) -> int:
        """
        Returns the output dimension of the encoder.
        """
        return self.hidden_dim
