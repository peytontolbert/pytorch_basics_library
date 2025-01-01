"""Weight Initialization Module.

This module provides various weight initialization strategies for PyTorch neural networks.
It includes implementations of popular initialization methods like Xavier/Glorot and
Kaiming/He initialization, as well as basic distribution-based initializers.

Example:
    >>> import torch.nn as nn
    >>> from pytorch_basics_library.initializers import initialize_model, kaiming_normal
    >>> 
    >>> # Initialize a model with Kaiming initialization
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... )
    >>> model = initialize_model(model, 'kaiming')
    >>> 
    >>> # Or use a pre-configured initializer
    >>> model = initialize_model(model, kaiming_normal)
"""

import math
from typing import Optional, Union, Dict, TypeVar, Tuple, Type, Literal, Any, Callable
import torch
import torch.nn as nn
from .device_management import device_manager
from .tensor_utils import tensor_ops
import torch.nn.init as init

# Type variables
T = TypeVar('T', bound=torch.Tensor)
M = TypeVar('M', bound=nn.Module)
InitStr = Literal['xavier', 'kaiming', 'uniform', 'normal', 'orthogonal']

def xavier_uniform_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    **kwargs: Any
) -> torch.Tensor:
    """Xavier uniform initialization with gain support.
    
    Args:
        tensor: Tensor to initialize
        gain: Scaling factor
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
        
    Example:
        >>> tensor = torch.empty(10, 20)
        >>> xavier_uniform_(tensor, gain=2.0)
    """
    return init.xavier_uniform_(tensor, gain=gain)

def xavier_normal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    **kwargs: Any
) -> torch.Tensor:
    """Xavier normal initialization with gain support.
    
    Args:
        tensor: Tensor to initialize
        gain: Scaling factor
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.xavier_normal_(tensor, gain=gain)

def kaiming_uniform_(
    tensor: torch.Tensor,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
    **kwargs: Any
) -> torch.Tensor:
    """Kaiming uniform initialization with mode and nonlinearity support.
    
    Args:
        tensor: Tensor to initialize
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity function name
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)

def kaiming_normal_(
    tensor: torch.Tensor,
    mode: str = 'fan_in',
    nonlinearity: str = 'leaky_relu',
    **kwargs: Any
) -> torch.Tensor:
    """Kaiming normal initialization with mode and nonlinearity support.
    
    Args:
        tensor: Tensor to initialize
        mode: Either 'fan_in' or 'fan_out'
        nonlinearity: Nonlinearity function name
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)

def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1.0,
    **kwargs: Any
) -> torch.Tensor:
    """Orthogonal initialization with gain support.
    
    Args:
        tensor: Tensor to initialize
        gain: Scaling factor
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.orthogonal_(tensor, gain=gain)

def uniform_(
    tensor: torch.Tensor,
    a: float = 0.0,
    b: float = 1.0,
    **kwargs: Any
) -> torch.Tensor:
    """Uniform initialization with bounds support.
    
    Args:
        tensor: Tensor to initialize
        a: Lower bound
        b: Upper bound
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.uniform_(tensor, a=a, b=b)

def normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    **kwargs: Any
) -> torch.Tensor:
    """Normal initialization with mean and std support.
    
    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Initialized tensor
    """
    return init.normal_(tensor, mean=mean, std=std)

# Add type alias for initialization functions
InitFn = Callable[[torch.Tensor, ...], torch.Tensor]

class Initializer:
    """Base class for weight initialization strategies.
    
    This class defines the interface for weight initializers and provides
    common validation utilities. All initializer implementations should
    inherit from this class.
    
    Example:
        >>> class CustomInitializer(Initializer):
        ...     def initialize(self, tensor: torch.Tensor) -> None:
        ...         self._validate_parameters(tensor)
        ...         # Custom initialization logic here
    """
    
    @staticmethod
    def _validate_parameters(tensor: torch.Tensor) -> None:
        """Validates input tensor for initialization.
        
        Args:
            tensor: The tensor to validate
            
        Raises:
            TypeError: If tensor is not a torch.Tensor
            ValueError: If tensor is not a floating point tensor
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Parameter must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if not tensor.is_floating_point():
            raise ValueError(
                f"Only floating point tensors are supported, got {tensor.dtype}"
            )
    
    def initialize(self, tensor: T) -> None:
        """Initialize the tensor. Must be implemented by subclasses.
        
        Args:
            tensor: The tensor to initialize
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    def __call__(self, tensor: T) -> None:
        """Make initializer callable.
        
        Args:
            tensor: The tensor to initialize
        """
        self.initialize(tensor)

    def create_tensor(
        self,
        size: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Create and initialize a new tensor.
        
        Args:
            size: Shape of the tensor to create
            dtype: Data type of the tensor
            device: Device to place tensor on
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Initialized tensor
            
        Example:
            >>> initializer = XavierInitializer()
            >>> tensor = initializer.create_tensor((10, 20), device='cuda:0')
        """
        tensor = tensor_ops.create_tensor(
            [0] * len(size),  # Create empty tensor of right shape
            dtype=dtype,
            device=device
        ).resize_(size)
        tensor.requires_grad_(requires_grad)
        self.initialize(tensor)
        return tensor

    @staticmethod
    def _calculate_fans(tensor: torch.Tensor) -> Tuple[int, int]:
        """Calculate fan in and fan out.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (fan_in, fan_out)
            
        Raises:
            ValueError: If tensor has fewer than 2 dimensions
        """
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                f"Fan in and fan out require at least 2D tensor, got {dimensions}D"
            )
            
        if dimensions == 2:  # Linear
            fan_in, fan_out = tensor.size(1), tensor.size(0)
        else:  # Convolution
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            for i in range(2, dimensions):
                receptive_field_size *= tensor.size(i)
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
            
        return fan_in, fan_out

class XavierInitializer(Initializer):
    """Xavier/Glorot initialization strategy.
    
    This initializer implements the method described in:
    "Understanding the difficulty of training deep feedforward neural networks"
    - Glorot, X. & Bengio, Y. (2010)
    
    Example:
        >>> initializer = XavierInitializer(gain=2.0, distribution='normal')
        >>> tensor = torch.empty(10, 20)
        >>> # Can be used either way:
        >>> initializer.initialize(tensor)  # Using initialize method
        >>> initializer(tensor)            # Using call method
        >>> print(tensor.std())  # Should be close to gain * sqrt(2/(fan_in + fan_out))
    """
    
    def __init__(
        self,
        gain: float = 1.0,
        distribution: Literal['uniform', 'normal'] = 'uniform'
    ) -> None:
        """Initialize XavierInitializer.
        
        Args:
            gain: Scaling factor for the weights
            distribution: Type of distribution to use
            
        Raises:
            ValueError: If distribution is not 'uniform' or 'normal'
            ValueError: If gain is not positive
        """
        if gain <= 0:
            raise ValueError(f"Gain must be positive, got {gain}")
        if distribution not in ['uniform', 'normal']:
            raise ValueError(
                f"Distribution must be 'uniform' or 'normal', got {distribution}"
            )
        
        self.gain = gain
        self.distribution = distribution
    
    def initialize(self, tensor: T) -> None:
        """Initialize using Xavier/Glorot method.
        
        Args:
            tensor: The tensor to initialize
            
        Raises:
            ValueError: If tensor has fewer than 2 dimensions
        """
        self._validate_parameters(tensor)
        
        fan_in, fan_out = self._calculate_fans(tensor)
        std = self.gain * math.sqrt(2.0 / (fan_in + fan_out))
        
        if self.distribution == 'uniform':
            bound = math.sqrt(3.0) * std
            nn.init.uniform_(tensor, -bound, bound)
        else:
            nn.init.normal_(tensor, 0, std)
    
    @staticmethod
    def _calculate_fans(tensor: torch.Tensor) -> Tuple[int, int]:
        """Calculate fan in and fan out.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (fan_in, fan_out)
            
        Raises:
            ValueError: If tensor has fewer than 2 dimensions
        """
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                f"Fan in and fan out require at least 2D tensor, got {dimensions}D"
            )
            
        if dimensions == 2:  # Linear
            fan_in, fan_out = tensor.size(1), tensor.size(0)
        else:  # Convolution
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            for i in range(2, dimensions):
                receptive_field_size *= tensor.size(i)
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
            
        return fan_in, fan_out

class KaimingInitializer(Initializer):
    """Kaiming/He initialization strategy.
    
    This initializer implements the method described in:
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet"
    - He, K. et al. (2015)
    
    Example:
        >>> initializer = KaimingInitializer(mode='fan_out', nonlinearity='leaky_relu')
        >>> tensor = torch.empty(10, 20)
        >>> # Can be used either way:
        >>> initializer.initialize(tensor)  # Using initialize method
        >>> initializer(tensor)            # Using call method
    """
    
    def __init__(
        self,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        nonlinearity: str = 'relu',
        distribution: Literal['normal', 'uniform'] = 'normal'
    ) -> None:
        """Initialize KaimingInitializer.
        
        Args:
            mode: Whether to use fan_in or fan_out
            nonlinearity: The non-linear function (e.g., 'relu', 'leaky_relu')
            distribution: Type of distribution to use
            
        Raises:
            ValueError: If mode or distribution is invalid
        """
        if mode not in ['fan_in', 'fan_out']:
            raise ValueError(f"Mode must be 'fan_in' or 'fan_out', got {mode}")
        if distribution not in ['normal', 'uniform']:
            raise ValueError(
                f"Distribution must be 'normal' or 'uniform', got {distribution}"
            )
            
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution
        
        try:
            self.gain = nn.init.calculate_gain(nonlinearity)
        except ValueError as e:
            raise ValueError(f"Invalid nonlinearity: {nonlinearity}") from e
    
    def initialize(self, tensor: T) -> None:
        """Initialize using Kaiming/He method.
        
        Args:
            tensor: The tensor to initialize
        """
        self._validate_parameters(tensor)
        
        fan_in, fan_out = XavierInitializer._calculate_fans(tensor)
        fan = fan_in if self.mode == 'fan_in' else fan_out
        
        std = self.gain / math.sqrt(fan)
        if self.distribution == 'uniform':
            bound = math.sqrt(3.0) * std
            nn.init.uniform_(tensor, -bound, bound)
        else:
            nn.init.normal_(tensor, 0, std)

class UniformInitializer(Initializer):
    """Uniform distribution initialization.
    
    Example:
        >>> initializer = UniformInitializer(a=-0.5, b=0.5)
        >>> tensor = torch.empty(10, 20)
        >>> # Can be used either way:
        >>> initializer.initialize(tensor)  # Using initialize method
        >>> initializer(tensor)            # Using call method
        >>> print(tensor.min(), tensor.max())  # Should be close to (-0.5, 0.5)
    """
    
    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        """Initialize UniformInitializer.
        
        Args:
            a: Lower bound
            b: Upper bound
            
        Raises:
            ValueError: If b <= a
        """
        if b <= a:
            raise ValueError(f"Upper bound must be greater than lower bound, got a={a}, b={b}")
        
        self.a = a
        self.b = b
    
    def initialize(self, tensor: T) -> None:
        """Initialize using uniform distribution.
        
        Args:
            tensor: The tensor to initialize
        """
        self._validate_parameters(tensor)
        nn.init.uniform_(tensor, self.a, self.b)

class NormalInitializer(Initializer):
    """Normal distribution initialization.
    
    Example:
        >>> initializer = NormalInitializer(mean=0.0, std=0.01)
        >>> tensor = torch.empty(10, 20)
        >>> # Can be used either way:
        >>> initializer.initialize(tensor)  # Using initialize method
        >>> initializer(tensor)            # Using call method
        >>> print(tensor.mean(), tensor.std())  # Should be close to (0.0, 0.01)
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Initialize NormalInitializer.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation
            
        Raises:
            ValueError: If std is not positive
        """
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        
        self.mean = mean
        self.std = std
    
    def initialize(self, tensor: T) -> None:
        """Initialize using normal distribution.
        
        Args:
            tensor: The tensor to initialize
        """
        self._validate_parameters(tensor)
        nn.init.normal_(tensor, self.mean, self.std)

class OrthogonalInitializer(Initializer):
    """Orthogonal initialization strategy.
    
    Creates an orthogonal matrix using QR decomposition. This initialization
    helps preserve the magnitude of gradients during backpropagation.
    
    Example:
        >>> initializer = OrthogonalInitializer(gain=2.0)
        >>> tensor = torch.empty(10, 20)
        >>> # Can be used either way:
        >>> initializer.initialize(tensor)  # Using initialize method
        >>> initializer(tensor)            # Using call method
        >>> # Check orthogonality: product with transpose should be close to identity
        >>> product = tensor @ tensor.t()
        >>> print(torch.allclose(product, torch.eye(10), atol=1e-7))  # True
    """
    
    def __init__(self, gain: float = 1.0) -> None:
        """Initialize OrthogonalInitializer.
        
        Args:
            gain: Scaling factor for the weights
            
        Raises:
            ValueError: If gain is not positive
        """
        if gain <= 0:
            raise ValueError(f"Gain must be positive, got {gain}")
        
        self.gain = gain
    
    def initialize(self, tensor: T) -> None:
        """Initialize using orthogonal matrices.
        
        Args:
            tensor: The tensor to initialize
            
        Raises:
            ValueError: If tensor has fewer than 2 dimensions
        """
        self._validate_parameters(tensor)
        
        if tensor.dim() < 2:
            raise ValueError(
                f"Orthogonal initialization requires at least 2D tensor, got {tensor.dim()}D"
            )
        
        rows, cols = tensor.size(0), tensor.size(1)
        flattened_shape = (rows, cols * tensor[0].numel() // cols)
        
        # Generate a random matrix
        random_mat = torch.randn(flattened_shape)
        
        # Compute QR factorization
        q, r = torch.linalg.qr(random_mat)
        
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        
        # Reshape to original tensor shape
        tensor.view_as(q).copy_(q)
        tensor.mul_(self.gain)

def initialize_model(
    model: M,
    initializer: Union[InitStr, Initializer, InitFn],
    initializer_config: Optional[Dict] = None,
    device: Optional[Union[str, torch.device]] = None
) -> M:
    """Initialize weights of a model using specified initializer.
    
    Args:
        model: PyTorch model to initialize
        initializer: String name, Initializer instance, or initialization function
        initializer_config: Configuration for initializer if string name is used
        device: Device to move model to after initialization
        
    Returns:
        The initialized model
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> # Using string name
        >>> model = initialize_model(model, 'xavier', {'gain': 2.0})
        >>> # Using function
        >>> model = initialize_model(model, kaiming_normal_)
        >>> # Using initializer instance
        >>> init = KaimingInitializer(mode='fan_out')
        >>> model = initialize_model(model, init)
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Model must be an instance of nn.Module, got {type(model).__name__}"
        )
    
    if isinstance(initializer, str):
        if initializer_config is None:
            initializer_config = {}
            
        initializer = initializer.lower()
        if initializer == 'xavier':
            init = XavierInitializer(**initializer_config)
        elif initializer == 'kaiming':
            init = KaimingInitializer(**initializer_config)
        elif initializer == 'uniform':
            init = UniformInitializer(**initializer_config)
        elif initializer == 'normal':
            init = NormalInitializer(**initializer_config)
        elif initializer == 'orthogonal':
            init = OrthogonalInitializer(**initializer_config)
        else:
            raise ValueError(
                f"Unknown initializer: {initializer}. "
                "Valid options are: xavier, kaiming, uniform, normal, orthogonal"
            )
        init_fn = init.initialize
    elif isinstance(initializer, Initializer):
        init_fn = initializer.initialize
    else:
        init_fn = initializer
    
    # Initialize parameters
    for name, param in model.named_parameters():
        if 'weight' in name:
            init_fn(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    
    # Move to device if specified
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        model = model.to(device)
    
    return model

# Create default instances for common initializations
xavier_uniform = XavierInitializer(distribution='uniform')
xavier_normal = XavierInitializer(distribution='normal')
kaiming_uniform = KaimingInitializer(distribution='uniform')
kaiming_normal = KaimingInitializer(distribution='normal')
uniform = UniformInitializer()
normal = NormalInitializer()
orthogonal = OrthogonalInitializer() 