from .acoustic import AcousticReservoir
from .base import PhysicalReservoir, make_mask
from .coupled import SerialPhysicalReservoir
from .hardware import HardwareReservoir
from .mutual import MutualCoupledReservoir
from .optical import OptoelectronicReservoir
from .parametric_coupling import ParametricallyCoupledReservoir

__all__ = [
    "AcousticReservoir",
    "HardwareReservoir",
    "MutualCoupledReservoir",
    "OptoelectronicReservoir",
    "ParametricallyCoupledReservoir",
    "PhysicalReservoir",
    "SerialPhysicalReservoir",
    "make_mask",
]