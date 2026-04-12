from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel
from skrf import Frequency

# Add packages/chinatsu to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.chinatsu.mna import MnaCircuit, TensorGammaZ0


# Pydantic models for request/response
class Point(BaseModel):
    x: float
    y: float


class BlockPin(BaseModel):
    name: str | None = None


class Block(BaseModel):
    id: str
    label: str | None = None
    pins: list[BlockPin] | None = None
    position: Point | None = None
    rotation: int = 0
    type: str = "snp"
    value: str | None = None


class BlockLinkPin(BaseModel):
    kind: str | None = "pin"
    node: str
    pin: int


class PointLinkPin(Point):
    kind: str = "point"


class Link(BaseModel):
    from_: BlockLinkPin | PointLinkPin
    to_: BlockLinkPin | PointLinkPin
    points: list[Point] | None = None

    class Config:
        # Use alias to handle 'from' and 'to' keywords
        populate_by_name = True

    def __init__(self, **data):
        # Handle both 'from'/'to' (JSON) and 'from_'/'to_' (internal)
        if "from" in data:
            data["from_"] = data.pop("from")
        if "to" in data:
            data["to_"] = data.pop("to")
        super().__init__(**data)

    @property
    def from_pin(self):
        return self.from_

    @property
    def to_pin(self):
        return self.to_


class CircuitData(BaseModel):
    blocks: list[Block]
    links: list[Link]


class SimulationRequest(BaseModel):
    circuit: CircuitData
    freq_start: float = 0.1e9  # 0.1 GHz default
    freq_stop: float = 10e9    # 10 GHz default
    freq_points: int = 1001


class SParameterResponse(BaseModel):
    frequency: list[float]  # Hz
    s_matrix: list[list[list[float]]]  # shape: [ports, ports, freq_points]


def parse_value(value_str: str | None, default_unit: str) -> float:
    """Parse component value from string like 'C=1 pF {t}' or 'R=50 Ohm'"""
    if not value_str:
        return 1.0

    # Extract number and unit using regex
    # Matches patterns like "C=1 pF", "L=1.5 nH", "R=50 Ohm"
    match = re.search(r"=\s*([\d.]+)\s*(\w+)", value_str)
    if not match:
        return 1.0

    num = float(match.group(1))
    unit = match.group(2).lower()

    # Unit conversions
    unit_multipliers = {
        "pf": 1e-12,
        "nf": 1e-9,
        "uf": 1e-6,  # microfarad
        "μf": 1e-6,
        "f": 1.0,
        "nh": 1e-9,
        "uh": 1e-6,  # microhenry
        "μh": 1e-6,
        "mh": 1e-3,
        "h": 1.0,
        "ohm": 1.0,
        "ω": 1.0,
        "kohm": 1e3,
        "kω": 1e3,
        "mohm": 1e6,
        "mω": 1e6,
    }

    multiplier = unit_multipliers.get(unit, 1.0)
    return num * multiplier


def get_port_number(block_id: str) -> int | None:
    """Extract port number from block ID like 'TermG1' -> 0, 'TermG2' -> 1"""
    match = re.search(r"\d+", block_id)
    if match:
        return int(match.group()) - 1  # Convert to 0-based index
    return None


def build_connections(circuit: CircuitData) -> tuple[list[list[tuple[Any, int]]], list[str]]:
    """
    Convert CircuitData to MnaCircuit connections format.
    Returns connections list and port block IDs in order.
    """
    from skrf import Circuit

    # Build node connectivity map
    # Each node is a list of (block_id, pin_index) connections
    node_map: dict[int, list[tuple[str, int]]] = {}
    node_counter = 0
    port_blocks: list[str] = []

    # First pass: identify all unique connection points (nodes)
    # A node is created for each unique set of connected pins
    connections: list[set[tuple[str, int]]] = []

    for link in circuit.links:
        from_pin = link.from_pin
        to_pin = link.to_pin

        # Only handle block pins (not point pins for now)
        if from_pin.kind == "point" or to_pin.kind == "point":
            continue

        from_tuple = (from_pin.node, from_pin.pin) # type: ignore
        to_tuple = (to_pin.node, to_pin.pin)       # type: ignore

        # Find or create connection set
        found_set = None
        for conn_set in connections:
            if from_tuple in conn_set or to_tuple in conn_set:
                found_set = conn_set
                break

        if found_set is None:
            found_set = set()
            connections.append(found_set)

        found_set.add(from_tuple)
        found_set.add(to_tuple)

    # Identify port blocks
    for block in circuit.blocks:
        if block.type == "port":
            port_blocks.append(block.id)

    # Sort ports by number for consistent ordering
    port_blocks.sort(key=lambda x: get_port_number(x) or 0)

    return [list(conns) for conns in connections], port_blocks


def create_component(block: Block, frequency: Frequency, z: TensorGammaZ0):
    """Create a skrf component from a Block"""
    from skrf import Circuit

    comp_type = block.type
    value_str = block.value or ""

    if comp_type == "capacitor":
        value = parse_value(value_str, "F")
        value = np.array(value)
        return z.capacitor(value, name=block.id)
    elif comp_type == "inductor":
        value = parse_value(value_str, "H")
        value = np.array(value)
        return z.inductor(value, name=block.id)
    elif comp_type == "resistor":
        value = parse_value(value_str, "Ohm")
        value = np.array(value)
        return z.resistor(value, name=block.id)
    elif comp_type == "port":
        # Ports are handled specially - create a circuit port
        port_num = get_port_number(block.id) or 0
        port = Circuit.Port(frequency, name=block.id)
        # Mark as circuit port for MnaCircuit
        port._ext_attrs = {"_is_circuit_port": True}
        return port
    elif comp_type == "ground":
        gnd = Circuit.Ground(frequency, name=block.id)
        gnd._ext_attrs = {}  # Ensure _ext_attrs exists
        return gnd
    elif comp_type == "snp":
        # For SNP blocks, we would need to load from file
        # For now, create a placeholder (user needs to implement file loading)
        # Default to a 50 Ohm resistor if no file specified
        value = np.array(50)
        return z.resistor(value, name=block.id)
    else:
        # Unknown type - default to open
        return Circuit.Open(frequency, name=block.id)


def build_mna_connections(
    circuit: CircuitData,
    frequency: Frequency,
    z: TensorGammaZ0
) -> tuple[list[list[tuple[Any, int]]], int]:
    """Build connections list in MnaCircuit format. Returns (connections, num_ports)."""
    # Create component instances and track port blocks
    components: dict[str, Any] = {}
    port_blocks: list[str] = []

    for block in circuit.blocks:
        comp = create_component(block, frequency, z)
        components[block.id] = comp
        if block.type == "port":
            port_blocks.append(block.id)

    # Sort ports by their number for consistent ordering
    port_blocks.sort(key=lambda x: get_port_number(x) or 0)

    # Build node connectivity using Union-Find
    # Keys: ("block", block_id, pin) or ("point", x, y)
    parent: dict[tuple, tuple] = {}

    def find(key: tuple) -> tuple:
        if key not in parent:
            parent[key] = key
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(key1: tuple, key2: tuple):
        root1, root2 = find(key1), find(key2)
        if root1 != root2:
            parent[root2] = root1

    # Process all links
    for link in circuit.links:
        from_pin = link.from_pin
        to_pin = link.to_pin

        # Build keys based on pin type
        if isinstance(from_pin, PointLinkPin):
            from_key = ("point", from_pin.x, from_pin.y)
        else:
            from_key = ("block", from_pin.node, from_pin.pin)

        if isinstance(to_pin, PointLinkPin):
            to_key = ("point", to_pin.x, to_pin.y)
        else:
            to_key = ("block", to_pin.node, to_pin.pin)

        union(from_key, to_key)

    # Group by root
    groups: dict[tuple, list[tuple]] = {}
    for key in parent:
        root = find(key)
        if root not in groups:
            groups[root] = []
        groups[root].append(key)

    # Convert to MnaCircuit format
    mna_connections: list[list[tuple[Any, int]]] = []
    for group_keys in groups.values():
        conn_list: list[tuple[Any, int]] = []
        for key in group_keys:
            if key[0] == "block":
                _, block_id, pin = key
                if block_id in components:
                    conn_list.append((components[block_id], pin))
        if conn_list:
            mna_connections.append(conn_list)

    # Add floating ports (ports with no connections) as single-element nodes
    # This ensures they are registered with MnaCircuit even if completely unconnected
    connected_ports = set()
    for group_keys in groups.values():
        for key in group_keys:
            if key[0] == "block":
                _, block_id, pin = key
                if block_id in port_blocks:
                    connected_ports.add(block_id)

    for port_id in port_blocks:
        if port_id not in connected_ports:
            # Port is completely unconnected - add it as a standalone node
            mna_connections.append([(components[port_id], 0)])

    return mna_connections, len(port_blocks)


def get_num_ports(circuit: CircuitData) -> int:
    """Count the number of port blocks in the circuit"""
    return sum(1 for block in circuit.blocks if block.type == "port")


async def simulate(body: SimulationRequest) -> SParameterResponse:
    """
    Simulate circuit and return S-parameters.
    """
    try:
        # Check for minimum requirements
        num_ports = get_num_ports(body.circuit)
        if num_ports < 1:
            raise HTTPException(
                status_code=400,
                detail="Circuit must have at least one port to compute S-parameters"
            )
        if num_ports > 4:
            raise HTTPException(
                status_code=400,
                detail="Maximum 4 ports supported"
            )

        # Create frequency sweep
        frequency = Frequency(
            body.freq_start,
            body.freq_stop,
            body.freq_points,
            unit='Hz'
        )

        # Skip DC point if present (scikit-rf doesn't handle DC well)
        if frequency.f[0] == 0:
            frequency = Frequency(
                frequency.start + frequency.step,
                frequency.stop,
                frequency.npoints - 1,
                unit='Hz'
            )

        z0 = np.array(50.0)  # Default reference impedance
        z = TensorGammaZ0(frequency, z0)

        # Build connections
        connections, num_ports = build_mna_connections(body.circuit, frequency, z)

        if not connections:
            raise HTTPException(
                status_code=400,
                detail="No valid connections found in circuit"
            )

        # Create and solve circuit
        mna = MnaCircuit(connections)

        # Get S-parameters
        s_params = mna.s_external  # Shape: (freq_points, num_ports, num_ports)

        # Extract magnitude (linear) for each S-parameter
        # s_params[f, output_port, input_port]
        # Convert to [output_port][input_port][frequency] format for easier consumption
        freq_list = frequency.f.tolist()

        s_matrix = [
            [
                np.abs(s_params[:, out_port, in_port]).tolist()
                for in_port in range(num_ports)
            ]
            for out_port in range(num_ports)
        ]

        return SParameterResponse(
            frequency=freq_list,
            s_matrix=s_matrix,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")
