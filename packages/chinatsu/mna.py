from typing import Callable
from skrf import Network, Frequency
from skrf.circuit import Circuit
from skrf.io.touchstone import DefinedGammaZ0
from torch import Tensor
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import torch, unittest, logging, os

OHM50 = np.array(50.0)

class TensorYNetwork(Network):
    def __init__(self, comp: Network, tensor: Tensor | Callable[[], Tensor]) -> None:
        super().__init__()
        self.tensor = tensor
        self.name = comp.name
        self.frequency = comp.frequency
        self.s = comp.s
        self.z0 = comp.z0

    def get_y(self):
        return self.tensor() if callable(self.tensor) else self.tensor

class TensorCapacitor(TensorYNetwork):
    def get_y(self):
        C = self.tensor() if callable(self.tensor) else self.tensor
        C = C * 1e-12
        w = torch.tensor(self.frequency.w, device=C.device)
        a = 1j * w * C
        y = torch.stack([
            torch.stack([a, -a], dim=1),
            torch.stack([-a, a], dim=1)
        ], dim=1)
        return y

class TensorInductor(TensorYNetwork):
    def get_y(self):
        L = self.tensor() if callable(self.tensor) else self.tensor
        L = L * 1e-9
        w = torch.tensor(self.frequency.w, device=L.device)
        a = 1.0 / (1j * w * L)
        y = torch.stack([
            torch.stack([a, -a], dim=1),
            torch.stack([-a, a], dim=1)
        ], dim=1)
        return y

class TensorGammaZ0(DefinedGammaZ0):
    def tensor_capacitor(self, value: Tensor | Callable[[], Tensor], **kwargs):
        v = (value() if callable(value) else value).detach().cpu().numpy()
        return TensorCapacitor(super().capacitor(v, **kwargs), value)
    def tensor_inductor(self, value: Tensor | Callable[[], Tensor], **kwargs):
        v = (value() if callable(value) else value).detach().cpu().numpy()
        return TensorInductor(super().inductor(v, **kwargs), value)

class MnaCircuit(Circuit):
    def parse_nodes(self, connections: list[list[tuple[Network, int]]]) -> \
            tuple[list[list[tuple[Network, int]]], list[int], dict[int, list[int]]]:
        z = DefinedGammaZ0(self.frequency, self.z_mean)
        mna_nodes = connections[0:0]
        mna_source: list[int] = []
        conn_index: dict[int, list[int]] = { }
        start_index = 0
        for conns in connections:
            node_list: list[tuple[Network, int]] = []
            node = len(mna_nodes)
            for comp, pin in conns:
                attrs = comp._ext_attrs
                if attrs.get('_is_circuit_port'):
                    resistor = z.resistor(self.z_mean, name=f'-res-{comp.name}')
                    mna_source.append(node)
                    node_list.append((resistor, pin))
                else:
                    node_list.append((comp, pin))
            conn_index[node] = list(range(start_index, start_index + len(conns)))
            mna_nodes.append(node_list)
            start_index += len(conns)
        return mna_nodes, mna_source, conn_index

    def __init__(self, connections: list[list[tuple[Network, int]]], device=None) -> None:
        super().__init__(connections)
        self.device = device
        self.z_mean = self.z0.mean()
        self.mna_nodes, self.mna_source, self.conn_index = self.parse_nodes(connections)

        self.node_from_port: dict[tuple[str, int], int] = { }
        used_comp_y: dict[str, Tensor | Callable[[], Tensor]] = { }
        for node, conns in enumerate(self.mna_nodes):
            for comp, pin in conns:
                self.node_from_port[(comp.name, pin)] = node
                comp_y = comp.get_y if hasattr(comp, 'get_y') else comp.y
                comp_y = comp_y if callable(comp_y) or torch.is_tensor(comp_y) else torch.tensor(comp_y, device=self.device)
                used_comp_y[comp.name] = comp_y

        self.mat_0 = torch.zeros([len(self.frequency), len(self.mna_nodes), len(self.mna_nodes)], device=self.device, dtype=torch.complex128)
        self.mat_0, _ = self.update_mna_mat(self.mat_0, { n: y for n, y in used_comp_y.items() if not callable(y) })
        # Note: we have `self.mat_0` here so only TensorNetworks will be used in `self.update_tensor()`
        self.mat_a, self.idx_a = self.update_mna_mat(self.mat_0, { n: y for n, y in used_comp_y.items() if callable(y) })

    def update_mna_mat(self, mat_a: Tensor, comp_y: dict[str, Tensor | Callable]):
        F = mat_a.shape[0]
        device = mat_a.device
        all_arr = [[], [], [], []]
        all_ntw: list[tuple[Tensor | Callable, list[tuple[int, int]]]] = []
        for name, ntw_y in comp_y.items():
            val_y = ntw_y() if callable(ntw_y) else ntw_y
            nodes = [self.node_from_port.get((name, i), -1) for i in range(val_y.shape[1])]
            ij_index: list[tuple[int, int]] = []
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if u >= 0 and v >= 0:
                        idx0 = torch.arange(F, device=device)
                        idx1 = torch.full((F,), u, device=device)
                        idx2 = torch.full((F,), v, device=device)
                        for idx, val in enumerate([idx0, idx1, idx2, val_y[:, i, j]]):
                            all_arr[idx].append(val)
                        ij_index.append((i, j))
            if len(ij_index):
                all_ntw.append((ntw_y, ij_index))

        if len(all_arr[0]):
            idx0, idx1, idx2, vals = [torch.cat(item) for item in all_arr]
            mat_a = mat_a.index_put((idx0, idx1, idx2), vals, accumulate=True)
            return mat_a, (idx0, idx1, idx2, all_ntw)
        else:
            return mat_a, None

    def update_tensor(self):
        if self.idx_a is not None:
            idx0, idx1, idx2, all_comps = self.idx_a
            vals = []
            for ntw_y, ij_index in all_comps:
                val_y = ntw_y() if callable(ntw_y) else ntw_y
                for i, j in ij_index:
                    vals.append(val_y[:, i, j])
            vals = torch.cat(vals)
            self.mat_a = self.mat_0.index_put((idx0, idx1, idx2), vals, accumulate=True)
        return self

    @property
    def s_tensor(self) -> Tensor:
        # https://qucs.github.io/tech/node58.html
        i0 = np.sqrt(8 / self.z_mean)

        num_src = len(self.mna_source)
        mat_b = torch.zeros([len(self.frequency), len(self.mna_nodes), num_src], device=self.device, dtype=torch.complex128)
        if num_src:
            rhs = (i0 * torch.eye(num_src, device=self.device, dtype=torch.complex128)).unsqueeze(0)
            mat_b[:, self.mna_source, :] = rhs.expand(len(self.frequency), -1, -1)
        voltage = torch.linalg.solve(self.mat_a, mat_b)
        v = voltage[:, self.mna_source, :].transpose(1, 2)

        # FIXME: is this correct?
        return v * 2 / (i0 * self.z_mean) - torch.eye(len(self.mna_source), device=self.device, dtype=torch.complex128)

    @property
    def s_external(self) -> np.ndarray:
        return self.s_tensor.cpu().detach().cpu().numpy()
    
    def node_voltages(self, power, phase) -> torch.Tensor:
        power = np.array(power, dtype=np.complex128)
        phase = np.exp(1j * np.array(phase), dtype=np.complex128)
        power = torch.tensor(power * phase, device=self.device)
        # solve `Ax=b`
        mat_b = torch.zeros([len(self.frequency), len(self.mna_nodes)], device=self.device, dtype=torch.complex128)
        mat_b[:, self.mna_source] = torch.sqrt(8 * power / self.z_mean)
        return torch.linalg.solve(self.mat_a, mat_b)
    
    def voltages(self, power, phase):
        voltages = self.node_voltages(power, phase)
        # keep output compatible with skrf
        v = torch.zeros([len(self.frequency), self.dim], device=self.device, dtype=torch.complex128)
        for j in range(len(self.mna_nodes)):
            for k in self.conn_index[j]:
                v[:, k] = voltages[:, j]
        return v.detach().cpu().numpy()
    
    def component_currents(self, voltages: Tensor, comp: Network):
        v = torch.zeros([len(self.frequency), comp.nports], device=voltages.device, dtype=torch.complex128)
        for port in range(comp.nports):
            node = self.node_from_port.get((comp.name, port), -1)
            if node >= 0:
                v[:, port] = voltages[:, node]
        comp_y = comp.get_y() if hasattr(comp, 'get_y') else comp.y # type: ignore
        i = torch.zeros([len(self.frequency), comp.nports], device=voltages.device, dtype=torch.complex128)
        for port in range(comp.nports):
            i[:, port] = (v * comp_y[:, :, port]).sum(dim=1) # type: ignore
        return i

    def currents(self, power, phase):
        voltages = self.node_voltages(power, phase)
        currents = { }
        for conn in self.mna_nodes:
            for comp, _ in conn:
                if not comp.name in currents:
                    currents[comp.name] = self.component_currents(voltages, comp)
        # keep compatible with skrf
        i = torch.zeros([len(self.frequency), self.dim], dtype=torch.complex128)
        k = 0
        for conns in self.connections:
            for comp, port in conns:
                if comp.name in currents:
                    i[:, k] = currents[comp.name][:, port]
                k += 1
        # fix for grounded ports
        for j in range(0, k, 2):
            a, b = i[:, j].real.any(), i[:, j + 1].real.any()
            if b and not a:
                i[:, j] = -i[:, j + 1]
            if a and not b:
                i[:, j + 1] = -i[:, j]
        return i.detach().cpu().numpy()

def plot_s_external(f: Frequency, s1: np.ndarray, s2: np.ndarray):
    _, m, n = s1.shape
    _, axs = plt.subplots(m, n, sharex=True, sharey=True)
    for i in range(m):
        for j in range(n):
            ax = axs[i, j] if m > 1 or n > 1 else axs #type: ignore
            ax.plot(f.f, s1[:, i, j], f.f, s2[:, i, j], '--')
            s = f'S{i+1},{j+1}'
            ax.legend([f'{s} (MNA)', f'{s} (skrf)'])
            if i == m - 1:
                ax.set_xlabel('Frequency / Hz')
            if j == 0:
                ax.set_ylabel('S11 (Abs)')
    plt.show()

class Test(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        f = Frequency(0e9, 2e9, 1001, unit='Hz')
        f = Frequency(f.start + f.step, f.stop, f.npoints - 1, unit='Hz')
        z = TensorGammaZ0(f)
        n = z.resistor(OHM50, name="r0")
        self.n, self.f, self.z = n, f, z
        p0 = Circuit.Port(f, name='port-0')
        p1 = Circuit.Port(f, name='port-1')
        g = Circuit.Ground(f, name='gnd-0')
        o = Circuit.Open(f, name='open-0')
        rs = [z.resistor(OHM50, name=f'res-{i}') for i in range(3)]
        cx_list = [
            [
                [(p0, 0), (n, 0)],
                [(n, 1), (g, 0)],
            ], [
                [(p0, 0), (n, 0)],
                [(n, 1), (o, 0)],
            ], [
                [(p0, 0), (n, 0)],
                [(n, 1), (p1, 0)],
            ], [
                [(p0, 0), (n, 0)],
                [(n, 1), (rs[0], 0)],
                *[[(ri, 1), (rj, 0)] for ri, rj in zip(rs[0:-1], rs[1:])],
                [(rs[-1], 1), (g, 0)],
            ]
        ]
        self.circuit_list = [(MnaCircuit(cx), Circuit(cx)) for cx in cx_list]

    def test_s_external(self):
        for mna, cir in self.circuit_list:
            a, b = mna.s_external, cir.s_external
            logging.warning(f'INFO: s parameter difference {np.abs(a - b).mean()}')
            if os.environ.get('SHOW_SPARA_PLOTS'):
                plot_s_external(self.f, np.abs(a), np.abs(b))
            self.assertTrue(np.allclose(a, b, atol=1e-6))
        logging.info('INFO: s parameter ok')

    def test_voltages(self):
        for mna, cir in self.circuit_list:
            sz = len(mna.mna_source)
            power, phase = [1] * sz, [0] * sz
            a, b = mna.voltages(power, phase), cir.voltages(power, phase) #type: ignore
            logging.warning(f'INFO: voltages difference {np.abs(a - b).mean()}')
            self.assertTrue(np.allclose(a, b))
        logging.info('INFO: voltages ok')

    def test_currents(self):
        for mna, cir in self.circuit_list:
            sz = len(mna.mna_source)
            power, phase = [1] * sz, [0] * sz
            a, b = mna.currents(power, phase), cir.currents(power, phase) #type: ignore
            logging.warning(f'INFO: currents difference {np.abs(a - b).mean()}')
            self.assertTrue(np.allclose(a, b, atol=1e-4))
            # FIXME: it's strange we have an error peak near center frequency
            if os.environ.get('SHOW_CURRENT_PLOTS'):
                plt.plot(mna.frequency.f, np.abs(a - b))
                plt.show()
        logging.info('INFO: currents ok')

    def test_performance(self):
        n, f, z = self.n, self.f, self.z
        p0 = Circuit.Port(f, name='port-0')
        g = Circuit.Ground(f, name='gnd-0')
        rs = [z.resistor(OHM50, name=f'res-{i}') for i in range(50)]
        cx = [
            [(p0, 0), (n, 0)],
            [(n, 1), (rs[0], 0)],
            *[[(ri, 1), (rj, 0)] for ri, rj in zip(rs[0:-1], rs[1:])],
            [(rs[-1], 1), (g, 0)],
        ]

        circuit_list: list[Circuit] = [
            Circuit(cx),
            MnaCircuit(cx),
            MnaCircuit(cx, device='cuda'),
        ]
        test_cases = {
            'v': lambda circuit: circuit.voltages([1], [0]),
            's': lambda circuit: circuit.s_tensor if hasattr(circuit, 's_tensor') else circuit.s_external,
        }
        for n, f in test_cases.items():
            circuit_time = []
            for circuit in circuit_list:
                device = getattr(circuit, 'device', None) or 'cpu'
                name = f'{circuit.__class__.__name__} ({device})'
                t0 = perf_counter()
                for _ in range(50):
                    f(circuit)
                t0 = perf_counter() - t0
                circuit_time.append(t0)
                logging.warning(f'INFO: {name} compute {n} done in {t0} s ({int(circuit_time[0]/t0)}x)')
            if len(circuit_time) == 3:
                t0, t1, t2 = circuit_time
                self.assertTrue(t0 > t1)
                self.assertTrue(t0 > t2)
    
    def create_circuit(self, repeat = 1):
        f, z = self.f, self.z
        device = 'cuda'

        p0 = Circuit.Port(f, name='port-0')
        g = Circuit.Ground(f, name='gnd-0')
        r = z.resistor(OHM50, name='r-0')

        cx = [[(p0, 0), (r, 0)]]
        next_pin = (r, 1)
        tensors: list[Tensor] = []
        networks: list[TensorYNetwork] = []
        for i in range(repeat):
            c0 = torch.tensor(10., device=device, requires_grad=True)
            l0 = torch.tensor(10., device=device, requires_grad=True)
            c = z.tensor_capacitor(c0, name=f'c-{i}')
            l = z.tensor_inductor(l0, name=f'l-{i}')
            tensors.append(c0)
            tensors.append(l0)
            networks.append(c)
            networks.append(l)
            cx.append([next_pin, (c, 0)])
            cx.append([(c, 1), (l, 0)])
            next_pin = (l, 1)
        cx.append([next_pin, (g, 0)])

        return cx, tensors, networks

    def test_optimization(self):
        f, z = self.f, self.z
        device = 'cuda'
        freqs, = np.where(np.logical_and(f.f > 0.9e9, f.f < 1.2e9))
        s11 = 0.2
        cx, tensors, networks = self.create_circuit(50)
        mna = MnaCircuit(cx, device=device)
        s0 = np.abs(mna.s_external[:, 0, 0])

        start = perf_counter()
        optimizer = torch.optim.Adamax(tensors, lr=0.5)
        for i in range(200):
            # Note: we have to update mat_a with l0 and c0
            mna.update_tensor()
            loss = torch.nn.functional.leaky_relu(mna.s_tensor[freqs, 0, 0].abs() - s11).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                logging.warning(f'ITER {i}: {loss.item()}')
        logging.warning(f'PREF: optimized in {perf_counter() - start} seconds')

        mna.update_tensor()
        s1 = np.abs(mna.s_external[:, 0, 0])
        for ntw in networks:
            ntw.y = ntw.get_y().detach().cpu().numpy()
        cir = Circuit(cx)
        s2 = np.abs(cir.s_external[:, 0, 0])

        self.assertTrue(np.allclose(s1, s2, atol=1e-5))
        if os.environ.get('SHOW_OPTIMIZE_PLOTS'):
            plt.plot(f.f, s0, f.f, s1, '--', f.f, s2, '-.', f.f[freqs], f.f[freqs] * 0 + s11, 'r')
            plt.legend(['Original', 'Optimized (MNA)', 'Verified (skrf)', 'Target'])
            plt.xlabel('Frequency / Hz')
            plt.ylabel('S11 (Abs)')
            plt.show()

if __name__ == '__main__':
    unittest.main()
