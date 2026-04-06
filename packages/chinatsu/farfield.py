import numpy as np
import torch
import unittest

from pathlib import Path
from io import StringIO
from skrf import Network, Frequency

import os, sys
sys.path.append(os.path.normpath(f'{__file__}/../'))
from mna import MnaCircuit, TensorGammaZ0, TensorYNetwork

ETA0 = 120 * np.pi
POWER_PREFIX = '// Radiated/Accepted/Stimulated Power , Frequency'
FIELD_PREFIX = '// >> Phi, Theta, Re(E_Theta), Im(E_Theta), Re(E_Phi), Im(E_Phi):'
TORCH_TRAPEZOID = torch.trapezoid if hasattr(torch, 'trapezoid') else torch.trapz

def compute_snp_currents(snp: Network, device=None, stim_power=0.5):
    conns = []
    for pin in range(snp.nports):
        port = MnaCircuit.Port(snp.frequency, f'p{pin}')
        conns.append([[snp, pin], [port, 0]])
    mna = MnaCircuit(conns, device=device)
    ret = []
    for pin in range(snp.nports):
        power = np.zeros([len(snp.frequency), snp.nports])
        power[:, pin] = stim_power
        phase = np.zeros(snp.nports)
        voltages = mna.node_voltages(power, phase)
        ret.append(mna.component_currents(voltages, snp))
    return torch.stack(ret, dim=1) # shape [F, P, P]

def read_sections(file: str, prefix = '//'):
    current_prefix = ''
    current_lines: list[str] = []
    out: list[tuple[str, list[str]]] = []
    for line in Path(file).read_text(encoding='utf8').splitlines():
        line = line.strip()
        if not line:
            continue
        elif line.startswith(prefix):
            if current_prefix:
                out.append((current_prefix, current_lines))
            current_prefix = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_prefix:
        out.append((current_prefix, current_lines))
    return out

def read_section_arrays(sections, prefix: str):
    return [np.loadtxt(StringIO('\n'.join(lines))) for name, lines in sections if name == prefix]

def load_ffs(file: str):
    sections = read_sections(file)
    freq = read_section_arrays(sections, POWER_PREFIX)[0].reshape(-1, 4)[:, 3]
    fields = read_section_arrays(sections, FIELD_PREFIX)
    ffs = np.stack([
        np.stack([arr[:, 2] + 1j * arr[:, 3], arr[:, 4] + 1j * arr[:, 5]])
    for arr in fields])
    return Frequency.from_f(freq, unit='Hz'), fields[0][:, :2], ffs # shape [F, 2, L]

def load_ffs_files(ffs_files: list[str]):
    freq = Frequency(unit='GHz')
    angles, ret = np.zeros([]), []
    for file in ffs_files:
        freq, angles, ffs = load_ffs(file)
        ret.append(ffs)
    return freq, angles, np.stack(ret) # shape [P, F, 2, L]

def interp_conns(conn_list: list[list[tuple[Network, int]]], freq: Frequency):
    ntw_dict: dict[str, Network] = { }
    for conns in conn_list:
        for ntw, _ in conns:
            ntw_dict[ntw.name] = ntw
    for name, ntw in list(ntw_dict.items()):
        if np.array_equal(ntw.frequency.f, freq.f):
            continue
        ntw_dict[name] = ntw.interpolate(freq) # type: ignore
    return [
        [(ntw_dict[ntw.name], pin) for ntw, pin in conns]
    for conns in conn_list]

def farfield_grid(phi_theta: np.ndarray):
    ntheta = np.count_nonzero(np.isclose(phi_theta[:, 0], phi_theta[0, 0]))
    nphi = len(phi_theta) // ntheta
    phi = np.deg2rad(phi_theta[:, 0].reshape(nphi, ntheta))
    theta = np.deg2rad(phi_theta[:, 1].reshape(nphi, ntheta))
    if nphi > 1 and np.isclose(phi[-1, 0], phi[0, 0] + 2 * np.pi):
        phi, theta = phi[:-1], theta[:-1]
    return nphi, ntheta, phi, theta

class Farfield:
    def __init__(self, snp: Network, ffs_files: list[str], conn_list: list[list[tuple[Network, int]]], device=None) -> None:
        self.device = torch.device(device) if device is not None else None
        self.freq, angles, ffs = load_ffs_files(ffs_files)
        self.snp: Network = snp.interpolate(self.freq) # type: ignore
        self.snp = TensorYNetwork(self.snp, torch.tensor(self.snp.y, device=self.device, dtype=torch.complex128))
        self.nphi, self.ntheta, ff_phi, ff_theta = farfield_grid(angles)
        nphi_eff = ff_phi.shape[0]
        ffs = ffs.reshape(len(ffs), len(self.freq), 2, self.nphi, self.ntheta)[:, :, :, :nphi_eff]
        self.ffs = torch.as_tensor(np.ascontiguousarray(ffs), device=self.device, dtype=torch.complex128)
        self.ff_phi = torch.tensor(ff_phi[:, 0], device=self.device, dtype=torch.float64)
        self.ff_theta = torch.tensor(ff_theta[0], device=self.device, dtype=torch.float64)
        self.sin_theta = torch.sin(torch.tensor(ff_theta, device=self.device, dtype=torch.float64))
        self.nphi = nphi_eff
        self.z0 = torch.tensor(np.asarray(self.snp.z0), device=self.device, dtype=torch.complex128)
        self.koe = torch.linalg.inv(compute_snp_currents(self.snp, device=self.device))
        self.mna = MnaCircuit(interp_conns(conn_list, self.freq), device=self.device)

    def compute(self, power, phase):
        voltages = self.mna.node_voltages(power, phase)
        currents = self.mna.component_currents(voltages, self.snp)
        weights = torch.einsum('fij,fj->fi', self.koe, currents)
        ffs = torch.einsum('fp,pfcxy->fcxy', weights, self.ffs)
        density = (torch.abs(ffs[:, 0]) ** 2 + torch.abs(ffs[:, 1]) ** 2) * self.sin_theta
        prad = TORCH_TRAPEZOID(TORCH_TRAPEZOID(density, self.ff_theta, dim=2), self.ff_phi, dim=1) / (2 * ETA0)
        ant_v = torch.zeros_like(currents)
        for port in range(self.snp.nports):
            node = self.mna.node_from_port.get((self.snp.name, port), -1)
            if node >= 0:
                ant_v[:, port] = voltages[:, node]
        pstim = (torch.abs(ant_v + self.z0 * currents) ** 2 / (8 * self.z0.real)).sum(dim=1)
        return torch.where(pstim > 0, prad / pstim, torch.zeros_like(prad))

class FarfieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = Path(__file__).parent / "assets"
        cls.file1 = out / "dipole2-port1.ffs"
        cls.file2 = out / "dipole2-port2.ffs"
        cls.snp_path = out / "dipole2.s2p"

    def read_expected_efficiency(self, path: Path) -> np.ndarray:
        power = read_section_arrays(read_sections(str(path)), POWER_PREFIX)[0].reshape(-1, 4)
        return power[:, 0] / power[:, 2]

    def build_direct_farfield(self) -> Farfield:
        snp = Network(str(self.snp_path))
        ports = [MnaCircuit.Port(snp.frequency, name=f"src-{i}") for i in range(snp.nports)]
        conns = [[(snp, i), (ports[i], 0)] for i in range(snp.nports)]
        return Farfield(snp, [str(self.file1), str(self.file2)], conns, device="cpu")

    def test_efficiency_matches_ffs_header(self):
        ff = self.build_direct_farfield()
        eta_1 = ff.compute([0.5, 0.0], [0.0, 0.0]).detach().cpu().numpy()
        eta_2 = ff.compute([0.0, 0.5], [0.0, 0.0]).detach().cpu().numpy()
        self.assertLess(np.abs(eta_1 - self.read_expected_efficiency(self.file1)).max(), 1.5e-2)
        self.assertLess(np.abs(eta_2 - self.read_expected_efficiency(self.file2)).max(), 1e-2)

    def test_optimize_efficiency_with_mna_component(self):
        freq, _, _ = load_ffs(str(self.file1))
        snp: Network = Network(str(self.snp_path)).interpolate(freq) # type: ignore
        z = TensorGammaZ0(freq)
        src = MnaCircuit.Port(freq, name="src-0")
        cap_value = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
        cap = z.tensor_capacitor(cap_value, name="c-0")
        ff = Farfield(
            snp,
            [str(self.file1), str(self.file2)],
            [[(src, 0), (snp, 0), (cap, 0)], [(snp, 1), (cap, 1)]],
            device="cpu",
        )
        optimizer = torch.optim.Adam([cap_value], lr=10.0)
        ff.mna.update_tensor()
        base = ff.compute([0.5], [0.0]).mean()
        for _ in range(40):
            ff.mna.update_tensor()
            loss = -ff.compute([0.5], [0.0]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                cap_value.clamp_(1e-3, 200.0)
        ff.mna.update_tensor()
        final = ff.compute([0.5], [0.0]).mean()
        self.assertGreater(final.item(), base.item() + 0.2)
        self.assertLess(cap_value.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
