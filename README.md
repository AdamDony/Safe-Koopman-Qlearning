# SafeQ: Model-Free, Optimal, and Safe Q-Learning in Koopman Eigenfunction Coordinates

Code accompanying the paper:

> Md Nur-A-Adam Dony and Syed Ali Asad Rizvi,
> "Model-Free, Optimal, and Safe Q-Learning in Koopman Eigenfunction Coordinates,"
> submitted to *IEEE Transactions on Control Systems Technology*.

SafeQ learns a safe optimal feedback for input-affine discrete-time nonlinear
systems purely from data: eigenfunction identification, quadratic Q-kernel
learning by least-squares policy iteration, a certified forward-invariant
safety region, and a model-free control-barrier filter — one pipeline, never
using the drift `f` or input map `g`.

## Contents

| Path | Description |
|---|---|
| `matlab/safe_qlearn_koopman_ALL.m` | Self-contained MATLAB script reproducing the paper's simulation studies: Example 1 (linear car lane-keeping, trivial lift) and Example 2 (nonlinear benchmark with data-identified eigenfunctions). Generates both 6-panel figures. |
| `qcar/qcar_controller.py` | Real-time SafeQ controller for the Quanser QCar1 (Section IV-C): EKF pose estimation, bounded exploration with a known probe, least-squares plant/Q-kernel identification, instrumental-variable LSPI, stability gate, discrete CBF safety filter, and speed PI loop, at 100 Hz in the QLabs Cityscape digital twin. |

## Requirements

**MATLAB study** — base MATLAB R2018b+ (uses `yline`/`xline`/`sgtitle`).
No toolboxes required; the reference DARE is solved by Riccati iteration.

```matlab
>> safe_qlearn_koopman_ALL
```

Outputs `example1_car_matlab.pdf/.png` and `example2_vaidya_matlab.pdf/.png`.

**QCar experiment** — Python 3.8+, `numpy`, `opencv-python`, `pyqtgraph`, and
the Quanser Python API (`pal`, `hal`) that ships with [Quanser Interactive
Labs](https://www.quanser.com/products/quanser-interactive-labs/) / QCar SDK.
Launch the QLabs Cityscape environment, then:

```bash
python qcar/qcar_controller.py
```

The script runs a 25 s exploration phase (2500 samples at 100 Hz), identifies
the error dynamics and safe Q-kernel from data, verifies the closed-loop
spectral radius, and deploys the learned gain through the CBF filter and rate
limiter. It works unchanged on the physical QCar1 (`IS_PHYSICAL_QCAR`).

## Citation

```bibtex
@article{dony2026safeq,
  author  = {Dony, Md Nur-A-Adam and Rizvi, Syed Ali Asad},
  title   = {Model-Free, Optimal, and Safe Q-Learning in Koopman Eigenfunction Coordinates},
  journal = {IEEE Transactions on Control Systems Technology},
  note    = {under review},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
