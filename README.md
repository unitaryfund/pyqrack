# pyqrack
Pure Python bindings for the pure C++11/OpenCL Qrack quantum computer simulator library

(**PyQrack** is just pure Qrack.)

To use this package, it's helpful to be familiar with [vm6502q/qrack](https://github.com/vm6502q/qrack). Users gain **much** more control over options by building **vm6502q/qrack** and [vm6502q/pyqrack](https://github.com/vm6502q/pyqrack) from source. For advanced users, building from source is the intended primary method of **PyQrack** package distribution and use.

Import and instantiate [`QrackSimulator`](https://github.com/vm6502q/pyqrack/blob/main/pyqrack/qrack_simulator.py) instances. This simulator can perform arbitrary single qubit and controlled-single-qubit gates, as well as other specific gates like `SWAP`.

Any 2x2 bit operator matrices is represented by a list of 4 `complex` floating point numbers, in [**row-major order**](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

Single and array "`b`" parameters represent [**Pauli operator bases**](https://en.wikipedia.org/wiki/Pauli_matrices). They are specified according to the enumeration of the [`Pauli`](https://github.com/vm6502q/pyqrack/blob/main/pyqrack/pauli.py) class.

`MC[x]` and `MAC[x]` methods are controlled single bit gates, with as many control qubits as you specify via Python list `c` argument. `MCX` is multiply-controlled Pauli X, and `MACX` is "anti-"controlled Pauli X, i.e. "anti-control" activates the gate if all control bits are specifically **off**, as opposed to **on**.

The package installation directory contains a `qrack_cl_precompile` folder with executables for supported platforms, to compile OpenCL kernels once, beforehand, avoiding the need to recompile "just-in-time" every time that you load this package in a Python interpreter. If you no longer want to use precompiled kernels, or if precompilation fails, just delete the `~/.qrack` directory, or the equivalent `.qrack` sub-directory in the user home folder of your operating system.

To load the required **vm6502q/qrack** libraries from a different location, set the `PYQRACK_SHARED_LIB_PATH` environment variable. If an absolute path is set for this variable, it will be treated as an absolute path. If a relative path is used, the **pyqrack** package `qrack_system` directory is the root. For example, iOS binaries included in the **pyqrack** package can be loaded with `export PYQRACK_SHARED_LIB_PATH="qrack_lib/iOS/libqrack_pinvoke.6.2.0.dylib"`. See the installed package content for other supported default build options.

Please feel welcome to open an issue, if you'd like help. 😃
