# (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
#
# "QrackSystem" is a central access point, in Python, to the vm6502q/qrack
# shared library, in C++11. For ease, it wraps all Qrack shared library
# functions with native Python method signatures.
#
# The "QrackSystem" references underlying distinct Qrack simulator instances
# via integer IDs that are created upon request to allocate a new simulator.
# While you can directly use QrackSystem to manage all your simulators, we
# suggest that you instead instantiate an instance of "QrackSimulator", which
# requests its own new simulator ID and supplies it to "QrackSystem" for all
# Qrack shared library calls, therefore acting as an independent Qrack
# simulator object.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

import os
from ctypes import *
from sys import platform as _platform


class QrackSystem:
    def __init__(self):
        shared_lib_path = ""
        if os.environ.get('PYQRACK_SHARED_LIB_PATH') != None:
            shared_lib_path = os.environ.get('PYQRACK_SHARED_LIB_PATH')
        elif _platform == "win32":
            shared_lib_path = os.path.dirname(__file__) + "/qrack_lib/qrack_pinvoke.dll"
        elif _platform == "darwin":
            shared_lib_path = os.path.dirname(__file__) + "/qrack_lib/libqrack_pinvoke.dylib"
        else:
            shared_lib_path = os.path.dirname(__file__) + "/qrack_lib/libqrack_pinvoke.so"

        try:
            self.qrack_lib = CDLL(shared_lib_path)
        except Exception as e:
            if _platform == "win32":
                shared_lib_path = "C:/Program Files/libqrack*/lib/qrack_pinvoke.lib"
            elif _platform == "darwin":
                shared_lib_path = "/usr/local/lib/qrack/libqrack_pinvoke.dylib"
            else:
                shared_lib_path = "/usr/local/lib/qrack/libqrack_pinvoke.so"

            try:
                self.qrack_lib = CDLL(shared_lib_path)
            except Exception as e:
                if _platform == "win32":
                    shared_lib_path = "C:/Program Files (x86)/libqrack*/lib/qrack_pinvoke.lib"
                elif _platform == "darwin":
                    shared_lib_path = "/usr/lib/qrack/libqrack_pinvoke.dylib"
                else:
                    shared_lib_path = "/usr/lib/qrack/libqrack_pinvoke.so"

                try:
                    self.qrack_lib = CDLL(shared_lib_path)
                except Exception as e:
                    print("IMPORTANT: Did you remember to install OpenCL, if your Qrack version was built with OpenCL?")
                    raise e

        self.fppow = 5
        if "QRACK_FPPOW" in os.environ:
            self.fppow = int(os.environ.get('QRACK_FPPOW'))
        if self.fppow < 4 or self.fppow > 7:
            raise ValueError(
                "QRACK_FPPOW environment variable must be an integer >3 and <8. (Qrack builds from 4 for fp16/half, up to 7 for fp128/quad."
            )

        # Define function signatures, up front

        # non-quantum

        self.qrack_lib.DumpIds.restype = None
        self.qrack_lib.DumpIds.argtypes = [c_ulonglong, CFUNCTYPE(None, c_ulonglong)]

        self.qrack_lib.Dump.restype = None
        self.qrack_lib.Dump.argtypes = [
            c_ulonglong,
            CFUNCTYPE(c_ulonglong, c_double, c_double)
        ]

        # These next two methods need to have c_double pointers, if PyQrack is built with fp64.
        self.qrack_lib.InKet.restype = None
        self.qrack_lib.OutKet.restype = None
        self.qrack_lib.OutProbs.restype = None

        if self.fppow < 6:
            self.qrack_lib.InKet.argtypes = [c_ulonglong, POINTER(c_float)]
            self.qrack_lib.OutKet.argtypes = [c_ulonglong, POINTER(c_float)]
            self.qrack_lib.OutProbs.argtypes = [c_ulonglong, POINTER(c_float)]
        else:
            self.qrack_lib.InKet.argtypes = [c_ulonglong, POINTER(c_double)]
            self.qrack_lib.OutKet.argtypes = [c_ulonglong, POINTER(c_double)]
            self.qrack_lib.OutProbs.argtypes = [c_ulonglong, POINTER(c_double)]

        self.qrack_lib.init.restype = c_ulonglong
        self.qrack_lib.init.argtypes = []

        self.qrack_lib.get_error.restype = c_int
        self.qrack_lib.get_error.argtypes = [c_ulonglong]

        self.qrack_lib.init_count.restype = c_ulonglong
        self.qrack_lib.init_count.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.init_count_pager.restype = c_ulonglong
        self.qrack_lib.init_count_pager.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.init_count_type.restype = c_ulonglong
        self.qrack_lib.init_count_type.argtypes = [
            c_ulonglong,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool,
            c_bool
        ]

        self.qrack_lib.init_clone.restype = c_ulonglong
        self.qrack_lib.init_clone.argtypes = [c_ulonglong]

        self.qrack_lib.destroy.restype = None
        self.qrack_lib.destroy.argtypes = [c_ulonglong]

        self.qrack_lib.seed.restype = None
        self.qrack_lib.seed.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.set_concurrency.restype = None
        self.qrack_lib.set_concurrency.argtypes = [c_ulonglong, c_ulonglong]

        # pseudo-quantum

        self.qrack_lib.ProbAll.restype = None
        if self.fppow == 5:
            self.qrack_lib.ProbAll.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
        elif self.fppow == 6:
            self.qrack_lib.ProbAll.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]

        self.qrack_lib.Prob.restype = c_double
        self.qrack_lib.Prob.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.ProbRdm.restype = c_double
        self.qrack_lib.ProbRdm.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.PermutationProb.restype = c_double
        self.qrack_lib.PermutationProb.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_bool)
        ]

        self.qrack_lib.PermutationProbRdm.restype = c_double
        self.qrack_lib.PermutationProbRdm.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_bool),
            c_bool
        ]

        self.qrack_lib.PermutationExpectation.restype = c_double
        self.qrack_lib.PermutationExpectation.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.PermutationExpectationRdm.restype = c_double
        self.qrack_lib.PermutationExpectationRdm.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_bool
        ]

        self.qrack_lib.FactorizedExpectation.restype = c_double
        self.qrack_lib.FactorizedExpectation.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.FactorizedExpectationRdm.restype = c_double
        self.qrack_lib.FactorizedExpectationRdm.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_bool
        ]

        if self.fppow == 5:
            self.qrack_lib.FactorizedExpectationFp.restype = c_double
            self.qrack_lib.FactorizedExpectationFp.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.FactorizedExpectationFpRdm.restype = c_double
            self.qrack_lib.FactorizedExpectationFpRdm.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                c_bool
            ]
            self.qrack_lib.UnitaryExpectation.restype = c_double
            self.qrack_lib.UnitaryExpectation.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.MatrixExpectation.restype = c_double
            self.qrack_lib.MatrixExpectation.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.UnitaryExpectationEigenVal.restype = c_double
            self.qrack_lib.UnitaryExpectationEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                POINTER(c_float)
            ]
            self.qrack_lib.MatrixExpectationEigenVal.restype = c_double
            self.qrack_lib.MatrixExpectationEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                POINTER(c_float)
            ]
        elif self.fppow == 6:
            self.qrack_lib.FactorizedExpectationFp.restype = c_double
            self.qrack_lib.FactorizedExpectationFp.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.FactorizedExpectationFpRdm.restype = c_double
            self.qrack_lib.FactorizedExpectationFpRdm.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                c_bool
            ]
            self.qrack_lib.UnitaryExpectation.restype = c_double
            self.qrack_lib.UnitaryExpectation.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.MatrixExpectation.restype = c_double
            self.qrack_lib.MatrixExpectation.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.UnitaryExpectationEigenVal.restype = c_double
            self.qrack_lib.UnitaryExpectationEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                POINTER(c_double)
            ]
            self.qrack_lib.MatrixExpectationEigenVal.restype = c_double
            self.qrack_lib.MatrixExpectationEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                POINTER(c_double)
            ]

        self.qrack_lib.PauliExpectation.restype = c_double
        self.qrack_lib.PauliExpectation.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.Variance.restype = c_double
        self.qrack_lib.Variance.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.VarianceRdm.restype = c_double
        self.qrack_lib.VarianceRdm.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_bool
        ]

        self.qrack_lib.FactorizedVariance.restype = c_double
        self.qrack_lib.FactorizedVariance.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.FactorizedVarianceRdm.restype = c_double
        self.qrack_lib.FactorizedVarianceRdm.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_bool
        ]

        if self.fppow == 5:
            self.qrack_lib.FactorizedVarianceFp.restype = c_double
            self.qrack_lib.FactorizedVarianceFp.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.FactorizedVarianceFpRdm.restype = c_double
            self.qrack_lib.FactorizedVarianceFpRdm.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                c_bool
            ]
            self.qrack_lib.UnitaryVariance.restype = c_double
            self.qrack_lib.UnitaryVariance.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.MatrixVariance.restype = c_double
            self.qrack_lib.MatrixVariance.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float)
            ]
            self.qrack_lib.UnitaryVarianceEigenVal.restype = c_double
            self.qrack_lib.UnitaryVarianceEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                POINTER(c_float)
            ]
            self.qrack_lib.MatrixVarianceEigenVal.restype = c_double
            self.qrack_lib.MatrixVarianceEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_float),
                POINTER(c_float)
            ]
        elif self.fppow == 6:
            self.qrack_lib.FactorizedVarianceFp.restype = c_double
            self.qrack_lib.FactorizedVarianceFp.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.FactorizedVarianceFpRdm.restype = c_double
            self.qrack_lib.FactorizedVarianceFpRdm.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                c_bool
            ]
            self.qrack_lib.UnitaryVariance.restype = c_double
            self.qrack_lib.UnitaryVariance.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.MatrixVariance.restype = c_double
            self.qrack_lib.MatrixVariance.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double)
            ]
            self.qrack_lib.UnitaryVarianceEigenVal.restype = c_double
            self.qrack_lib.UnitaryVarianceEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                POINTER(c_double)
            ]
            self.qrack_lib.MatrixVarianceEigenVal.restype = c_double
            self.qrack_lib.MatrixVarianceEigenVal.argtypes = [
                c_ulonglong,
                c_ulonglong,
                POINTER(c_ulonglong),
                POINTER(c_double),
                POINTER(c_double)
            ]

        self.qrack_lib.PauliVariance.restype = c_double
        self.qrack_lib.PauliVariance.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.JointEnsembleProbability.restype = c_double
        self.qrack_lib.JointEnsembleProbability.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_ulonglong
        ]

        self.qrack_lib.PhaseParity.restype = None
        self.qrack_lib.PhaseParity.argtypes = [
            c_ulonglong,
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.PhaseRootN.restype = None
        self.qrack_lib.PhaseRootN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.ResetAll.restype = None
        self.qrack_lib.ResetAll.argtypes = [c_ulonglong]

        # allocate and release

        self.qrack_lib.allocateQubit.restype = None
        self.qrack_lib.allocateQubit.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.release.restype = c_bool
        self.qrack_lib.release.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.num_qubits.restype = c_ulonglong
        self.qrack_lib.num_qubits.argtypes = [c_ulonglong]

        # single-qubit gates

        self.qrack_lib.X.restype = None
        self.qrack_lib.X.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.Y.restype = None
        self.qrack_lib.Y.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.Z.restype = None
        self.qrack_lib.Z.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.H.restype = None
        self.qrack_lib.H.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.S.restype = None
        self.qrack_lib.S.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.T.restype = None
        self.qrack_lib.T.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjS.restype = None
        self.qrack_lib.AdjS.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjT.restype = None
        self.qrack_lib.AdjT.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.U.restype = None
        self.qrack_lib.U.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_double,
            c_double,
            c_double
        ]

        self.qrack_lib.Mtrx.restype = None
        self.qrack_lib.Mtrx.argtypes = [c_ulonglong, POINTER(c_double), c_ulonglong]

        # multi-controlled single-qubit gates

        self.qrack_lib.MCX.restype = None
        self.qrack_lib.MCX.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCY.restype = None
        self.qrack_lib.MCY.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCZ.restype = None
        self.qrack_lib.MCZ.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCH.restype = None
        self.qrack_lib.MCH.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCS.restype = None
        self.qrack_lib.MCS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCT.restype = None
        self.qrack_lib.MCT.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCAdjS.restype = None
        self.qrack_lib.MCAdjS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCAdjT.restype = None
        self.qrack_lib.MCAdjT.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MCU.restype = None
        self.qrack_lib.MCU.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_double,
            c_double,
            c_double
        ]

        self.qrack_lib.MCMtrx.restype = None
        self.qrack_lib.MCMtrx.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_double),
            c_ulonglong
        ]

        # multi-anti-controlled single-qubit gates

        self.qrack_lib.MACX.restype = None
        self.qrack_lib.MACX.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACY.restype = None
        self.qrack_lib.MACY.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACZ.restype = None
        self.qrack_lib.MACZ.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACH.restype = None
        self.qrack_lib.MACH.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACS.restype = None
        self.qrack_lib.MACS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACT.restype = None
        self.qrack_lib.MACT.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACAdjS.restype = None
        self.qrack_lib.MACAdjS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACAdjT.restype = None
        self.qrack_lib.MACAdjT.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        self.qrack_lib.MACU.restype = None
        self.qrack_lib.MACU.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_double,
            c_double,
            c_double
        ]

        self.qrack_lib.MACMtrx.restype = None
        self.qrack_lib.MACMtrx.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_double),
            c_ulonglong
        ]

        self.qrack_lib.UCMtrx.restype = None
        self.qrack_lib.UCMtrx.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_double),
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.Multiplex1Mtrx.restype = None
        self.qrack_lib.Multiplex1Mtrx.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_double)
        ]

        # coalesced single qubit gates

        self.qrack_lib.MX.restype = None
        self.qrack_lib.MX.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.MY.restype = None
        self.qrack_lib.MY.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.MZ.restype = None
        self.qrack_lib.MZ.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        # rotations

        self.qrack_lib.R.restype = None
        self.qrack_lib.R.argtypes = [c_ulonglong, c_ulonglong, c_double, c_ulonglong]

        self.qrack_lib.MCR.restype = None
        self.qrack_lib.MCR.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong
        ]

        # exponential of Pauli operators

        self.qrack_lib.Exp.restype = None
        self.qrack_lib.Exp.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_double,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCExp.restype = None
        self.qrack_lib.MCExp.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            c_double,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        # measurements

        self.qrack_lib.M.restype = c_ulonglong
        self.qrack_lib.M.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.ForceM.restype = c_ulonglong
        self.qrack_lib.ForceM.argtypes = [c_ulonglong, c_ulonglong, c_bool]

        self.qrack_lib.MAll.restype = c_ulonglong
        self.qrack_lib.MAll.argtypes = [c_ulonglong]

        self.qrack_lib.Measure.restype = c_ulonglong
        self.qrack_lib.Measure.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_int),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MeasureShots.restype = None
        self.qrack_lib.MeasureShots.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        # swap

        self.qrack_lib.SWAP.restype = None
        self.qrack_lib.SWAP.argtypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.ISWAP.restype = None
        self.qrack_lib.ISWAP.argtypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.AdjISWAP.restype = None
        self.qrack_lib.AdjISWAP.argtypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.FSim.restype = None
        self.qrack_lib.FSim.argtypes = [
            c_ulonglong,
            c_double,
            c_double,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.CSWAP.restype = None
        self.qrack_lib.CSWAP.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.ACSWAP.restype = None
        self.qrack_lib.ACSWAP.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong
        ]

        # Schmidt decomposition

        self.qrack_lib.Compose.restype = None
        self.qrack_lib.Compose.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.Decompose.restype = c_ulonglong
        self.qrack_lib.Decompose.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.Dispose.restype = None
        self.qrack_lib.Dispose.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        # (quasi-)Boolean gates

        self.qrack_lib.AND.restype = None
        self.qrack_lib.AND.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.OR.restype = None
        self.qrack_lib.OR.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.XOR.restype = None
        self.qrack_lib.XOR.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.NAND.restype = None
        self.qrack_lib.NAND.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.NOR.restype = None
        self.qrack_lib.NOR.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        self.qrack_lib.XNOR.restype = None
        self.qrack_lib.XNOR.argtypes = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            c_ulonglong
        ]

        # half classical (quasi-)Boolean gates

        self.qrack_lib.CLAND.restype = None
        self.qrack_lib.CLAND.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLOR.restype = None
        self.qrack_lib.CLOR.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLXOR.restype = None
        self.qrack_lib.CLXOR.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLNAND.restype = None
        self.qrack_lib.CLNAND.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLNOR.restype = None
        self.qrack_lib.CLNOR.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        self.qrack_lib.CLXNOR.restype = None
        self.qrack_lib.CLXNOR.argtypes = [c_ulonglong, c_bool, c_ulonglong, c_ulonglong]

        # Fourier transform

        self.qrack_lib.QFT.restype = None
        self.qrack_lib.QFT.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.IQFT.restype = None
        self.qrack_lib.IQFT.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        # Arithmetic-Logic-Unit (ALU)

        self.qrack_lib.ADD.restype = None
        self.qrack_lib.ADD.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.SUB.restype = None
        self.qrack_lib.SUB.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.ADDS.restype = None
        self.qrack_lib.ADDS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.SUBS.restype = None
        self.qrack_lib.SUBS.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MUL.restype = None
        self.qrack_lib.MUL.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.DIV.restype = None
        self.qrack_lib.DIV.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MULN.restype = None
        self.qrack_lib.MULN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.DIVN.restype = None
        self.qrack_lib.DIVN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.POWN.restype = None
        self.qrack_lib.POWN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCADD.restype = None
        self.qrack_lib.MCADD.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCSUB.restype = None
        self.qrack_lib.MCSUB.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCMUL.restype = None
        self.qrack_lib.MCMUL.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCDIV.restype = None
        self.qrack_lib.MCDIV.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCMULN.restype = None
        self.qrack_lib.MCMULN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCDIVN.restype = None
        self.qrack_lib.MCDIVN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.MCPOWN.restype = None
        self.qrack_lib.MCPOWN.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.LDA.restype = None
        self.qrack_lib.LDA.argType = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte)
        ]

        self.qrack_lib.ADC.restype = None
        self.qrack_lib.ADC.argType = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte)
        ]

        self.qrack_lib.SBC.restype = None
        self.qrack_lib.SBC.argType = [
            c_ulonglong,
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte)
        ]

        self.qrack_lib.Hash.restype = None
        self.qrack_lib.Hash.argType = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            POINTER(c_ubyte)
        ]

        # miscellaneous

        self.qrack_lib.TrySeparate1Qb.restype = c_bool
        self.qrack_lib.TrySeparate1Qb.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.TrySeparate2Qb.restype = c_bool
        self.qrack_lib.TrySeparate2Qb.argtypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.TrySeparateTol.restype = c_bool
        self.qrack_lib.TrySeparateTol.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong),
            c_double
        ]

        self.qrack_lib.Separate.restype = None
        self.qrack_lib.Separate.argtypes = [
            c_ulonglong,
            c_ulonglong,
            POINTER(c_ulonglong)
        ]

        self.qrack_lib.GetUnitaryFidelity.restype = c_double
        self.qrack_lib.GetUnitaryFidelity.argtypes = [c_ulonglong]

        self.qrack_lib.ResetUnitaryFidelity.restype = None
        self.qrack_lib.ResetUnitaryFidelity.argtypes = [c_ulonglong]

        self.qrack_lib.SetSdrp.restype = None
        self.qrack_lib.SetSdrp.argtypes = [c_ulonglong, c_double]

        self.qrack_lib.SetNcrp.restype = None
        self.qrack_lib.SetNcrp.argtypes = [c_ulonglong, c_double]

        self.qrack_lib.SetReactiveSeparate.restype = None
        self.qrack_lib.SetReactiveSeparate.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.SetTInjection.restype = None
        self.qrack_lib.SetTInjection.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.SetNoiseParameter.restype = None
        self.qrack_lib.SetNoiseParameter.argtypes = [c_ulonglong, c_double]

        self.qrack_lib.Normalize.restype = None
        self.qrack_lib.Normalize.argtypes = [c_ulonglong]

        self.qrack_lib.qstabilizer_out_to_file.restype = None
        self.qrack_lib.qstabilizer_out_to_file.argtypes = [c_ulonglong, c_char_p]

        self.qrack_lib.qstabilizer_in_from_file.restype = None
        self.qrack_lib.qstabilizer_in_from_file.argtypes = [c_ulonglong, c_char_p]

        self.qrack_lib.init_qneuron.restype = c_ulonglong
        self.qrack_lib.init_qneuron.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong), c_ulonglong, c_ulonglong, c_double, c_double]

        self.qrack_lib.clone_qneuron.restype = c_ulonglong
        self.qrack_lib.clone_qneuron.argtypes = [c_ulonglong]

        self.qrack_lib.destroy_qneuron.restype = None
        self.qrack_lib.destroy_qneuron.argtypes = [c_ulonglong]

        self.qrack_lib.set_qneuron_angles.restype = None
        self.qrack_lib.get_qneuron_angles.restype = None

        if self.fppow == 5:
            self.qrack_lib.set_qneuron_angles.argtypes = [c_ulonglong, POINTER(c_float)]
            self.qrack_lib.get_qneuron_angles.argtypes = [c_ulonglong, POINTER(c_float)]
        elif self.fppow == 6:
            self.qrack_lib.set_qneuron_angles.argtypes = [c_ulonglong, POINTER(c_double)]
            self.qrack_lib.get_qneuron_angles.argtypes = [c_ulonglong, POINTER(c_double)]

        self.qrack_lib.set_qneuron_alpha.restype = None
        self.qrack_lib.set_qneuron_alpha.argtypes = [c_ulonglong, c_double]

        self.qrack_lib.get_qneuron_alpha.restype = c_double
        self.qrack_lib.get_qneuron_alpha.argtypes = [c_ulonglong]

        self.qrack_lib.set_qneuron_activation_fn.restype = None
        self.qrack_lib.set_qneuron_activation_fn.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.get_qneuron_activation_fn.restype = c_ulonglong
        self.qrack_lib.get_qneuron_activation_fn.argtypes = [c_ulonglong]

        self.qrack_lib.qneuron_predict.restype = c_double
        self.qrack_lib.qneuron_predict.argtypes = [c_ulonglong, c_bool, c_bool]

        self.qrack_lib.qneuron_unpredict.restype = c_double
        self.qrack_lib.qneuron_unpredict.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.qneuron_learn_cycle.restype = c_double
        self.qrack_lib.qneuron_learn_cycle.argtypes = [c_ulonglong, c_bool]

        self.qrack_lib.qneuron_learn.restype = None
        self.qrack_lib.qneuron_learn.argtypes = [c_ulonglong, c_double, c_bool, c_bool]

        self.qrack_lib.qneuron_learn_permutation.restype = None
        self.qrack_lib.qneuron_learn_permutation.argtypes = [c_ulonglong, c_double, c_bool, c_bool]

        self.qrack_lib.init_qcircuit.restype = c_ulonglong
        self.qrack_lib.init_qcircuit.argtypes = [c_bool, c_bool]

        self.qrack_lib.init_qcircuit_clone.restype = c_ulonglong
        self.qrack_lib.init_qcircuit_clone.argtypes = [c_ulonglong]

        self.qrack_lib.qcircuit_inverse.restype = c_ulonglong
        self.qrack_lib.qcircuit_inverse.argtypes = [c_ulonglong]

        self.qrack_lib.qcircuit_past_light_cone.restype = c_ulonglong
        self.qrack_lib.qcircuit_past_light_cone.argtypes = [c_ulonglong, c_ulonglong, POINTER(c_ulonglong)]

        self.qrack_lib.destroy_qcircuit.restype = None
        self.qrack_lib.destroy_qcircuit.argtypes = [c_ulonglong]

        self.qrack_lib.get_qcircuit_qubit_count.restype = c_ulonglong
        self.qrack_lib.get_qcircuit_qubit_count.argtypes = [c_ulonglong]

        self.qrack_lib.qcircuit_swap.restype = None
        self.qrack_lib.qcircuit_swap.argtypes = [c_ulonglong, c_ulonglong, c_ulonglong]

        self.qrack_lib.qcircuit_append_1qb.restype = None
        self.qrack_lib.qcircuit_append_1qb.argtypes = [c_ulonglong, POINTER(c_double), c_ulonglong]

        self.qrack_lib.qcircuit_append_mc.restype = None
        self.qrack_lib.qcircuit_append_mc.argtypes = [c_ulonglong, POINTER(c_double), c_ulonglong, POINTER(c_ulonglong), c_ulonglong, c_ulonglong]

        self.qrack_lib.qcircuit_run.restype = None
        self.qrack_lib.qcircuit_run.argtypes = [c_ulonglong, c_ulonglong]

        self.qrack_lib.qcircuit_out_to_file.restype = None
        self.qrack_lib.qcircuit_out_to_file.argtypes = [c_ulonglong, c_char_p]

        self.qrack_lib.qcircuit_in_from_file.restype = None
        self.qrack_lib.qcircuit_in_from_file.argtypes = [c_ulonglong, c_char_p]

        self.qrack_lib.qcircuit_out_to_string_length.restype = c_size_t
        self.qrack_lib.qcircuit_out_to_string_length.argtypes = [c_ulonglong]

        self.qrack_lib.qcircuit_out_to_string.restype = None
        self.qrack_lib.qcircuit_out_to_string.argtypes = [c_ulonglong, c_char_p]
