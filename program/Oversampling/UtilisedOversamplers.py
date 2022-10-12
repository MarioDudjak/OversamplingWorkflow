from program.Oversampling.BorderlineSMOTE import BorderlineSMOTE
from program.Oversampling.LinearSMOTE import LinearSMOTE
from program.Oversampling.NoSmoteOversampler import NoSmoteOversampler
from program.Oversampling.RandomOversampler import RandomOversampler
from program.Oversampling.RandomSMOTE import RandomSMOTE
from program.Oversampling.RandomSelectionLinearSMOTE import RandomSelectionLinearSMOTE
from program.Oversampling.RandomSelectionSMOTE import RandomSelectionSMOTE
from program.Oversampling.SMOTE import SMOTE
from program.Oversampling.SNOCC import SNOCC
from program.Oversampling.WSMOTE import WSMOTE
from program.Oversampling.WSMOTEInterpretation1 import WSMOTEInterpretation1
from program.Oversampling.WaRSMOTE import WaRSMOTE
from program.Oversampling.MAGICv1 import MAGICv1
from program.Oversampling.MAGICv2 import MAGICv2
from program.Oversampling.MAGICv3 import MAGICv3
from program.Oversampling.MAGICv4 import MAGICv4
from program.Oversampling.MAGICv5 import MAGICv5
from program.Oversampling.MAGICv6 import MAGICv6
from program.Oversampling.MAGICv7 import MAGICv7
from program.Oversampling.MAGICv8 import MAGICv8
from program.Oversampling.MAGICv9 import MAGICv9
from program.Oversampling.ReverseSMOTE import ReverseSMOTE

OVERSAMPLERS = {
    # "NoSmote": NoSmoteOversampler(),
    # # # # "SMOTEImb": SMOTEImb(),
    # "Smote1": SMOTE(k=3, N=100),
    # "Smote2": SMOTE(k=3, N=200),
    # "Smote3": SMOTE(k=3, N=500),
    # "Smote4": SMOTE(k=3, N=1000),
    # "Smotev1p1": SMOTE(k=5, N=100, version='v1'),
    # "Smotev2p1": SMOTE(k=5, N=100, version='v2'),
    # "Smotev1p2": SMOTE(k=5, N='IR', version='v1'),
    # "Smotev2p2": SMOTE(k=5, N='IR', version='v2'),
    # "Smotev1p3": SMOTE(k=3, N='IR', version='v1'),
    # "Smotev2p3": SMOTE(k=3, N='IR', version='v2'),
    # # "Smote6": SMOTE(k=5, N=200),
    # # "Smote7": SMOTE(k=5, N=500),
    # # "Smote8": SMOTE(k=5, N=1000),
    # # "Smote9": SMOTE(k=7, N=100),
    # # "Smote10": SMOTE(k=7, N=200),
    # # "Smote11": SMOTE(k=7, N=500),
    # # "Smote12": SMOTE(k=7, N=1000),
    # # "Smote13": SMOTE(k=10, N=100),
    # # "Smote14": SMOTE(k=10, N=200),
    # # "Smote15": SMOTE(k=10, N=500),
    # # "Smote16": SMOTE(k=10, N=1000),
    # "BorderlineSmote1": BorderlineSMOTE(k=5, N=100),
    # "BorderlineSmote2": BorderlineSMOTE(k=5, N=200),
    # "BorderlineSmote3": BorderlineSMOTE(k=5, N=300),
    # "BorderlineSmote4": BorderlineSMOTE(k=5, N=400),
    # "BorderlineSmote5": BorderlineSMOTE(k=5, N=500),
    # "BorderlineSmote6": BorderlineSMOTE(k=5, N=600),
    # "BorderlineSmote7": BorderlineSMOTE(k=5, N='IR'),
    # # # # # "SmoteLinear": LinearSMOTE(),
    # "RandomSmote1": RandomSMOTE(N=100),
    # "RandomSmote2": RandomSMOTE(N=200),
    # "RandomSmote3": RandomSMOTE(N=300),
    # "RandomSmote4": RandomSMOTE(N=400),
    # "RandomSmote5": RandomSMOTE(N=500),
    # "RandomSmote6": RandomSMOTE(N=600),
    # "RandomSmote7": RandomSMOTE(N='IR'),
    # # # # # "RandomSelectionSmote": RandomSelectionSMOTE(),
    # # # # "RandomSelectionLinearSmote": RandomSelectionLinearSMOTE(),
    # # # # "SmoteBorderline": BorderlineSMOTE(),
    # # # # "SmoteBorderline2": BorderlineSMOTE2(N=N, k=k, m=(2 * k) + 1),
#     "RandomOversampler1": RandomOversampler(N=100),
# "RandomOversampler2": RandomOversampler(N=200),
# "RandomOversampler3": RandomOversampler(N=300),
# "RandomOversampler4": RandomOversampler(N=400),
# "RandomOversampler5": RandomOversampler(N=500),
# "RandomOversampler6": RandomOversampler(N=600),
#     "RandomOversampler7": RandomOversampler(N='IR'),
    # # # "RandomOversampler3": RandomOversampler(N=500),
    # # # "RandomOversampler4": RandomOversampler(N=1000),
    # # # # "SNOCC": SNOCC(),
    # "WSMOTE1": WSMOTE(N=100, k=5),
    # "WSMOTE2": WSMOTE(N=200, k=5),
    # "WSMOTE3": WSMOTE(N=300, k=5),
    # "WSMOTE4": WSMOTE(N=400, k=5),
    # "WSMOTE5": WSMOTE(N=500, k=5),
    # "WSMOTE6": WSMOTE(N=600, k=5),
    # "WSMOTE7": WSMOTE(k=5, N='IR'),
    # # # # "WSMOTEInterpretation1": WSMOTEInterpretation1()
    # "WaRSMOTE": WaRSMOTE(),
    # "ReverseSMOTE1": ReverseSMOTE(k=5),
    # "ReverseSMOTE": ReverseSMOTE(k=10),
    # "MAGICv1.1.0": MAGICv1(version="v1.0"),
    # "MAGICv1.1.1": MAGICv1(version="v1.1"),
    # "MAGICv1.2.0": MAGICv1(version="v2.0"),
    # "MAGICv1.2.1": MAGICv1(version="v2.1"),
    # "MAGICv1.3.0": MAGICv1(version="v3.0"),
    # "MAGICv1.3.1": MAGICv1(version="v3.1"),
    # "MAGICv1.4.0": MAGICv1(version="v4.0"),
    # "MAGICv1.4.1": MAGICv1(version="v4.1"),
    # # "MAGICv2.1.0": MAGICv2(version="v1.0"),
    # # "MAGICv2.2.0": MAGICv2(version="v2.0"),
    # # "MAGICv2.3.0": MAGICv2(version="v3.0"),
    # # "MAGICv2.4.0": MAGICv2(version="v4.0"),
    # # "MAGICv3.1.0": MAGICv3(version="v1.0"),
    # # "MAGICv3.2.0": MAGICv3(version="v2.0"),
    # # "MAGICv3.3.0": MAGICv3(version="v3.0"),
    # # "MAGICv3.4.0": MAGICv3(version="v4.0"),
    # # "MAGICv4.2.0": MAGICv4(version="v2.0"),
    # # "MAGICv4.2.1": MAGICv4(version="v2.1"),
    # "MAGICv5.1.0": MAGICv5(version="v1.0"),
    # "MAGICv5.1.1": MAGICv5(version="v1.1"),
    # "MAGICv5.2.0": MAGICv5(version="v2.0"),
    # "MAGICv5.2.1": MAGICv5(version="v2.1"),
    # "MAGICv5.3.0": MAGICv5(version="v3.0"),
    # "MAGICv5.3.1": MAGICv5(version="v3.1"),
    # "MAGICv5.4.0": MAGICv5(version="v4.0"),
    # "MAGICv5.4.1": MAGICv5(version="v4.1"),
    # "MAGICV8.1.0": MAGICv8(version="v1.0"),
    # "MAGICv8.1.1": MAGICv8(version="v1.1"),
    # "MAGICV8.2.0": MAGICv8(version="v2.0"),
    # "MAGICv8.2.1": MAGICv8(version="v2.1"),
    # "MAGICV8.3.0": MAGICv8(version="v3.0"),
    # "MAGICv8.3.1": MAGICv8(version="v3.1"),
    # "MAGICV8.4.0": MAGICv8(version="v4.0"),
    # "MAGICv8.4.1": MAGICv8(version="v4.1"),
    # "MAGICV9.1.0": MAGICv9(version="v1.0"),
    # "MAGICv9.1.1": MAGICv9(version="v1.1"),
    "MAGICV9.2.0": MAGICv9(version="v2.0"),
    # "MAGICv9.2.1": MAGICv9(version="v2.1"),
    # "MAGICV9.3.0": MAGICv9(version="v3.0"),
    # "MAGICv9.3.1": MAGICv9(version="v3.1"),
    # "MAGICV9.4.0": MAGICv9(version="v4.0"),
    # "MAGICv9.4.1": MAGICv9(version="v4.1")
}
oversamplerIds = OVERSAMPLERS.keys()
