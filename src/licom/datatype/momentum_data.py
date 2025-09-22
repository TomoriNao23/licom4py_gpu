from .data_abc import DataABC


class MomentumData(DataABC):

    _field  = {
        # 2D fields on A-grid (isd:ied, jsd:jed)
        '2d': [
            'ub', 'vb', 'ubp', 'vbp',
            'h0', 'h0p',
            'advx', 'advy',
            'ub_ct', 'vb_ct',
            'ub_cx', 'vb_cy',
            'vb_cx', 'ub_cy',
            'h0f', 'h0bf'
            'pax', 'pay', 'pxb', 'pyb',
            'whx', 'why'
        ],
        # 3D fields on A-grid (npz, isd:ied, jsd:jed)
        '3d': ['ua', 'va']
    }

    def __init__(self):
        super().__init__()