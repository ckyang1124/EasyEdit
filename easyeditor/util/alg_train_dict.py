from ..trainer import MEND
from ..trainer import SERAC, SERAC_MULTI
from ..trainer import MALMEN
from ..trainer import EFK
# TODO: Comment out temporarily as it causes circular import issues

ALG_TRAIN_DICT = {
    'MEND': MEND,
    'SERAC': SERAC,
    'SERAC_MULTI': SERAC_MULTI,
    'MALMEN': MALMEN,
    'KE': EFK,
    # 'EFK': EFK
}