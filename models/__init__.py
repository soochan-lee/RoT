from .model import Model
from .transformer import Transformer
from .lstm import LSTM

MODEL = {
    'Transformer': Transformer,
    'LSTM': LSTM,
}
