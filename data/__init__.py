from .arithmetic import *
from .equation import *
from .sort import Min, SelectionSort, Merge, MergeSort, BubbleSort
from .sequence import LCS, LPS
from .mcm import MCM
from .knapsack import Knapsack
from .problem import Problem, ProblemSet

PROBLEM = {
    'Compare': Compare,
    'Add': Add,
    'Sub': Sub,
    'Sub_pos_int': Sub_pos_int,
    'Mul': Mul,
    'Div': Div,
    'Gcd': Gcd,
    'Lcm': Lcm,
    'Reduce': Reduce,
    'Add_frac': Add_frac,
    'Sub_frac': Sub_frac,
    'Mul_frac': Mul_frac,
    'Div_frac': Div_frac,
    # 'Add_neg': Add_neg,
    # 'Sub_neg': Sub_neg,
    # 'Mul_neg': Mul_neg,
    # 'Div_neg': Div_neg,
    'Operations': Operations,
    'Linear_1d': Linear_1d,
    'Linear_2d': Linear_2d,
    'Mul_both': Mul_both,
    'Elim': Elim,
    'Substitute': Substitute,
    # 'Quadratic_1d': Quadratic_1d,
    Min.name: Min,
    SelectionSort.name: SelectionSort,
    Merge.name: Merge,
    MergeSort.name: MergeSort,
    BubbleSort.name: BubbleSort,
    LCS.name: LCS,
    LPS.name: LPS,
    MCM.name: MCM,
    Knapsack.name: Knapsack,
}
