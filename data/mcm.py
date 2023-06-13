import random
from functools import lru_cache

from .problem import Problem, T
from .arithmetic import Compare, Add, Mul


class TernaryMul(Problem):
    name = 'TernaryMul'
    dependencies = {
        Mul: lambda config: config
    }
    symbols = ['*']

    def generate(self):
        pass

    @staticmethod
    def question(args):
        return f'<GO>{"*".join(str(arg) for arg in args)}='

    @staticmethod
    def thought(args) -> list[T]:
        a1, a2, a3 = args
        return [
            T(Mul, (a1, a2)),
            T(Mul, (a1 * a2, a3), 'tail')
        ]

    @staticmethod
    def answer(args):
        a1, a2, a3 = args
        return f'{a1 * a2 * a3}<STOP>'


class TernaryAdd(Problem):
    name = 'TernaryAdd'
    dependencies = {
        Add: lambda config: config
    }
    symbols = ['+']

    def generate(self):
        pass

    @staticmethod
    def question(args):
        return f'<GO>{"+".join(str(arg) for arg in args)}='

    @staticmethod
    def thought(args) -> list[T]:
        a1, a2, a3 = args
        return [
            T(Add, (a1, a2)),
            T(Add, (a1 + a2, a3), 'tail')
        ]

    @staticmethod
    def answer(args):
        return f'{sum(args)}<STOP>'


class MCM(Problem):
    """Matrix Chain Multiplication

    E.g.,
        <GO><MCM>3×6,6×7,7×5,5×9=
            <GO><MCM>3x6=3x6;0<STOP>
            <GO><MCM>6×7,7×5,5×9=(6×7,7×5),5×9;480<STOP>
            <GO>3*6*9=162<STOP>
            <GO>0+480+162=642<STOP>
            <TAIL><MCM>3×6,6×7|7×5,5×9<ACC>3×6((6×7,7×5),5×9);642=
                <GO><MCM>3×6,6×7=3×6,6×7;126<STOP>
                <GO><MCM>7×5,5×9=7×5,5×9;315<STOP>
                <GO>3*7*9=189<STOP>
                <GO>126+315+189=630<STOP>
                <GO>642<VS>630=<GT><STOP>
                <TAIL><MCM>3×6,6×7,7×5|5×9<ACC>(3×6,6×7),(7×5,5×9);630=
                    <GO><MCM>3×6,6×7,7×5=(3×6,6×7),7×5;231<STOP>
                    <GO><MCM>5×9=5×9;0<STOP>
                    <GO>3*5*9=135<STOP>
                    <GO>231+0+135=366<STOP>
                    <GO>630<VS>366=<GT><STOP>
                    ((3×6,6×7),7×5),5×9;366<STOP>
    """
    name = 'MCM'
    dependencies = {
        TernaryMul: lambda config: config,
        TernaryAdd: lambda config: config,
        Compare: lambda config: config,
    }
    symbols = ['<MCM>', ',', '×', '(', ')', '|', ';', '<ACC>']

    def generate(self):
        dims = [
            self.log_randrange(1, 10 ** self.config['max_digits'])
            for _ in range(self.config['num'] + 1)
        ]
        mats = tuple(
            (dims[i], dims[i + 1])
            for i in range(self.config['num'])
        )
        return mats, None, None

    @staticmethod
    def question(args):
        mats, min_order, min_cost = args
        if min_order is not None:
            l_mats, r_mats = mats
            l_concat = ','.join([f'{m}×{n}' for m, n in l_mats])
            r_concat = ','.join([f'{m}×{n}' for m, n in r_mats])
            q = f'<GO><MCM>{l_concat}|{r_concat}' \
                f'<ACC>{MCM.order_text(min_order)};{min_cost}='
        else:
            concat = ','.join([f'{m}×{n}' for m, n in mats])
            q = f'<GO><MCM>{concat}='
        return q

    @staticmethod
    def thought(args) -> list[T]:
        mats, min_order, min_cost = args

        # Base cases
        if len(mats) == 1:
            return []

        if min_order is None:
            # Top-level problem
            l_mats, r_mats = mats[:1], mats[1:]
        else:
            # Middle of recursion
            l_mats, r_mats = mats

        l_args = (l_mats, None, None)
        r_args = (r_mats, None, None)
        l_order, l_cost = MCM._answer(l_args)
        r_order, r_cost = MCM._answer(r_args)
        agg_cost = l_mats[0][0] * r_mats[0][0] * r_mats[-1][1]
        thoughts = [
            T(MCM, l_args),
            T(MCM, r_args),
            T(TernaryMul, (l_mats[0][0], r_mats[0][0], r_mats[-1][1])),
            T(TernaryAdd, (l_cost, r_cost, agg_cost)),
        ]

        cost = l_cost + r_cost + agg_cost
        if min_cost is not None:
            thoughts.append(T(Compare, (cost, min_cost)))
        if min_cost is None or cost < min_cost:
            min_cost = cost
            min_order = l_order, r_order

        if len(r_mats) > 1:
            new_l_mats = l_mats + (r_mats[0],)
            new_r_mats = r_mats[1:]
            thoughts.append(
                T(MCM, ((new_l_mats, new_r_mats), min_order, min_cost), 'tail'))

        return thoughts

    @staticmethod
    def answer(args):
        order, cost = MCM._answer(args)
        return f'{MCM.order_text(order)};{cost}<STOP>'

    @staticmethod
    @lru_cache(50000)
    def _answer(args):
        mats, min_order, min_cost = args
        if len(mats) == 1:
            assert min_order is None
            return mats, 0
        if min_order is None:
            l_mats, r_mats = mats[:1], mats[1:]
        else:
            l_mats, r_mats = mats
        l_order, l_cost = MCM._answer((l_mats, None, None))
        r_order, r_cost = MCM._answer((r_mats, None, None))
        agg_cost = l_mats[0][0] * r_mats[0][0] * r_mats[-1][1]
        cost = l_cost + r_cost + agg_cost
        if min_cost is None or cost < min_cost:
            min_cost = cost
            min_order = l_order, r_order

        if len(r_mats) == 1:
            return min_order, min_cost
        else:
            new_l_mats = l_mats + (r_mats[0],)
            new_r_mats = r_mats[1:]
            return MCM._answer(((new_l_mats, new_r_mats), min_order, min_cost))

    @staticmethod
    @lru_cache(50000)
    def order_text(order):
        if len(order) == 1:
            m, n = order[0]
            return f'{m}×{n}'

        order_l, order_r = order
        text_l = MCM.order_text(order_l)
        text_r = MCM.order_text(order_r)
        if len(order_l) > 1:
            text_l = f'({text_l})'
        if len(order_r) > 1:
            text_r = f'({text_r})'
        return f'{text_l},{text_r}'
