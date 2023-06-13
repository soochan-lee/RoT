import math
import random

from .problem import Problem, T


class Compare(Problem):
    """Compare two numbers
    E.g.,
        <GO>123<VS>234=<LT><STOP>
        <GO>84<VS>2=<GT><STOP>
        <GO>74<VS>74=<EQ><STOP>
        <GO>73847<VS>7243=<GT><STOP>
        <GO>73847<VS>73413=
            <GO>7<VS>7=<EQ><STOP>
            <GO>3847<VS>3413=<GT><STOP>
            <GT><STOP>
    """

    name = 'Compare'
    dependencies = {}
    symbols = ['<VS>', '<GT>', '<LT>', '<EQ>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        smaller = random.randrange(0, max_num)
        larger = random.randrange(0, max_num)
        if random.random() < 0.5:
            return smaller, larger
        else:
            return larger, smaller

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}<VS>{right}='

    @staticmethod
    def answer(args):
        left, right = args
        if left > right:
            return '<GT><STOP>'
        elif left < right:
            return '<LT><STOP>'
        else:
            return '<EQ><STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        # Base cases
        if left < 10 and right < 10:
            return []

        thoughts = []
        digit_l, digit_r = len(str(left)), len(str(right))
        if digit_l == digit_r:
            # Compare first digit
            l_first, r_first = int(str(left)[0]), int(str(right)[0])
            thoughts.append(T(Compare, (l_first, r_first)))
            if l_first == r_first:
                # Compare the rest
                l_rest = int(str(left)[1:])
                r_rest = int(str(right)[1:])
                thoughts.append(T(Compare, (l_rest, r_rest)))

        return thoughts


class Add(Problem):
    """Addition of two positive integers
    E.g.,
        # Without carry
        <GO>31+28=
            <GO>1+8=9<STOP>
            <GO>3+2=5<STOP>
            59<STOP>

        # With carry
        <GO>39+28=
            <GO>9+8=17<STOP>
            <GO>3+1=4<STOP>
            <GO>4+2=6<STOP>
            67<STOP>

        # Solve recursively
        <GO>394+281=
            <GO>4+1=5<STOP>
            <GO>39+28=67<STOP>
            675<STOP>
    """
    name = 'Add'
    dependencies = {}
    symbols = ['+']

    def generate(self):
        # Generate [0, 999...999]
        max_num = 10 ** self.config['max_digits']
        left = random.randrange(0, max_num)
        right = random.randrange(0, max_num)
        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}+{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{left + right}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        # Base cases
        if left < 10 and right < 10:
            return []

        l_last, r_last = left % 10, right % 10
        thoughts = [T(Add, (l_last, r_last))]

        l_rest, r_rest = left // 10, right // 10
        if l_last + r_last >= 10:
            thoughts.append(T(Add, (l_rest, 1)))
            l_rest += 1

        if l_rest > 0 and r_rest > 0:
            thoughts.append(T(Add, (l_rest, r_rest)))

        return thoughts


class Sub(Problem):
    """Subtraction of two positive integers
    E.g.,
        # Memorize trivial case
        <GO>14-8=6<STOP>

        # Without borrow
        <GO>445-283=
            <GO>15-3=12<STOP>
            <GO>44-28=16<STOP>
            162<STOP>

        # With borrow
        <GO>441-383=
            <GO>11-3=8<STOP>
            <GO>44-1=43<STOP>
            <GO>43-38=5<STOP>
            162<STOP>
    """

    name = 'Sub'
    dependencies = {}
    symbols = ['-']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        left = random.randrange(0, max_num)
        right = random.randrange(0, max_num)

        if left < right:
            return right, left
        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}-{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{left - right}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        # Base cases
        if left <= 19 and right <= 9:
            return []

        l_last = left % 10 + 10
        r_last = right % 10
        thoughts = [T(Sub, (l_last, r_last))]
        l_rest, r_rest = left // 10, right // 10
        if l_last - r_last < 10:
            thoughts.append(T(Sub, (l_rest, 1)))
            l_rest -= 1
        if r_rest > 0:
            thoughts.append(T(Sub, (l_rest, r_rest)))

        return thoughts

    def enum_args(self):
        max_num = 10 ** self.config['max_digits']
        args = []
        for left in range(max_num):
            for right in range(left + 1):
                args.append((left, right))
        return args


class Mul(Problem):
    """Multiplication
    E.g.,
        # Memorize trivial cases
        <GO>3*4=12<STOP>
        <GO>37284*1=37284<STOP>
        <GO>0*748=0<STOP>

        # Solve recursively
        <GO>123*45=
            <GO>123*5=615<STOP>
            <GO>123*4=492<STOP>
            <GO>4920+615=5535<STOP>
            5535<STOP>
    """
    name = 'Mul'
    dependencies = {
        Add: lambda config: {'max_digits': config['max_digits'] * 2}
    }
    symbols = ['*']

    def generate(self):
        # Generate [0, 999...999]
        max_digits = self.config['max_digits']
        max_num = 10 ** max_digits
        left = self.log_randrange(0, max_num)
        right = self.log_randrange(0, max_num)
        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}*{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{left * right}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        # Base cases
        if left <= 1 or right <= 1:
            return []
        if left <= 9 and right <= 9:
            return []

        thoughts = []
        if right < 10:
            thoughts.append(T(Mul, (left % 10, right)))
            thoughts.append(T(Mul, (left // 10, right)))

            a1 = (left % 10) * right
            a2 = (left // 10) * right
            thoughts.append(T(Add, (a2 * 10, a1), 'tail'))
        else:
            a1 = left * (right % 10)
            thoughts.append(T(Mul, (left, right % 10)))

            a2 = left * (right // 10)
            thoughts.append(T(Mul, (left, right // 10)))

            thoughts.append(T(Add, (a2 * 10, a1), 'tail'))
        return thoughts


class Div(Problem):
    """Integer division
    E.g.,
        # Memorize trivial case
        <GO>17÷3=5<R>2<STOP>
        <GO>245÷34=
            <GO>245<VS>34=<GT><STOP>
            <GO>245<VS>340=<LT><STOP>
            <GO>245-34=211<STOP>
            <GO>211÷34=6<R>7<STOP>
            <GO>6+1=7<STOP>
            7<R>7<STOP>
        <GO>2285÷34=
            <GO>2285<VS>34=<GT><STOP>
            <GO>2285<VS>340=<GT><STOP>
            <GO>228÷34=6<R>24<STOP>
            <GO>245÷34=7<R>7<STOP>
            67<R>7<STOP>
    """
    name = 'Div'
    dependencies = {
        Compare: lambda config: {'max_digits': config['max_digits'] + 1},
        Sub: lambda config: config,
    }
    symbols = ['÷', '<R>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        divisor = self.log_randrange(1, max_num)
        quotient = self.log_randrange(0, (max_num - 1) / divisor)
        max_remainder = min(divisor, max_num - quotient * divisor)
        remainder = self.log_randrange(0, max_remainder)
        dividend = divisor * quotient + remainder
        return dividend, divisor

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}÷{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{left // right}<R>{left % right}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args
        thoughts = [T(Compare, (left, right))]

        # Base cases
        if left <= right:
            return thoughts

        thoughts.append(T(Compare, (left, right * 10)))
        if left <= right * 10:
            diff = left - right
            thoughts.append(T(Sub, (left, right)))
            thoughts.append(T(Div, (diff, right)))
        else:
            thoughts.append(T(Div, (left // 10, right)))
            left_remainder = (left // 10) % right * 10 + left % 10
            thoughts.append(T(Div, (left_remainder, right)))
        return thoughts

    def enum_args(self):
        max_num = 10 ** self.config['max_digits']
        args = []
        for left in range(0, max_num):
            for right in range(1, max_num):
                args.append((left, right))
        return args


class Gcd(Problem):
    """Greatest Common Divisor
    E.g.,
        # Base case
        <GO>72<GCD>36=
            <GO>72<VS>36=<GT><STOP>
            <GO>72÷36=2<R>0<STOP>
            36<STOP>
        <GO>36<GCD>72=
            <GO>36<VS>72=<LT><STOP>
            <GO>72÷36=2<R>0<STOP>
            36<STOP>
        <GO>36<GCD>36=
            <GO>36<VS>36=<EQ><STOP>
            36<STOP>

        # Solve recursively
        <GO>78696<GCD>19332=
            <GO>78696<VS>19332=<GT><STOP>
            <GO>78696÷19332=4<R>1368<STOP>
            <GO>19332<GCD>1368=36<STOP>
            36<STOP>
    """
    name = 'Gcd'
    dependencies = {
        Compare: lambda config: {'max_digits': config['max_digits'] + 1},
        Div: lambda config: config,
    }
    symbols = ['<GCD>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        cd = self.log_randrange(1, max_num)

        if random.random() < 0.85:
            left = cd * self.log_randrange(1, max_num // cd + 1)
            right = cd * self.log_randrange(1, max_num // cd + 1)
        else:
            left = right = cd

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}<GCD>{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{math.gcd(left, right)}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        thoughts = [T(Compare, (left, right))]

        if left == right:
            return thoughts
        elif left < right:
            left, right = right, left

        thoughts.append(T(Div, (left, right)))

        rest = left % right
        # Except base case
        if rest != 0:
            thoughts.append(T(Gcd, (right, rest)))

        return thoughts


class Lcm(Problem):
    """Least Common Multiple
    E.g.,
        <GO>36<LCM>72=
            <GO>36<GCD>72=36<STOP>
            <GO>36÷36=1<R>0<STOP>
            <GO>1*72=72<STOP>
            72<STOP>
        <GO>78696<LCM>19332=
            <GO>78696<GCD>19332=36<STOP>
            <GO>78696÷36=2186<R>0<STOP>
            <GO>2186*19332=42259752<STOP>
            42259752<STOP>
    """
    name = 'Lcm'
    dependencies = {
        Gcd: lambda config: config,
        Div: lambda config: config,
        Mul: lambda config: config,
    }
    symbols = ['<LCM>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        cd = self.log_randrange(1, max_num)
        left = cd * random.randrange(1, max_num // cd + 1)
        right = cd * random.randrange(1, max_num // cd + 1)

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}<LCM>{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{math.lcm(left, right)}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        thoughts = [
            T(Gcd, (left, right)),
            T(Div, (left, math.gcd(left, right))),
            T(Mul, (left // math.gcd(left, right), right))
        ]

        return thoughts


class Reduce(Problem):
    """Make a fraction irreducible
    E.g.,
        <GO><REDUCE>10/5=
            <GO>10<GCD>5=5<STOP>
            <GO>10÷5=2<R>0<STOP>
            <GO>5÷5=1<R>0<STOP>
            <GO>1<VS>1=<EQ>
            2<STOP>
        <GO><REDUCE>10/4=
            <GO>10<GCD>4=2<STOP>
            <GO>10÷2=5<R>0<STOP>
            <GO>4÷2=2<R>0<STOP>
            <GO>2<VS>1=<GT>
            5/2<STOP>
        <GO><REDUCE>10/3=
            <GO>10<GCD>3=1<STOP>
            <GO>10÷1=10<R>0<STOP>
            <GO>3÷1=3<R>0<STOP>
            <GO>3<VS>1=<GT>
            10/3<STOP>
        # trivial case
        <GO><REDUCE>10/1=
            10<STOP>
        <GO><REDUCE>0/23=
            0<STOP>
    """
    name = 'Reduce'
    dependencies = {
        Gcd: lambda config: config,
        Div: lambda config: config,
    }
    symbols = ['<REDUCE>', '/']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        rand = random.random()
        if rand < 0.65:
            cd = self.log_randrange(1, max_num)
            numer = cd * random.randrange(1, max_num // cd + 1)
            denom = cd * random.randrange(1, max_num // cd + 1)
        elif rand < 0.7:
            numer = self.log_randrange(0, max_num)
            denom = 1
        else:
            numer = self.log_randrange(0, max_num)
            denom = self.log_randrange(1, max_num)
        return numer, denom

    @staticmethod
    def question(args):
        numer, denom = args
        return f'<GO><REDUCE>{numer}/{denom}='

    @staticmethod
    def answer(args):
        return frac_to_str(args, reduce=True)

    @staticmethod
    def thought(args) -> list[T]:
        numer, denom = args

        # Trivial case
        if denom == 1:
            return []
        if numer == 0:
            return []

        return [T(Gcd, (numer, denom)),
                T(Div, (numer, math.gcd(numer, denom))),
                T(Div, (denom, math.gcd(numer, denom)))]

    @staticmethod
    def get_answer(args):
        numer, denom = args

        if numer == 0:
            return 0, 1

        gcd = math.gcd(numer, denom)
        numer = numer // gcd
        denom = denom // gcd

        if denom < 0:
            numer = -numer
            denom = -denom

        return numer, denom


class Sub_pos_int(Problem):
    """Subtraction of two positive integers including smaller - larger
    E.g.
        <GO>441<SUB_POS_INT>383=
            <GO>441<VS>383=<GT><STOP>
            <GO>441-383=62<STOP>
            62<STOP>
        <GO>383<SUB_POS_INT>441=
            <GO>383<VS>441=<LT><STOP>
            <GO>441-383=62<STOP>
            -62<STOP>
        <GO>383<SUB_POS_INT>383=
            <GO>383<VS>383=<EQ><STOP>
            0<STOP>
    """

    name = 'Sub_pos_int'
    dependencies = {
        Sub: lambda config: config,
        Compare: lambda config: {'max_digits': config['max_digits'] + 1},
    }

    symbols = ['<SUB_POS_INT>', '-']

    def generate(self):
        max_num = 10 ** self.config['max_digits']
        left = self.log_randrange(0, max_num)
        right = self.log_randrange(0, max_num)
        return left, right

    @staticmethod
    def question(args):
        left, right = args
        return f'<GO>{left}<SUB_POS_INT>{right}='

    @staticmethod
    def answer(args):
        left, right = args
        return f'{left - right}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        thoughts = [T(Compare, (left, right))]

        if left != right:
            thoughts.append(T(Sub, (max(left, right), min(left, right))))

        return thoughts


class Add_frac(Problem):
    """Fraction Addition
    E.g.,
        <GO>23/10<ADD_FRAC>6/14=
            <GO>23*14=322<STOP>
            <GO>6*10=60<STOP>
            <GO>322+60=382<STOP>
            <GO>10*14=140<STOP>
            <GO><REDUCE>382/140=191/70<STOP>
            191/70<STOP>

        <GO>23/10<ADD_FRAC>6=
            <GO>23/10<ADD_FRAC>6/1=83/10<STOP>
            83/10<STOP>

        <GO>6<ADD_FRAC>23/10=
            <GO>6/1<ADD_FRAC>23/10=83/10<STOP>
            83/10<STOP>

        <GO>6<ADD_FRAC>23=
            <GO>6+23=29
            29<STOP>
    """
    name = 'Add_frac'
    dependencies = {
        Add: lambda config: {'max_digits': config['max_digits'] * 2},
        Mul: lambda config: config,
        Reduce: lambda config: {'max_digits': config['max_digits'] * 2},
    }
    symbols = ['<ADD_FRAC>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        rand = random.random()
        if rand < 0.7:
            numer_left = self.log_randrange(1, max_num)
            denom_left = self.log_randrange(1, max_num)
            left = (numer_left, denom_left)
        else:
            left = self.log_randrange(1, max_num)

        rand = random.random()
        if rand < 0.7:
            numer_right = self.log_randrange(1, max_num)
            denom_right = self.log_randrange(1, max_num)
            right = (numer_right, denom_right)
        else:
            right = self.log_randrange(1, max_num)

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            return f'<GO>{left}<ADD_FRAC>{right}='
        elif isinstance(left, int):
            numer_right, denom_right = right
            return f'<GO>{left}<ADD_FRAC>{numer_right}/{denom_right}='
        elif isinstance(right, int):
            numer_left, denom_left = left
            return f'<GO>{numer_left}/{denom_left}<ADD_FRAC>{right}='
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            return f'<GO>{numer_left}/{denom_left}<ADD_FRAC>{numer_right}/{denom_right}='

    @staticmethod
    def answer(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            return Add.answer(args)
        elif isinstance(left, int):
            numer_right, denom = right
            numer = left * denom + numer_right
        elif isinstance(right, int):
            numer_left, denom = left
            numer = right * denom + numer_left
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right

            numer = numer_left * denom_right + numer_right * denom_left
            denom = denom_left * denom_right

        return frac_to_str((numer, denom), reduce=True)

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        if isinstance(left, int) and isinstance(right, int):
            return [T(Add, (left, right))]

        elif isinstance(left, int):
            return [T(Add_frac, ((left, 1), right))]

        elif isinstance(right, int):
            return [T(Add_frac, (left, (right, 1)))]

        else:
            numer_left, denom_left = left
            numer_right, denom_right = right

            numer = numer_left * denom_right + numer_right * denom_left
            denom = denom_left * denom_right

            return [T(Mul, (numer_left, denom_right)),
                    T(Mul, (numer_right, denom_left)),
                    T(Add, (numer_left * denom_right, numer_right * denom_left)),
                    T(Mul, (denom_left, denom_right)),
                    T(Reduce, (numer, denom))]


class Sub_frac(Problem):
    """Subtraction of two positive fractions
    E.g.,
        <GO>23/10<SUB_FRAC>6/14=
            <GO>23*14=322<STOP>
            <GO>6*10=60<STOP>
            <GO>322<SUB_POS_INT>60=262<STOP>
            <GO>10*14=140<STOP>
            <GO><REDUCE>262/140=131/70<STOP>
            131/70<STOP>

        <GO>6/14<SUB_FRAC>23/10=
            <GO>6*10=60<STOP>
            <GO>23*14=322<STOP>
            <GO>60<SUB_POS_INT>322=-262<STOP>
            <GO>14*10=140<STOP>
            <GO><REDUCE>262/140=131/70<STOP>
            -131/70<STOP>

        <GO>6/14<SUB_FRAC>3/7=
            <GO>6*7=42<STOP>
            <GO>3*14=42<STOP>
            <GO>42<SUB_POS_INT>42=0<STOP>
            0<STOP>

        <GO>23/10<SUB_FRAC>6=
            <GO>23/10<SUB_FRAC>6/1=-37/10<STOP>
            -37/10<STOP><STOP>

        <GO>6<SUB_FRAC>23/10=
            <GO>6/1<SUB_FRAC>23/10<STOP>
            37/10<STOP>

        <GO>6<SUB_FRAC>23=
            <GO>6<SUB_POS_INT>23=-17<STOP>
            -17<STOP>
        <GO>23<SUB_FRAC>6=
            <GO>23<SUB_POS_INT>6=17<STOP>
            17<STOP>
    """
    name = 'Sub_frac'
    dependencies = {
        Mul: lambda config: config,
        Sub_pos_int: lambda config: {'max_digits': config['max_digits'] * 2},
        Reduce: lambda config: {'max_digits': config['max_digits'] * 2},
    }
    symbols = ['<SUB_FRAC>', '-']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        rand = random.random()
        if rand < 0.7:
            numer_left = self.log_randrange(1, max_num)
            denom_left = self.log_randrange(1, max_num)
            left = (numer_left, denom_left)
        else:
            left = self.log_randrange(1, max_num)

        rand = random.random()
        if rand < 0.7:
            numer_right = self.log_randrange(1, max_num)
            denom_right = self.log_randrange(1, max_num)
            right = (numer_right, denom_right)
        else:
            right = self.log_randrange(1, max_num)

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            return f'<GO>{left}<SUB_FRAC>{right}='
        elif isinstance(left, int):
            numer_right, denom_right = right
            return f'<GO>{left}<SUB_FRAC>{numer_right}/{denom_right}='
        elif isinstance(right, int):
            numer_left, denom_left = left
            return f'<GO>{numer_left}/{denom_left}<SUB_FRAC>{right}='
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            return f'<GO>{numer_left}/{denom_left}<SUB_FRAC>{numer_right}/{denom_right}='

    @staticmethod
    def answer(args):
        return f'{frac_to_str(Sub_frac.get_answer(args))}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        if isinstance(left, int) and isinstance(right, int):
            return [T(Sub_pos_int, (left, right))]
        elif isinstance(left, int):
            return [T(Sub_frac, ((left, 1), right))]
        elif isinstance(right, int):
            return [T(Sub_frac, (left, (right, 1)))]
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right

            numer = numer_left * denom_right - numer_right * denom_left
            denom = denom_left * denom_right

            return [T(Mul, (numer_left, denom_right)),
                    T(Mul, (numer_right, denom_left)),
                    T(Sub_pos_int, (numer_left * denom_right, numer_right * denom_left)),
                    T(Mul, (denom_left, denom_right)),
                    T(Reduce, (abs(numer), denom))]

    @staticmethod
    def get_answer(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            numer =  left - right
            denom = 1
        elif isinstance(left, int):
            numer_right, denom = right
            numer = left * denom - numer_right
        elif isinstance(right, int):
            numer_left, denom = left
            numer = numer_left - right * denom
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right

            numer = numer_left * denom_right - numer_right * denom_left
            denom = denom_left * denom_right

        return Reduce.get_answer((numer, denom))


class Mul_frac(Problem):
    """Fraction Multiplication
    E.g.,
        <GO>105/10<MUL_FRAC>6/14=
            <GO>105*6=630<STOP>
            <GO>10*14=140<STOP>
            <GO><REDUCE>630/140=9/2<STOP>
            9/2<STOP>

        <GO>105/10<MUL_FRAC>6=
            <GO>105/10<MUL_FRAC>6/1=63<STOP>
            63<STOP>

        <GO>105<MUL_FRAC>6/14=
            <GO>105/1<MUL_FRAC>6/14=45<STOP>
            45<STOP>

        <GO>105<MUL_FRAC>6=
            <GO>105*6=630<STOP>
            630<STOP>
    """
    name = 'Mul_frac'
    dependencies = {
        Mul: lambda config: config,
        Reduce: lambda config: {'max_digits': config['max_digits'] * 2},
    }
    symbols = ['<MUL_FRAC>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        rand = random.random()
        if rand < 0.7:
            numer_left = self.log_randrange(1, max_num)
            denom_left = self.log_randrange(1, max_num)
            left = (numer_left, denom_left)
        else:
            left = self.log_randrange(0, max_num)

        rand = random.random()
        if rand < 0.7:
            numer_right = self.log_randrange(1, max_num)
            denom_right = self.log_randrange(1, max_num)
            right = (numer_right, denom_right)
        else:
            right = self.log_randrange(0, max_num)

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            return f'<GO>{left}<MUL_FRAC>{right}='
        elif isinstance(left, int):
            numer_right, denom_right = right
            return f'<GO>{left}<MUL_FRAC>{numer_right}/{denom_right}='
        elif isinstance(right, int):
            numer_left, denom_left = left
            return f'<GO>{numer_left}/{denom_left}<MUL_FRAC>{right}='
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            return f'<GO>{numer_left}/{denom_left}<MUL_FRAC>{numer_right}/{denom_right}='

    @staticmethod
    def answer(args):
        return f'{frac_to_str(Mul_frac.get_answer(args))}<STOP>'

    @staticmethod
    def thought(args, recurse=False) -> list[tuple[str, list[tuple], str]]:
        left, right = args

        if left == 0 or right == 0:
            return []

        if isinstance(left, int) and isinstance(right, int):
            return [T(Mul, (left, right))]
        elif isinstance(left, int):
            return [T(Mul_frac, ((left, 1), right))]
        elif isinstance(right, int):
            return [T(Mul_frac, (left, (right, 1)))]
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            numer = numer_left * numer_right
            denom = denom_left * denom_right

            return [T(Mul, (numer_left, numer_right)),
                    T(Mul, (denom_left, denom_right)),
                    T(Reduce, (numer, denom))]

    @staticmethod
    def get_answer(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            numer = left * right
            denom = 1
        elif isinstance(left, int):
            numer_right, denom = right
            numer = left * numer_right
        elif isinstance(right, int):
            numer_left, denom = left
            numer = right * numer_left
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            numer = numer_left * numer_right
            denom = denom_left * denom_right

        if numer == 0:
            return 0

        return Reduce.get_answer((numer, denom))


class Div_frac(Problem):
    """Division between two positive fractions
    E.g.,
        <GO>23/25<DIV_FRAC>6/45=
            <GO>23/25<MUL_FRAC>45/6=345/52<STOP>
            345/52<STOP>

        <GO>23/25<DIV_FRAC>2=
            <GO>23/25<DIV_FRAC>2/1=23/50<STOP>
            23/50<STOP>

        <GO>2<DIV_FRAC>23/25=
            <GO>2/1<DIV_FRAC>23/25<STOP>
            50/23<STOP>

        <GO>23<DIV_FRAC>6=
            <GO><REDUCE>23/6=23/6<STOP>
            23/6<STOP>
    """
    name = 'Div_frac'
    dependencies = {
        Reduce: lambda config: config,
        Mul_frac: lambda config: config,
    }
    symbols = ['<DIV_FRAC>']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        rand = random.random()
        if rand < 0.7:
            numer_left = self.log_randrange(1, max_num)
            denom_left = self.log_randrange(1, max_num)
            left = (numer_left, denom_left)
        else:
            left = self.log_randrange(1, max_num)

        rand = random.random()
        if rand < 0.7:
            numer_right = self.log_randrange(1, max_num)
            denom_right = self.log_randrange(1, max_num)
            right = (numer_right, denom_right)
        else:
            right = self.log_randrange(1, max_num)

        return left, right

    @staticmethod
    def question(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            return f'<GO>{left}<DIV_FRAC>{right}='
        elif isinstance(left, int):
            numer_right, denom_right = right
            return f'<GO>{left}<DIV_FRAC>{numer_right}/{denom_right}='
        elif isinstance(right, int):
            numer_left, denom_left = left
            return f'<GO>{numer_left}/{denom_left}<DIV_FRAC>{right}='
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            return f'<GO>{numer_left}/{denom_left}<DIV_FRAC>{numer_right}/{denom_right}='

    @staticmethod
    def answer(args):
        return f'{frac_to_str(Div_frac.get_answer(args))}<STOP>'

    @staticmethod
    def thought(args, recurse=False) -> list[tuple[str, list[tuple], str]]:
        left, right = args

        if left == 0:
            return []

        if isinstance(left, int) and isinstance(right, int):
            return [T(Reduce, (left, right))]

        elif isinstance(left, int):
            return [T(Div_frac, ((left, 1), right))]

        elif isinstance(right, int):
            return [T(Div_frac, (left, (right, 1)))]
        else:
            numer_right, denom_right = right
            return [T(Mul_frac, (left, (denom_right, numer_right)))]

    @staticmethod
    def get_answer(args):
        left, right = args
        if isinstance(left, int) and isinstance(right, int):
            numer = left
            denom = right
        elif isinstance(left, int):
            numer_right, denom_right = right
            numer = left * denom_right
            denom = numer_right
        elif isinstance(right, int):
            numer_left, denom_left = left
            numer = numer_left
            denom = denom_left * right
        else:
            numer_left, denom_left = left
            numer_right, denom_right = right
            numer = numer_left * denom_right
            denom = denom_left * numer_right

        if numer == 0:
            return 0, 1

        return Reduce.get_answer((numer, denom))


class Operations(Problem):
    """Extension of Four Fundamental Arithmetic Operations to Negative Number
    E.g.,
        # Each operands can be an integer or an fraction
        <GO>23<ADD>6=
            <GO>23<ADD_FRAC>6=29<STOP>
            29<STOP>
        <GO>-23<ADD>6=
            <GO>23<SUB_FRAC>6=17<STOP>
            -17<STOP>
        <GO>23<ADD>-6=
            <GO>23<SUB_FRAC>6=17<STOP>
            17<STOP>
        <GO>-23<ADD>-6=
            <GO>23<ADD_FRAC>6=29<STOP>
            -29<STOP>

        <GO>23<SUB>6=
            <GO>23<SUB_FRAC>6=17<STOP>
            17<STOP>
        <GO>-23<SUB>-6=
            <GO>23<SUB_FRAC>6=17<STOP>
            -17<STOP>
        <GO>23<SUB>-6=
            <GO>23<ADD_FRAC>6=29<STOP>
            29<STOP>
        <GO>-23<SUB>6=
            <GO>23<ADD_FRAC>6=29<STOP>
            -29<STOP>

        <GO>23<MUL>6=
            <GO>23<MUL_FRAC>6=138<STOP>
            138<STOP>
        <GO>-23<MUL>6=
            <GO>23<MUL_FRAC>6=138<STOP>
            -138<STOP>
        <GO>23<MUL>-6=
            <GO>23<MUL_FRAC>6=138<STOP>
            -138<STOP>
        <GO>-23<MUL>-6=
            <GO>23<MUL_FRAC>6=138<STOP>
            138<STOP>

        <GO>23<DIV>6=
            <GO>23<DIV_FRAC>6=23/6<STOP>
            23/6<STOP>
        <GO>-23<DIV>6=
            <GO>23<DIV_FRAC>6=23/6<STOP>
            -23/6<STOP>
        <GO>23<DIV>-6=
            <GO>23<DIV_FRAC>6=23/6<STOP>
            -23/6<STOP>
        <GO>-23<DIV>-6=
            <GO>23<DIV_FRAC>6=23/6<STOP>
            23/6<STOP>
    """
    name = 'Operations'
    dependencies = {
        Add_frac: lambda config: config,
        Sub_frac: lambda config: config,
        Mul_frac: lambda config: config,
        Div_frac: lambda config: config,
    }
    symbols = ['<ADD>', '<SUB>', '<MUL>', '<DIV>', '-', '/']

    def generate(self):
        max_num = 10 ** self.config['max_digits']

        prob = random.sample(('Add', 'Sub', 'Mul', 'Div'), 1)[0]
        # prob = random.sample(('Add', 'Sub'), 1)[0]

        rand = random.random()
        if rand < 0.7:
            numer_left = self.log_randrange(1, max_num)
            denom_left = self.log_randrange(1, max_num)
            left = (numer_left, denom_left)
        else:
            left = self.log_randrange(0, max_num)

        if random.random() < 0.5:
            left = negate_frac(left)

        rand = random.random()
        if rand < 0.7:
            numer_right = self.log_randrange(1, max_num)
            denom_right = self.log_randrange(1, max_num)
            right = (numer_right, denom_right)
        else:
            if prob == 'Div':
                right = self.log_randrange(1, max_num)
            else:
                right = self.log_randrange(0, max_num)

        if random.random() < 0.5:
            right = negate_frac(right)

        return prob, left, right

    @staticmethod
    def question(args):
        prob, left, right = args

        q_list = ['<GO>']

        if isinstance(left, int):
            q_list.append(f'{left}')
        else:
            numer_left, denom_left = left
            q_list.append(f'{numer_left}/{denom_left}')

        q_list.append(f'<{prob.upper()}>')

        if isinstance(right, int):
            q_list.append(f'{right}')
        else:
            numer_right, denom_right = right
            q_list.append(f'{numer_right}/{denom_right}')

        q_list.append('=')

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        prob, left, right = args

        if prob == 'Add':
            if isinstance(left, int) and isinstance(right, int):
                return f'{left + right}<STOP>'
            elif isinstance(left, int):
                numer_right, denom = right
                numer = left * denom + numer_right
            elif isinstance(right, int):
                numer_left, denom = left
                numer = numer_left + right * denom
            else:
                numer_left, denom_left = left
                numer_right, denom_right = right
                numer = numer_left * denom_right + numer_right * denom_left
                denom = denom_left * denom_right

        elif prob == 'Sub':
            if isinstance(left, int) and isinstance(right, int):
                return f'{left - right}<STOP>'
            elif isinstance(left, int):
                numer_right, denom = right
                numer = left * denom - numer_right
            elif isinstance(right, int):
                numer_left, denom = left
                numer = numer_left - right * denom
            else:
                numer_left, denom_left = left
                numer_right, denom_right = right
                numer = numer_left * denom_right - numer_right * denom_left
                denom = denom_left * denom_right

        elif prob == 'Mul':
            if isinstance(left, int) and isinstance(right, int):
                return f'{left * right}<STOP>'
            elif isinstance(left, int):
                numer_right, denom = right
                numer = left * numer_right
            elif isinstance(right, int):
                numer_left, denom = left
                numer = numer_left * right
            else:
                numer_left, denom_left = left
                numer_right, denom_right = right
                numer = numer_left * numer_right
                denom = denom_left * denom_right

        else:
            if isinstance(left, int) and isinstance(right, int):
                numer = left
                denom = right
            elif isinstance(left, int):
                numer_right, denom_right = right
                numer = left * denom_right
                denom = numer_right
            elif isinstance(right, int):
                numer_left, denom_left = left
                numer = numer_left
                denom = denom_left * right
            else:
                numer_left, denom_left = left
                numer_right, denom_right = right
                numer = numer_left * denom_right
                denom = denom_left * numer_right

        return f'{frac_to_str((numer, denom), reduce=True)}<STOP>'


    @staticmethod
    def thought(args) -> list[T]:
        prob, left, right = args

        if prob == 'Add':
            if is_frac_neg(left) == is_frac_neg(right):
                thoughts = [T(Add_frac, (abs_frac(left), abs_frac(right)))]
            else:
                thoughts = [T(Sub_frac, (abs_frac(left), abs_frac(right)))]

        elif prob == 'Sub':
            if is_frac_neg(left) == is_frac_neg(right):
                thoughts = [T(Sub_frac, (abs_frac(left), abs_frac(right)))]
            else:
                thoughts = [T(Add_frac, (abs_frac(left), abs_frac(right)))]

        elif prob == 'Mul':
            thoughts = [T(Mul_frac, (abs_frac(left), abs_frac(right)))]

        else:
            thoughts = [T(Div_frac, (abs_frac(left), abs_frac(right)))]

        return thoughts


def is_frac_neg(arg):
    if isinstance(arg, int):
        return arg < 0
    else:
        return arg[0] < 0

def negate_frac(arg):
    if isinstance(arg, int):
        return -arg
    else:
        return -arg[0], arg[1]

def abs_frac(arg):
    if isinstance(arg, int):
        return abs(arg)
    else:
        return abs(arg[0]), arg[1]

def frac_to_str(frac, reduce=False):
    if isinstance(frac, int):
        return f'{frac}'

    if reduce:
        numer, denom = Reduce.get_answer(frac)
    else:
        numer, denom = frac

    if denom == 1:
        return f'{numer}'
    else:
        return f'{numer}/{denom}'