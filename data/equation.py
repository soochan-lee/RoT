from .arithmetic import *


class Linear_1d(Problem):
    """Solve 1D Linear Equation with Integer Coefficient
    E.g.,
        <GO><LINEAR_1D>3x=-7<SOLVE>
            <GO>-7<DIV>3=-7/3<STOP>
            x=-7/3<STOP>
        <GO><LINEAR_1D>5y=15<SOLVE>
            <GO>15<DIV>5=3<STOP>
            y=3<STOP>
    """
    name = 'Linear_1d'
    dependencies = {
        Operations: lambda config: config,
    }
    symbols = ['<LINEAR_1D>', '<SOLVE>', 'x', 'y', '/', '-', '+']

    def generate(self):
        max_coef = 10 ** self.config['max_digits']
        linear = self.log_randrange(1, max_coef)
        if random.random() < 0.5:
            linear = -linear
        constant = self.log_randrange(0, max_coef)
        if random.random() < 0.5:
            constant = -constant

        variable = random.sample(('x', 'y'), 1)[0]

        return variable, linear, constant

    @staticmethod
    def question(args):
        variable, linear, constant = args

        q_list = ['<GO><LINEAR_1D>']

        if linear < 0:
            q_list.append('-')
        if abs(linear) == 1:
            pass
        else:
            q_list.append(f'{abs(linear)}')

        q_list.extend([variable,
                       '=',
                       frac_to_str(constant),
                       '<SOLVE>'])

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        variable, linear, constant = args
        numer, denom = Div_frac.get_answer((constant, linear))

        return f'{variable}={frac_to_str((numer, denom))}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        if len(args) == 2:
            print(args)
        _, linear, constant = args
        return [T(Operations, ('Div', constant, linear))]


class Mul_both(Problem):
    """Multiply a constant to both sides of equation
    E.g.,
        <GO><MUL_BOTH>x+2y=5<SEP>8<SOLVE>
            <GO>1<MUL>8=8<STOP>
            <GO>2<MUL>8=16<STOP>
            <GO>5<MUL>8=40<STOP>
            8x+16=40<STOP>
        <GO><MUL_BOTH>-2x-y=-8<SEP>4<SOLVE>
            <GO>-2<MUL>4=-8<STOP>
            <GO>-1<MUL>4=-4<STOP>
            <GO>-8<MUL>4=-32<STOP>
            -8x-4y=-32<STOP>
        <GO><MUL_BOTH>2x=-8<SEP>4<SOLVE>
            <GO>2<MUL>4=8<STOP>
            <GO>-8<MUL>4=-32<STOP>
            8x=-32<STOP>
    """
    name = 'Mul_both'
    dependencies = {
        Operations: lambda config: config,
    }
    symbols = ['<MUL_BOTH>', '<SOLVE>', '<SEP>', 'x', 'y', '+', '-']

    def generate(self):
        x_coef, y_coef, const = self.sample_linear_2d(self.config['max_digits'])

        max_coef = 10 ** self.config['max_digits']
        multiplier = self.log_randrange(1, max_coef)

        return x_coef, y_coef, const, multiplier

    @staticmethod
    def question(args):
        # <GO><MUL_BOTH>x+2y=5<SEP>8<SOLVE>
        x_coef, y_coef, const, multiplier = args

        q_list = ['<GO><MUL_BOTH>',
                  make_linear_2d(x_coef, y_coef, const),
                  f'<SEP>{multiplier}<SOLVE>']

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        # 8x+16=40<STOP>
        x_coef, y_coef, const, multiplier = args

        ans = make_linear_2d(x_coef * multiplier,
                                     y_coef * multiplier,
                                     const * multiplier)
        return f'{ans}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        x_coef, y_coef, const, multiplier = args

        thoughts = []
        if x_coef != 0:
            thoughts.append(T(Operations, ('Mul', x_coef, multiplier)))
        if y_coef != 0:
            thoughts.append(T(Operations, ('Mul', y_coef, multiplier)))
        thoughts.append(T(Operations, ('Mul', const, multiplier)))

        return thoughts


class Elim(Problem):
    """
    E.g.,
        # Base case
        <GO><ELIM>x+y=5<SEP>2x-y=8<SOLVE>
            <GO>1<ADD>2=3<STOP>
            <GO>1<ADD>-1=0<STOP>
            <GO>5<ADD>8=13<STOP>
            3x=13<STOP>
        <GO><ELIM>x+y=5<SEP>2x+y=8<SOLVE>
            <GO>1<SUB>2=-1<STOP>
            <GO>1<SUB>1=0<STOP>
            <GO>5<SUB>8=-3<STOP>
            x=3<STOP>
        <GO><ELIM>-2x-y=5<SEP>2x+y=8<SOLVE>
            <GO>-2<ADD>2=0<STOP>
            <GO>-1<ADD>1=0<STOP>
            <GO>5<ADD>8=13<STOP>
            0=3<STOP>
        <GO><ELIM>-2x-y=-8<SEP>2x+y=8<SOLVE>
            <GO>-2<ADD>2=0<STOP>
            <GO>-1<ADD>1=0<STOP>
            <GO>-8<ADD>8=0<STOP>
            0=0<STOP>
        # Recursion
        <GO><ELIM>-4x+3y=-8<SEP>7x-4y=14<SOLVE>
            <GO><MUL_BOTH>-4x+3y=-8<SEP>7<SOLVE>-28x+21y=-56<STOP>
            <GO><MUL_BOTH>7x-4y=14<SEP>4<SOLVE>28x-16y=56<STOP>
            <GO><ELIM>-28x+21y=-56<SEP>28x-16y=56<SOLVE>5y=0<STOP>
    """
    name = 'Elim'
    dependencies = {
        Operations: lambda config: config,
        Mul_both: lambda config: config,
    }
    symbols = ['<ELIM>', '<SOLVE>', '<SEP>', 'x', 'y','+' ,'-']

    def generate(self):
        x_coef_l, y_coef_l, const_l = self.sample_linear_2d(self.config['max_digits'])
        x_coef_r, y_coef_r, const_r = self.sample_linear_2d(self.config['max_digits'])

        # There should be at least one variable to be eliminated
        if x_coef_l * x_coef_r == 0 and y_coef_l * y_coef_r == 0:
            return self.generate()

        return (x_coef_l, y_coef_l, const_l), (x_coef_r, y_coef_r, const_r)

    @staticmethod
    def question(args):
        # <GO><ELIM>-4x+3y=-8<SEP>7x-4y=14<SOLVE>
        left, right = args

        q_list = ['<GO><ELIM>',
                  make_linear_2d(left[0], left[1], left[2]),
                  '<SEP>',
                  make_linear_2d(right[0], right[1], right[2]),
                  '<SOLVE>']

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        # 5y=0<STOP>
        left, right = args
        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        if x_coef_l == x_coef_r:
            ans = make_linear_2d(0, y_coef_l - y_coef_r, const_l - const_r)
        elif x_coef_l == -x_coef_r:
            ans = make_linear_2d(0, y_coef_l + y_coef_r, const_l + const_r)
        elif y_coef_l == y_coef_r:
            ans = make_linear_2d(x_coef_l - x_coef_r, 0, const_l - const_r)
        elif y_coef_l == -y_coef_r:
            ans = make_linear_2d(x_coef_l + x_coef_r, 0, const_l + const_r)
        else:
            ans = make_linear_2d(0,
                                         y_coef_l * abs(x_coef_r) - y_coef_r * abs(x_coef_l),
                                         const_l * abs(x_coef_r) - const_r * abs(x_coef_l))
        return f'{ans}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args
        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        if x_coef_l == x_coef_r or y_coef_l == y_coef_r:
            thoughts = [T(Operations, ('Sub', x_coef_l, x_coef_r)),
                        T(Operations, ('Sub', y_coef_l, y_coef_r)),
                        T(Operations, ('Sub', const_l, const_r))]

        elif x_coef_l == -x_coef_r or y_coef_l == -y_coef_r:
            thoughts = [T(Operations, ('Add', x_coef_l, x_coef_r)),
                        T(Operations, ('Add', y_coef_l, y_coef_r)),
                        T(Operations, ('Add', const_l, const_r))]
        else:
            thoughts = [T(Mul_both, (x_coef_l, y_coef_l, const_l, abs(x_coef_r))),
                        T(Mul_both, (x_coef_r, y_coef_r, const_r, abs(x_coef_l))),
                        T(Elim,
                         ((x_coef_l * abs(x_coef_r), y_coef_l * abs(x_coef_r), const_l * abs(x_coef_r)),
                         (x_coef_r * abs(x_coef_l), y_coef_r * abs(x_coef_l), const_r * abs(x_coef_l)))
                         )
                        ]
        return thoughts

    @staticmethod
    def get_answer(args):
        left, right = args
        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        if x_coef_l == x_coef_r:
            return 'y', y_coef_l - y_coef_r, const_l - const_r
        elif x_coef_l == -x_coef_r:
            return 'y', y_coef_l + y_coef_r, const_l + const_r
        elif y_coef_l == y_coef_r:
            return 'x', x_coef_l - x_coef_r, const_l - const_r
        elif y_coef_l == -y_coef_r:
            return 'x', x_coef_l + x_coef_r, const_l + const_r
        else:
            return Elim.get_answer(((x_coef_l * abs(x_coef_r), y_coef_l * abs(x_coef_r), const_l * abs(x_coef_r)),
                                    (x_coef_r * abs(x_coef_l), y_coef_r * abs(x_coef_l), const_r * abs(x_coef_l))))


class Substitute(Problem):
    """Substitute a variable with a constant in 2d linear equation and solve it
    <GO><SUBSTITUTE>2x+3y=5<SEP>x=13/3<SOLVE>
        <GO>2<MUL>13/3=26/3<STOP>
        <GO>5<SUB>26/3=-11/3<STOP>
        3y=-11/3<STOP>
    <GO><SUBSTITUTE>2x+3y=5<SEP>y=13/3<SOLVE>
        <GO>3<MUL>13/3=13<STOP>
        <GO>5<SUB>13=-8<STOP>
        2x=-8<STOP>
    """
    name = 'Substitute'
    dependencies = {
        Operations: lambda config: config,
    }
    symbols = ['<SUBSTITUTE>', '<SOLVE>', '<SEP>', 'x', 'y', '+', '-', '/']

    def generate(self):
        x_coef, y_coef, const = self.sample_linear_2d(self.config['max_digits'], 1)
        var = random.sample(('x', 'y'), 1)[0]
        var_value = self.sample_fraction(self.config['max_digits'], reduce=True)
        return x_coef, y_coef, const, var, var_value

    @staticmethod
    def question(args):
        # <GO><SUBSTITUTE>2x+3y=5<SEP>x=13/3<SOLVE>
        x_coef, y_coef, const, var, var_value = args

        q_list = ['<GO><SUBSTITUTE>',
                  make_linear_2d(x_coef, y_coef, const),
                  f'<SEP>{var}=',
                  frac_to_str(var_value),
                  '<SOLVE>']

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        # 3y=-11/3<STOP>
        x_coef, y_coef, const, var, var_value = args
        if var == 'x':
            expr = make_linear_2d(0, y_coef, Sub_frac.get_answer((const, Mul_frac.get_answer((var_value, x_coef)))))
        else:
            expr = make_linear_2d(x_coef, 0, Sub_frac.get_answer((const, Mul_frac.get_answer((var_value, y_coef)))))
        return f'{expr}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        x_coef, y_coef, const, var, var_value = args
        if var == 'x':
            thoughts = [T(Operations, ('Mul', x_coef, var_value)),
                        T(Operations, ('Sub', const, Mul_frac.get_answer((var_value, x_coef))))]
        else:
            thoughts = [T(Operations, ('Mul', y_coef, var_value)),
                        T(Operations, ('Sub', const, Mul_frac.get_answer((var_value, y_coef))))]
        return thoughts

    @staticmethod
    def get_answer(args):
        x_coef, y_coef, const, var, var_value = args
        if var == 'x':
            return 'y', y_coef, Sub_frac.get_answer((const, Mul_frac.get_answer((var_value, x_coef))))
        else:
            return 'x', x_coef, Sub_frac.get_answer((const, Mul_frac.get_answer((var_value, y_coef))))


class Linear_2d(Problem):
    """Solve 2D Linear Equation
    E.g.,
        <GO><LINEAR_2D>x+y=5<SEP>2x-y=8<SOLVE>
            <GO><ELIM>x+y=5<SEP>2x-y=8<SOLVE>3x=13<STOP>
            <GO><LINEAR_1D>3x=13<SOLVE>x=13/3<STOP>
            <GO><SUBSTITUTE>x+y=5<SEP>x=13/3<SOLVE>y=2/3<STOP>
            <GO><LINEAR_1D>y=2/3<SOLVE>y=2/3<STOP>
            x=13/3<SEP>y=2/3<STOP>
        <GO><LINEAR_2D>-4x+3y=-8<SEP>7x-4y=14<SOLVE>
            <ELIM>-4x+3y=-8<SEP>7x-4y=14<SOLVE>5y=0<STOP>
            <GO><LINEAR_1D>5y=0<SOLVE>y=0<STOP>
            <GO><SUBSTITUTE>-4x+3y=-8<SEP>y=0<SOLVE>-4x=-8<STOP>
            <GO><LINEAR_1D>-4x=-8<SOLVE>x=2<STOP>
            x=2<SEP>y=0<STOP>
        # Impossible
        <GO><LINEAR_2D>-32x+24y=-64<SEP>32x-24y=56<SOLVE>
            <GO><ELIM>-32x+24y=-64<SEP>32x-24y=56<SOLVE>0=-8<STOP>
            <NO_SOL><STOP>
        # Indeterminate
        <GO><LINEAR_2D>-4x+3y=-8<SEP>-4x+3y=-8<SOLVE>
            <GO><ELIM>-4x+3y=-8<SEP>-4x+3y=-8<SOLVE>0=0<STOP>
            <INDET><STOP>
    """
    name = 'Linear_2d'
    dependencies = {
        Linear_1d: lambda config: config,
        Elim: lambda config: config,
        Substitute: lambda config: config,
    }
    symbols = ['<LINEAR_2D>', '<SOLVE>', '<SEP>', 'x', 'y', '<NO_SOL>', '<INDET>', '-']

    def generate(self):
        x_coef_l, y_coef_l, const_l = self.sample_linear_2d(self.config['max_digits'], min_num=1)
        x_coef_r, y_coef_r, const_r = self.sample_linear_2d(self.config['max_digits'], min_num=1)

        # There should be at least one variable to be eliminated
        if x_coef_l * x_coef_r == 0 and y_coef_l * y_coef_r == 0:
            return self.generate()
        return (x_coef_l, y_coef_l, const_l), (x_coef_r, y_coef_r, const_r)

    @staticmethod
    def question(args):
        # <GO><LINEAR_2D>-4x+3y=-8<SEP>7x-4y=14<SOLVE>
        left, right = args

        q_list = ['<GO><LINEAR_2D>',
                  make_linear_2d(left[0], left[1], left[2]),
                  '<SEP>',
                  make_linear_2d(right[0], right[1], right[2]),
                  '<SOLVE>']

        return ''.join(q_list)

    @staticmethod
    def answer(args):
        # x=2<SEP>y=3<STOP>
        left, right = args

        if Linear_2d.is_impossible_2d(left, right):
            return '<NO_SOL><STOP>'
        if Linear_2d.is_indeterminate_2d(left, right):
            return '<INDET><STOP>'

        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        x_value = frac_to_str((const_l * y_coef_r - const_r * y_coef_l,
                               x_coef_l * y_coef_r - x_coef_r * y_coef_l),
                              reduce = True)
        y_value = frac_to_str((const_l * x_coef_r - const_r * x_coef_l,
                               y_coef_l * x_coef_r - y_coef_r * x_coef_l),
                              reduce = True)

        return f'x={x_value}<SEP>y={y_value}<STOP>'

    @staticmethod
    def thought(args) -> list[T]:
        left, right = args

        thoughts = [T(Elim, (left, right))]

        if Linear_2d.is_impossible_2d(left, right) or Linear_2d.is_indeterminate_2d(left, right):
            return thoughts

        var, linear, const = Elim.get_answer((left, right))
        thoughts.append(T(Linear_1d, (var, linear, const)))

        var_value = Reduce.get_answer((const, linear))
        x_coef_l, y_coef_l, const_l = left
        thoughts.append(T(Substitute, (x_coef_l, y_coef_l, const_l, var, var_value)))

        var, linear, const = Substitute.get_answer((x_coef_l, y_coef_l, const_l, var, var_value))
        thoughts.append(T(Linear_1d, (var, linear, const)))

        return thoughts

    @staticmethod
    def is_impossible_2d(left, right):
        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        if x_coef_l * y_coef_r == x_coef_r * y_coef_l:
            if const_l * x_coef_r != const_r * x_coef_l:
                return True
        return False

    @staticmethod
    def is_indeterminate_2d(left, right):
        x_coef_l, y_coef_l, const_l = left
        x_coef_r, y_coef_r, const_r = right

        if x_coef_l * y_coef_r == x_coef_r * y_coef_l:
            if const_l * x_coef_r == const_r * x_coef_l:
                return True
        return False


# class Quadratic_1d(Problem):
#     """Solve 2D Linear Equation
#     E.g.,
#         # Base Case: Factorized
#         <GO><QUADRATIC_1D>(x+1)(x+2)=0<SOLVE>
#             <GO><LINEAR_1D>x+1=0<SOLVE>x=-1<STOP>
#             <GO><LINEAR_1D>x+2=0<SOLVE>x=-2<STOP>
#             x=-1<SEP>x=-2<STOP>
#         <GO><QUADRATIC_1D>(2x+1)(3x+2)=0<SOLVE>
#             <GO><LINEAR_1D>2x+1=0<SOLVE>x=-1/2<STOP>
#             <GO><LINEAR_1D>3x+2=0<SOLVE>x=-2/3<STOP>
#             x=-1/2<SEP>x=-2/3<STOP>
#         <GO><QUADRATIC_1D>(2x+1)^2=0<SOLVE>
#             <GO><LINEAR_1D>2x+1=0<SOLVE>x=-1/2<STOP>
#             x=-1/2<STOP>
#         # Solve Recursively: Perfect Square
#         <GO><QUADRATIC_1D>(2x+1)(2x+1)=0<SOLVE>
#             <GO><LINEAR_1D>(2x+1)^2=0<SOLVE>x=-1/2<STOP>
#             x=-1/2<STOP>
#         <GO><QUADRATIC_1D>x^2+2x+1=0<SOLVE>
#             <GO><FACTORIZE>x^2+2x+1<SOLVE>(x+1)^2<STOP>
#             <GO><QUADRATIC_1D>(x+1)^2=0<SOLVE>x=-1<STOP>
#             x=-1<STOP>
#         # Solve Recursively: Factorizable
#         <GO><QUADRATIC_1D>2x^2+3x+1=0<SOLVE>
#             <GO><FACTORIZE>2x^2+3x+1<SOLVE>(2x+1)(x+1)<STOP>
#             <QUADRATIC_1D>(2x+1)(x+1)=0<SOLVE>x=-1/2<SEP>x=-1<STOP>
#             x=-1/2<SEP>x=-1<STOP>
#         # Base case: Fail to factorize -> Check Discriminant
#         <GO><QUADRATIC_1D>2x^2+3x+2=0<SOLVE>
#             <GO><FACTORIZE>2x^2+3x+2<SOLVE><FAIL><STOP>
#             <GO><DISCRIMINANT>2x^2+3x+2<SOLVE>-7<STOP>
#             <NO_SOL><STOP>
#         # Base case: Call Quadratic Formula
#         <GO><QUADRATIC_1D>x^2+3x+1=0<SOLVE>
#             <GO><FACTORIZE>x^2+3x+1<SOLVE><FAIL><STOP>
#             <GO><DISCRIMINANT>x^2+3x+1<SOLVE>5<STOP>
#             <GO><QUAD_FORMULA>x^2+3x+1=0<SOLVE>x=-3/2+<SQRT>5<SEP>x=-3/2-<SQRT>5<STOP>
#             x=-3/2+<SQRT>5<SEP>x=-3/2-<SQRT>5<STOP>
#     """
#     name = 'Quadratic_1d'
#     dependencies = {
#         Linear_1d: lambda config: config,
#         # Factorize: lambda config: config,
#         # Discriminant: lambda config: config,
#         # Quad_formula: lambda config: config,
#     }
#     symbols = ['<QUADRATIC_1D>', '<SOLVE>', '<SEP>', '^', 'x', 'y', '<NO_SOL>', '-', '(', ')']
#
#     def generate(self, log_uniform=True):
#         pass
#     @staticmethod
#     def question(args):
#         pass
#     @staticmethod
#     def answer(args):
#         pass
#     @staticmethod
#     def thought(args) -> list[T]:
#         return []

def make_linear_2d(x_coef, y_coef, const):
    """Make 2d linear expression with its coefficients"""
    equation = []

    if x_coef == 0 and y_coef == 0:
        return f'0={frac_to_str(const)}'

    if x_coef != 0:
        if x_coef < 0:
            equation.append('-')
        if abs(x_coef) != 1:
            equation.append(f'{abs(x_coef)}')
        equation.append('x')

    if y_coef != 0:
        if y_coef < 0:
            equation.append('-')
        elif x_coef != 0:
            equation.append('+')
        if abs(y_coef) != 1:
            equation.append(f'{abs(y_coef)}')
        equation.append('y')

    equation.extend([f'={frac_to_str(const)}'])
    return ''.join(equation)