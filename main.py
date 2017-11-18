# -*- coding: utf-8 -*-
__author__ = 'Jinkey'

import sympy as syp

# 解一元二次方程组
print("解一元二次方程组" + "="*50)
x, y = syp.symbols('x y')
print syp.solve([2*x - y - 3, 3*x + y-7], [x, y])

# 求极限值
print("求极限值" + "="*50)
n = syp.Symbol('n')
print syp.limit(((n+3)/(n+2))**n, n, syp.oo)
print syp.Limit((syp.cos(x) - 1)/x, x, 0) # 只显示求极限公式不计算

# 求定积分
print("求定积分" + "="*50)
t, x = syp.symbols('t x')
m = syp.integrate(syp.sin(t)/(syp.pi-t), (t, 0, x))
n = syp.integrate(m, (x, 0, syp.pi))
print n
print syp.integrate(syp.exp(-x**2 - y**2), (x, -syp.oo, syp.oo), (y, -syp.oo, syp.oo))
print syp.Integral(syp.log(x)**2, x).doit()  # integrate 是求积分并计算，Integral 是求积分不计算

# 解微分方程
print("解微分方程" + "="*50)
f = syp.Function('f')  # 意义同Symbol，但这是函数因变量的符号化
x = syp.Symbol('x')
print syp.dsolve(syp.diff(f(x), x)-2*x*f(x), f(x))
print syp.Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), syp.sin(x))

# simplify判断两个式子是否相等
print("simplify判断两个式子是否相等" + "="*50)
x = syp.symbols('x')
a = x**2 + 6*x
b = 9
c = (x+3)**2
print syp.simplify(a + b)
print c.equals(a+b)
print syp.simplify(c-(a+b))

# 代入自变量的值求解
print("代入自变量的值求解" + "="*50)
str_expr = syp.sympify("x**2 + 3*x - 1/2")
print str_expr.subs(x, 2)
print str_expr.evalf(subs={x: 2.4})

# 保留 2 位小数
print("保留 2 位小数" + "="*50)
print(syp.sqrt(8).evalf(2))

# 结合 numpy 计算多个自变量值的解
import numpy
a = numpy.arange(10)
expr = syp.sin(x)
f = syp.lambdify(x, expr, "numpy")
print(f(a))

syp.pprint(syp.Integral(syp.sqrt(1/x), x), use_unicode=True)
print(syp.pretty(syp.Integral(syp.sqrt(1/x), x), use_unicode=False))

# 展开公式
print syp.expand((x + 1)**2)
print syp.expand((x + 1)*(x - 2) - (x - 1)*x)

# 折叠公式
print syp.factor(x**3 - x**2 + x - 1)

print syp.cancel((x**2 + 2*x + 1)/(x**2 + x))
print syp.simplify((x**2 + 2*x + 1)/(x**2 + x))
print syp.factor((x**2 + 2*x + 1)/(x**2 + x))

# 拆开分式
expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
print syp.apart(expr)

# 三角函数
print syp.asin(1)
print syp.trigsimp(syp.sin(x)**4 - 2*syp.cos(x)**2*syp.sin(x)**2 + syp.cos(x)**4)
print syp.simplify(syp.sin(x)**4 - 2*syp.cos(x)**2*syp.sin(x)**2 + syp.cos(x)**4)
# 展开三角函数
print syp.expand_trig(syp.sin(x + y))
print syp.expand(syp.sin(x + y))

# 幂函数
print syp.sqrt(x) == x**syp.Rational(1, 2)

z, t, c = syp.symbols('z t c')
print syp.powsimp(t**c*z**c)
print syp.expand_power_base((z*t)**c)
z, t, c = syp.symbols('z t c', positive=True)
print syp.powsimp(t**c*z**c)  # 自变量大于0才能合并
print syp.expand_power_base((z*t)**c)

print syp.powdenest((z**x)**y)

# 对数函数
x, y = syp.symbols('x y', positive=True)
print syp.expand_log(syp.log(x*y))
print syp.expand_log(syp.log(x**2))
print syp.logcombine(syp.log(x) + syp.log(y))

# 阶乘
print syp.factorial(x).subs(x, 4)

# 求导
print syp.diff(syp.cos(x), x)
print syp.Derivative(syp.cos(x), x).doit() # 只展示求导的公式不实际计算 ,doit 执行计算
print syp.diff(x**4, x, x, x)  # 多阶求导
print syp.diff(x**4, x, 3)  # 多阶求导
# diff(expr, x, y, y, z, z, z, z)  # 偏导数
# diff(expr, x, y, 2, z, 4)  # 偏导数


# 解方程
print syp.solveset(syp.Eq(x**2, 1), x)
print syp.solveset(x**2 - 1, x)
print syp.linsolve([x + y + z - 1, x + y + 2*z - 3 ], (x, y, z))


# 矩阵
M = syp.Matrix([[1, 2, 3], [3, 2, 1]])
N = syp.Matrix([[0, 1, 1], [0, 1, 1]])
print M*N   # 矩阵乘法
print M+N   # 矩阵加法
print 3*N
print M**2  # 矩阵的平方
print M.row(0)  # 获取矩阵的行
print M.col(-1) # 获取矩阵的列
print M.col_del(0)  # 删除第一列
print M.row_del(1)  # 删除第二行
print M.row_insert(1, syp.Matrix([[0, 4]]))
print M.col_insert(0, syp.Matrix([1, -2]))
print M.T   # 矩阵的转置

print syp.eye(3) # 对角矩阵
print syp.zeros(2, 3)
print syp.ones(2, 3)
print syp.diag(1, 2, 3)

M = syp.Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
print M.det()