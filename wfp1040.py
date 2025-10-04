# Requires: !pip install pulp in Google Collab
import pulp
import numpy as np

# ---------- Example input (replace with real data) ----------
m = 8   # number of sites
n = 18  # number of transporters

# Example site demands (tons) and rates
S = [196.562,	495.076,	254.053,	103.664,149.648,	343.098, 0,	147.738]   # length m
r = [50.44, 48.88, 43.68, 44.20, 26.52, 33.80, 21.84, 18.20]          # USD/ton

# Example percentages (fractions)
p = [0.068, 0.049, 0.067, 0.041, 0.053, 0.069, 0.047, 0.047, 0.044,
     0.037, 0.055, 0.048, 0.069, 0.069, 0.069, 0.066, 0.054, 0.045]

# ---------- Derived quantities ----------
T = sum(S)
R = sum(S[j] * r[j] for j in range(m))
t = [p[i] * T for i in range(n)]   # target tons per transporter
q = [p[i] * R for i in range(n)]   # target revenue per transporter

# ---------- Create LP ----------
prob = pulp.LpProblem("CargoAllocation", pulp.LpMinimize)

# Decision variables: allocation x[i][j] and binary y[i][j]
x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Continuous")
      for j in range(m)] for i in range(n)]

y = [[pulp.LpVariable(f"y_{i}_{j}", cat="Binary")
      for j in range(m)] for i in range(n)]

# Auxiliary abs deviation vars
u = [pulp.LpVariable(f"u_{i}", lowBound=0) for i in range(n)]  # tonnage deviation
v = [pulp.LpVariable(f"v_{i}", lowBound=0) for i in range(n)]  # revenue deviation

# Constraints: site demands
for j in range(m):
    prob += pulp.lpSum(x[i][j] for i in range(n)) == S[j]

# Linking constraints for min 10 and max 40 when assigned
for i in range(n):
    for j in range(m):
        prob += x[i][j] <= 50 * y[i][j]        # if y=0, x=0 ; if y=1, x ≤ 40
        prob += x[i][j] >= 10 * y[i][j]        # if y=1, x ≥ 10

# Tonnage deviation constraints
for i in range(n):
    total_tons = pulp.lpSum(x[i][j] for j in range(m))
    prob += u[i] >= total_tons - t[i]
    prob += u[i] >= -(total_tons - t[i])

# Revenue deviation constraints
for i in range(n):
    total_rev = pulp.lpSum(r[j] * x[i][j] for j in range(m))
    prob += v[i] >= total_rev - q[i]
    prob += v[i] >= -(total_rev - q[i])

# Objective: weighted sum of deviations
alpha = 1.0   # weight for tonnage deviation
beta = 0.001  # weight for money deviation
prob += alpha * pulp.lpSum(u[i] for i in range(n)) + beta * pulp.lpSum(v[i] for i in range(n))

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=1))

# Extract solution
x_sol = np.array([[pulp.value(x[i][j]) for j in range(m)] for i in range(n)])
tons_assigned = x_sol.sum(axis=1)
revenue_assigned = (x_sol * np.array(r)).sum(axis=1)

print("Transporter | Target tons | Assigned tons | Target rev | Assigned rev")
for i in range(n):
    print(f"{i:3d} | {t[i]:14.2f} | {tons_assigned[i]:15.2f} | {q[i]:14.2f} | {revenue_assigned[i]:10.2f}")

print("\nTonnage allocation (tons) per transporter per site:")
print("Rows = Transporters, Columns = Sites")
print(" " * 14 + " ".join([f"S{j:2d}" for j in range(m)]))

for i in range(n):
    row_vals = " ".join([f"{x_sol[i][j]:6.2f}" for j in range(m)])
    print(f"Transporter {i:2d}: {row_vals}")
