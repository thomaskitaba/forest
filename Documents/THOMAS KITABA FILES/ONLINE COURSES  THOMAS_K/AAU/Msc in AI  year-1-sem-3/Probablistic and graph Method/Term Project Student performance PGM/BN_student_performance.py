#!/usr/bin/python3
# ==============================================================================
# 0. SETUP AND IMPORTS
# General requirement for all PGM tasks.
# ==============================================================================
import numpy as np
import pandas as pd

from pgmpy.models import DiscreteBayesianNetwork, MarkovModel, BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, GibbsSampling, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator

# Optional: for structure learning
from pgmpy.estimators import HillClimbSearch, BicScore

# For visualization
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(0)

print("=== PGM Project: Student Performance via Bayesian Networks ===\n")

# ==============================================================================
# 1. PGM FUNDAMENTALS (Lab 1 style)
# Core concept: DiscreteFactor operations
# ==============================================================================
print("SECTION 1: Factor (Potential) basics\n")

# Create a toy DiscreteFactor φ(A, B) with binary A and B
phi = DiscreteFactor(['A', 'B'], cardinality=[2, 2], values=[1000, 1, 5, 100])
print("Factor φ(A, B):\n", phi)

# 1.1 Marginalize B => φ(A)
phi_A = phi.marginalize(['B'], inplace=False)
print("\nMarginalize B (sum out B) => φ(A):\n", phi_A)

# 1.2 Reduce (condition) on A = 0 => φ(B | A=0)
phi_B_given_A0 = phi.reduce([('A', 0)], inplace=False)
print("\nReduction: φ(B | A = 0):\n", phi_B_given_A0)

# 1.3 Multiply with another factor φ(B, C)
phi_BC = DiscreteFactor(['B', 'C'], cardinality=[2, 2], values=[2, 5, 5, 2])
phi_ABC = phi * phi_BC
print("\nMultiply φ(A, B) × φ(B, C) => φ(A, B, C):\n", phi_ABC)


# ==============================================================================
# 2. BAYESIAN NETWORK & INFERENCE (Lab 2 style)
# Core concept: Directed structure + CPDs + inference (VE & Gibbs)
# ==============================================================================
print("\nSECTION 2: Bayesian Networks & Inference\n")

# Toy “restaurant” model (from your template)
restaurant = DiscreteBayesianNetwork([
    ('location', 'cost'),
    ('quality', 'cost'),
    ('cost', 'no_of_people'),
    ('location', 'no_of_people')
])

# Define CPDs manually
cpd_location = TabularCPD('location', 2, [[0.6], [0.4]])
cpd_quality = TabularCPD('quality', 3, [[0.3], [0.5], [0.2]])
cpd_cost = TabularCPD(
    'cost', 2,
    [[0.8, 0.6, 0.1, 0.6, 0.6, 0.05],
     [0.2, 0.4, 0.9, 0.4, 0.4, 0.95]],
    evidence=['location', 'quality'],
    evidence_card=[2, 3]
)
cpd_no = TabularCPD(
    'no_of_people', 2,
    [[0.6, 0.8, 0.1, 0.6],
     [0.4, 0.2, 0.9, 0.4]],
    evidence=['cost', 'location'],
    evidence_card=[2, 2]
)

restaurant.add_cpds(cpd_location, cpd_quality, cpd_cost, cpd_no)
assert restaurant.check_model(), "Model is invalid!"

# Exact inference: Variable Elimination
inf_ve = VariableElimination(restaurant)
q_ve = inf_ve.query(variables=['cost'], evidence={'quality': 0})
print("\nExact inference: P(cost | quality = low):\n", q_ve)

# Approximate inference: Gibbs Sampling
inf_gibbs = GibbsSampling(restaurant)
q_gibbs = inf_gibbs.query(variables=['cost'], evidence={'quality': 0},
                          virtual_samples=5000, seed=42)
print("\nApproximate inference (Gibbs): P(cost | quality = low):\n", q_gibbs)


# ==============================================================================
# 3. PARAMETER LEARNING (Lab 4 style, toy example)
# Core concept: Learn CPDs (MLE) from observed data
# ==============================================================================
print("\nSECTION 3: Parameter Learning via MLE (toy data)\n")

# Generate synthetic data for A, B, C binary
n_samples = 2000
data = np.random.randint(0, 2, size=(n_samples, 3))
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("Sample data (first 5 rows):\n", df.head())

# Define known structure A → B → C
model_learn = DiscreteBayesianNetwork([('A', 'B'), ('B', 'C')])

# Use MaximumLikelihoodEstimator to estimate CPDs
mle = MaximumLikelihoodEstimator(model_learn, data=df)
cpd_A = mle.estimate_cpd(node='A')
cpd_B = mle.estimate_cpd(node='B')
cpd_C = mle.estimate_cpd(node='C')
print("\nEstimated CPD P(A):\n", cpd_A)
print("\nEstimated CPD P(B | A):\n", cpd_B)
print("\nEstimated CPD P(C | B):\n", cpd_C)

model_learn.add_cpds(cpd_A, cpd_B, cpd_C)
assert model_learn.check_model()

# Inference on learned model
inf_learn = VariableElimination(model_learn)
q_learn = inf_learn.query(variables=['A'], evidence={'C': 1})
print("\nInference on learned model: P(A | C = 1):\n", q_learn)


# ==============================================================================
# 4. REAL DATA: UCI Student Performance + MLE CPD learning + inference
# ==============================================================================
print("\nSECTION 4: Real Data — Student Performance Modeling\n")

def load_student_data(path_mat, path_por=None):
    df_mat = pd.read_csv(path_mat, sep=';')
    if path_por:
        df_por = pd.read_csv(path_por, sep=';')
        df = pd.concat([df_mat, df_por], ignore_index=True)
    else:
        df = df_mat
    return df

def preprocess_student(df):
    # Drop missing
    df = df.dropna()
    # Select features
    sel = ['studytime', 'famsup', 'health', 'G1', 'G2', 'G3']
    df2 = df[sel].copy()

    # Discretize / bin categories
    df2['study_cat'] = df2['studytime'].apply(lambda x: 'Low' if x <= 2 else 'High')
    df2['health_cat'] = df2['health'].apply(lambda x: 'Low' if x <= 3 else 'High')
    df2['famsup_cat'] = df2['famsup'].apply(lambda x: 'Yes' if x == 'yes' else 'No')
    df2['G1_cat'] = df2['G1'].apply(lambda x: 'Low' if x <= 10 else 'High')
    df2['G2_cat'] = df2['G2'].apply(lambda x: 'Low' if x <= 10 else 'High')
    df2['performance'] = df2['G3'].apply(lambda x: 'Good' if x >= 11 else 'Poor')

    # Final subset
    final = df2[[
        'study_cat', 'famsup_cat', 'health_cat',
        'G1_cat', 'G2_cat', 'performance'
    ]]
    return final

# Load your local data (download from UCI / Kaggle into working folder)
df_raw = load_student_data('student-mat.csv', path_por='student-por.csv')
print("Raw data shape:", df_raw.shape)

df_proc = preprocess_student(df_raw)
print("Processed head:\n", df_proc.head())

# Optionally, do a train/test split; here we use all for learning
# Define a plausible structure (expert-chosen)
student_model = DiscreteBayesianNetwork([
    ('study_cat', 'G1_cat'),
    ('famsup_cat', 'G1_cat'),
    ('G1_cat', 'G2_cat'),
    ('study_cat', 'G2_cat'),
    ('health_cat', 'performance'),
    ('G2_cat', 'performance')
])

# Learn CPDs from data (MLE)
mle_est = MaximumLikelihoodEstimator(student_model, data=df_proc)
# For each variable, estimate CPD
for var in student_model.nodes():
    cpd = mle_est.estimate_cpd(node=var)
    student_model.add_cpds(cpd)

assert student_model.check_model(), "Learned student model is invalid!"

# Show learned CPDs for a few variables
print("\nLearned CPD for performance:\n", student_model.get_cpds('performance'))
print("\nLearned CPD for G2_cat:\n", student_model.get_cpds('G2_cat'))

# Inference: P(performance | evidence)
infer_student = VariableElimination(student_model)
q_perf = infer_student.query(
    variables=['performance'],
    evidence={
        'study_cat': 'High',
        'famsup_cat': 'Yes',
        'health_cat': 'High'
    }
)
print("\nInference: P(performance | High study, Yes famsup, High health):\n", q_perf)

# (Optional) You can loop over test instances, do classification, compute accuracy,
# but that's an extra extension.

# ==============================================================================
# 5. VISUALIZATION of the Student Model
# ==============================================================================
print("\nSECTION 5: Visualization\n")

plt.figure(figsize=(8, 6))
nx.draw(
    student_model,
    with_labels=True,
    node_size=2000,
    node_color='lightblue',
    font_size=10,
    font_weight='bold',
    arrowsize=15
)
plt.title("Learned Bayesian Network: Student Performance")
plt.show()
