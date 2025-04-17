"""
Genetic Algorithm for Multi-Objective Optimization

This module implements a genetic algorithm for solving multi-objective optimization problems
with mixed variables. It includes custom operators for mating, repair, and survival selection.

Note: The MixedVariableMating class is adapted from the pymoo library 
(https://github.com/anyoptimization/pymoo) with minor modifications.
"""
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.survival import Survival
from pymoo.core.individual import Individual
from pymoo.core.infill import InfillCriterion
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Binary, Real, Integer, Choice

from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.rm import ChoiceRandomMutation

from pymoo.indicators.hv import Hypervolume
from pymoo.optimize import minimize

class MixedVariableMating(InfillCriterion):

    def __init__(self,
                 selection=RandomSelection(),
                 crossover=None,
                 mutation=None,
                 repair=None,
                 eliminate_duplicates=True,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)

        if crossover is None:
            crossover = {
                Binary: UX(),
                Real: SBX(),
                Integer: SBX(vtype=float, repair=RoundingRepair()),
                Choice: UX(),
            }

        if mutation is None:
            mutation = {
                Binary: BFM(),
                Real: PM(),
                Integer: PM(vtype=float, repair=RoundingRepair()),
                Choice: ChoiceRandomMutation(),
            }

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):

        # So far we assume all crossover need the same amount of parents and create the same number of offsprings
        XOVER_N_PARENTS = 2
        XOVER_N_OFFSPRINGS = 2

        # the variables with the concrete information
        vars = problem.vars

        # group all the variables by their types
        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)

            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        # # all different recombinations (the choices need to be split because of data types)
        recomb = []
        for clazz, list_of_vars in vars_by_type.items():
            if clazz == Choice:
                for e in list_of_vars:
                    recomb.append((clazz, [e]))
            else:
                recomb.append((clazz, list_of_vars))

        # create an empty population that will be set in each iteration
        off = Population.new(X=[{} for _ in range(n_offsprings)])

        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(problem, pop, n_select, XOVER_N_PARENTS, **kwargs)

        for clazz, list_of_vars in recomb:

            crossover = self.crossover[clazz]
            assert crossover.n_parents == XOVER_N_PARENTS and crossover.n_offsprings == XOVER_N_OFFSPRINGS

            _parents = [
                [Individual(X=np.array([parent.X[var] for var in list_of_vars], dtype="O" if clazz is Choice else None))
                  for parent in parents]
                for parents in pop
            ]

            _vars = {e: vars[e] for e in list_of_vars}
            _xl = np.array([vars[e].lb if hasattr(vars[e], "lb") else None for e in list_of_vars])
            _xu = np.array([vars[e].ub if hasattr(vars[e], "ub") else None for e in list_of_vars])
            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover(_problem, _parents, **kwargs)

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs)

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        return off

class MixedVariableSampling(Sampling):
  def _do(self, problem, n_samples, **kwargs):
      V = {name: var.sample(n_samples) for name, var in problem.vars.items()}
      X = []
      for k in range(n_samples):
          X.append({name: V[name][k] for name in problem.vars.keys()})
      return X

class MixedVariableGAWithRepair(GeneticAlgorithm):
    def _advance(self, infills=None, **kwargs):
        if infills is None:
            infills = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        infills = self.repair.do(self.problem, infills)
        self.evaluator.eval(self.problem, infills)
        combined_pop = Population.merge(self.pop, infills)
        self.pop = self.survival.do(self.problem, combined_pop, n_survive=self.pop_size)

    def _finalize(self):
        self.pop = self.repair.do(self.problem, self.pop)
        self.evaluator.eval(self.problem, self.pop)
        super()._finalize()

class MixedVariableDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(self.to_numpy(a.X), self.to_numpy(b.X))

    def to_numpy(self, x):
        if isinstance(x, dict):
            return np.array([x[key] for key in sorted(x.keys())])
        elif isinstance(x, Individual):
            return self.to_numpy(x.X)
        else:
            return np.array(list(x.values()))

class HVContributionSurvival(Survival):
    def __init__(self):
        super().__init__()
        self.hv = Hypervolume(ref_point=np.array([1e6, 1e6]))

    def _do(self, problem, pop, n_survive, D=None, **kwargs):
        F = pop.get("F")
        ref_point = get_reference_point(F)
        self.hv.ref_point = ref_point
        total_hv = self.hv.do(F)

        contributions = []
        for i in range(len(F)):
            hv_without_i = self.hv.do(np.delete(F, i, axis=0))
            contribution = total_hv - hv_without_i
            contributions.append(contribution)

        I = np.argsort(contributions)[::-1]
        return pop[I[:n_survive]]

class MixtureRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for x in X:
            mixture_sum = sum(x[var] for var in MIXTURE_VARS)
            if mixture_sum > 0:
                for var in MIXTURE_VARS:
                    x[var] /= mixture_sum
        return X

loaded_model1 = load('lightgbm_model_final1.joblib') #Load pretrained regression model for variable 1
loaded_model2 = load('lightgbm_model_final2.joblib') #Load pretrained regression model for variable 2
MIXTURE_VARS = ["содержаниеАЦ", "СодержаниеНЦ", "ПС"]

def get_reference_point(F):
    ideal_point = np.min(F, axis=0)
    nadir_point = np.max(F, axis=0)
    return nadir_point + 1e-6

def calculate_penalty(value, min_value, max_value):
    if value < min_value:
        return (min_value - value)
    elif value > max_value:
        return (value - max_value)
    else:
        return 0

def evaluate_fitness(X: pd.DataFrame, category: int) -> np.ndarray:
    X = X.copy()
    X.insert(0, 'Название', category)
    JBSA = -loaded_model1.predict(X)
    R = -loaded_model2.predict(X)
    return JBSA, R

class MultiObjectiveMixedVariableProblem(ElementwiseProblem):
    def __init__(self, category):
        vars = {
            "Растворитель1дма02meohdma41": Choice(options=list(range(2))),
            "Лавсан": Choice(options=list(range(2))),
            "Содержаниеполимера": Real(bounds=(0, 30)),
            "ПВПК30": Real(bounds=(0, 0.05)),
            "ZnBIM": Real(bounds=(0, 0.1)),
            "содержаниеАЦ": Real(bounds=(0, 1)),
            "СодержаниеНЦ": Real(bounds=(0, 1)),
            "ПС": Real(bounds=(0, 1))
        }
        super().__init__(vars=vars, n_obj=2, n_constr=0)
        self.category = category

    def _evaluate(self, X, out, *args, **kwargs):

        X_df = pd.DataFrame([X], columns=self.vars.keys())
        JBSA, R = evaluate_fitness(X_df, self.category)

        JBSA = np.asarray(JBSA)
        R = np.asarray(R)

        penalty_jbsa = np.zeros_like(JBSA)
        penalty_jbsa[JBSA < -400] = 1e6
        penalty_r = np.zeros_like(R)
        penalty_r[R < -100] = 1e6

        penalty_ac = calculate_penalty(X["содержаниеАЦ"], 0.15, 0.30)
        penalty_nc = calculate_penalty(X["СодержаниеНЦ"], 0.60, 0.85)
        penalty_ps = calculate_penalty(X["ПС"], 0, 0.1)

        scale_factor = 100
        penalty_features = scale_factor * (penalty_ac + penalty_nc + penalty_ps)
        penalty = penalty_jbsa + penalty_r + penalty_features

        out["F"] = np.column_stack([JBSA + penalty, R + penalty])

def plot_pareto_front(res, category, n_best=5):
    pareto_front = -res.F
    plt.figure(figsize=(10, 7))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], s=30, facecolors='none', edgecolors='blue', label='Pareto Front')

    best_indices = np.argsort(pareto_front[:, 1])[-n_best:][::-1]
    best_solutions = pareto_front[best_indices]
    plt.scatter(best_solutions[:, 0], best_solutions[:, 1], s=100, color='red', label='Best Solutions')

    for i, solution in enumerate(best_solutions):
        plt.annotate(f'{i+1}', (solution[0], solution[1]), xytext=(3, 3),
                     textcoords='offset points', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
                     fontsize=8)

    plt.title(f"Pareto Front for Category {category}")
    plt.xlabel("JBSA")
    plt.ylabel("R")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    print(f"\nTop {n_best} solutions for category {category}:")
    for i, index in enumerate(best_indices):
        print(f"Solution {i+1}:")
        print(f"JBSA = {pareto_front[index, 0]:.2f}, R = {pareto_front[index, 1]:.2f}")
        print(f"Variables: {res.X[index]}\n")

categories = X_analize['Название'].unique()
for i, category in enumerate(categories):
    problem = MultiObjectiveMixedVariableProblem(category)
    algorithm = MixedVariableGAWithRepair(
        pop_size=200,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        survival=HVContributionSurvival(),
        repair=MixtureRepair(),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 80),
                   seed=i+1,
                   verbose=False,
                   save_history=True)

    print("Best solution found for category:", category)
    plot_pareto_front(res, category, n_best=5)

    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])
    plt.figure(figsize=(10, 7))
    plt.title(f"Convergence for Category {category}")
    plt.plot(n_evals, opt, "--")
    plt.xlabel("Number of Evaluations")
    plt.ylabel("Objective Value")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    #SAVE_RESULT
    category_results_df = pd.DataFrame(columns=['Категория'] + list(problem.vars.keys()) + ['JBSA', 'R'])
    pareto_front = -res.F
    for solution, loss in zip(res.X, pareto_front):
        decoded_category = encoder.inverse_transform([[category]])[0][0]
        decoded_solution = {}
        for var, value in solution.items():
            decoded_solution[var] = value

        solution_df = pd.DataFrame(decoded_solution, index=[0]) 
        solution_df.insert(0, 'Категория', decoded_category)
        solution_df['JBSA'] = loss[0]
        solution_df['R'] = loss[1]
        category_results_df = pd.concat([category_results_df, solution_df], ignore_index=True)

    if i == 0:
        all_results_df = category_results_df
    else:
        all_results_df = pd.concat([all_results_df, category_results_df], ignore_index=True)
