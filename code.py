from dataclasses import dataclass, replace
from itertools import product
import time
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Model:
    utility: np.ndarray
    audit: np.ndarray
    emissions_without: np.ndarray
    emissions_with: np.ndarray
    reporting_cost: np.ndarray
    payoff: np.ndarray
    price: np.ndarray
    signal_index: np.ndarray
    gross_return: float
    payment_cap: float
    audit_cost: float
    budget: float
    reduction_target: float
    incentive_margin: float
    reference_law: np.ndarray


def validate(m):
    t, a = m.utility.shape
    if m.audit.ndim != 3 or m.audit.shape[:2] != (t, a):
        raise ValueError('Invalid audit distribution dimensions')
    k = m.audit.shape[2]
    if t < 1 or a < 2 or k < 2:
        raise ValueError('At least one type, two actions and two signals are required')
    if m.emissions_without.shape != (t, a) or m.emissions_with.shape != (t, a):
        raise ValueError('Invalid emissions dimensions')
    if m.reporting_cost.shape != (t,) or m.reference_law.shape != (k,):
        raise ValueError('Invalid reporting cost or benchmark dimensions')
    if m.payoff.shape[0] != len(m.signal_index) or m.payoff.ndim != 2:
        raise ValueError('Invalid financial payoff dimensions')
    if m.price.shape != (m.payoff.shape[1],):
        raise ValueError('Invalid traded asset prices')
    if np.any(m.signal_index < 0) or np.any(m.signal_index >= k):
        raise ValueError('Signal indices exceed the audit support')
    if set(np.unique(m.signal_index)) != set(range(k)):
        raise ValueError('Every audit signal must be represented in the financial state space')
    for value in (m.utility, m.audit, m.emissions_without, m.emissions_with, m.reporting_cost, m.payoff, m.price, m.reference_law):
        if not np.all(np.isfinite(value)):
            raise ValueError('Nonfinite model primitive')
    if np.min(m.audit) < -1e-12 or np.max(np.abs(m.audit.sum(axis=2) - 1)) > 1e-10:
        raise ValueError('Audit probabilities must be nonnegative and normalized')
    if np.min(m.reference_law) < -1e-12 or abs(m.reference_law.sum() - 1) > 1e-10:
        raise ValueError('Benchmark probability law is invalid')
    if np.any(m.reporting_cost < 0) or np.any(m.emissions_without < 0) or np.any(m.emissions_with < 0):
        raise ValueError('Costs and emissions must be nonnegative')
    if min(m.gross_return, m.payment_cap, m.budget) <= 0 or min(m.audit_cost, m.reduction_target, m.incentive_margin) < 0:
        raise ValueError('Invalid financial or environmental parameters')
    if m.audit_cost > m.budget + 1e-9:
        return
    q = np.column_stack((np.ones(m.payoff.shape[0]), m.payoff))
    rhs = np.r_[1., m.gross_return * m.price]
    witness = linprog(np.zeros(len(m.signal_index)), A_eq=q.T, b_eq=rhs, bounds=[(1e-8, None)] * len(m.signal_index), method='highs')
    if not witness.success:
        raise ValueError('Financial market does not admit a strictly positive martingale probability')


def eligible_actions(m):
    baselines = []
    effects = []
    eligible = []
    for t in range(m.utility.shape[0]):
        no_contract = np.flatnonzero(m.utility[t] >= m.utility[t].max() - 1e-10)
        baseline = float(np.min(m.emissions_without[t, no_contract]))
        incremental = baseline - m.emissions_with[t]
        baselines.append(tuple(int(i) for i in no_contract))
        effects.append(incremental)
        eligible.append(tuple(int(i) for i in np.flatnonzero(incremental >= m.reduction_target - 1e-10)))
    return tuple(eligible), np.array(effects), tuple(baselines)


def financial_matrices(m):
    h = np.eye(m.audit.shape[2], dtype=float)[m.signal_index]
    return h, m.payoff


def hedge_primal(m, reward):
    h, x = financial_matrices(m)
    n = x.shape[1]
    A = np.column_stack((-x, -m.gross_return * np.ones(x.shape[0])))
    result = linprog(np.r_[m.price, 1.], A_ub=A, b_ub=-h @ reward, bounds=[(None, None)] * (n + 1), method='highs')
    if not result.success:
        raise RuntimeError(f'Hedge primal failed: {result.message}')
    return float(result.fun), result


def hedge_dual(m, reward):
    h, x = financial_matrices(m)
    N = x.shape[0]
    Aeq = np.column_stack((np.ones(N), x))
    beq = np.r_[1., m.gross_return * m.price]
    result = linprog(-(h @ reward) / m.gross_return, A_eq=Aeq.T, b_eq=beq, bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(f'Hedge dual failed: {result.message}')
    return float(-result.fun), result


def linear_program(m, witness, objective='funding', budget=True, cap=None, participation=True, incentive=True):
    t, a, k = m.audit.shape
    n = m.payoff.shape[1]
    dim = k + n + 1
    chosen_cap = m.payment_cap if cap is None else cap
    allowed, _, _ = eligible_actions(m)
    rows = []
    rhs = []
    families = []
    for typ, chosen in enumerate(witness):
        if chosen not in allowed[typ]:
            raise ValueError('Witness is not environmentally eligible')
        for other in allowed[typ]:
            if other == chosen:
                continue
            row = np.zeros(dim)
            row[:k] = m.audit[typ, other] - m.audit[typ, chosen]
            rows.append(row)
            rhs.append(m.utility[typ, chosen] - m.utility[typ, other])
            families.append('eligible_optimality')
        if incentive:
            for other in range(a):
                if other in allowed[typ]:
                    continue
                row = np.zeros(dim)
                row[:k] = m.audit[typ, other] - m.audit[typ, chosen]
                rows.append(row)
                rhs.append(m.utility[typ, chosen] - m.utility[typ, other] - m.incentive_margin)
                families.append('strict_incentive')
        if participation:
            row = np.zeros(dim)
            row[:k] = -m.audit[typ, chosen]
            rows.append(row)
            rhs.append(m.utility[typ, chosen] - m.reporting_cost[typ] - np.max(m.utility[typ]))
            families.append('participation')
    H, X = financial_matrices(m)
    for state in range(H.shape[0]):
        row = np.zeros(dim)
        row[:k] = H[state]
        row[k:k + n] = -X[state]
        row[-1] = -m.gross_return
        rows.append(row)
        rhs.append(0.)
        families.append('hedging')
    if budget:
        row = np.zeros(dim)
        row[k:k + n] = m.price
        row[-1] = 1.
        rows.append(row)
        rhs.append(m.budget - m.audit_cost)
        families.append('budget')
    A = np.array(rows, dtype=float).reshape((-1, dim))
    b = np.array(rhs, dtype=float)
    c = np.zeros(dim)
    if objective == 'funding':
        c[k:k + n] = m.price
        c[-1] = 1.
    elif objective == 'reference_expectation':
        c[:k] = m.reference_law
    else:
        raise ValueError('Unknown objective')
    bounds = [(0., chosen_cap)] * k + [(None, None)] * (n + 1)
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    return result, A, b, c, bounds, families


def kkt_certificate(result, A, b, c, bounds):
    if not result.success:
        return None
    lam = result.ineqlin.marginals
    low = result.lower.marginals
    up = result.upper.marginals
    lower = np.array([v[0] if v[0] is not None else 0. for v in bounds])
    upper = np.array([v[1] if v[1] is not None else 0. for v in bounds])
    value = float(b @ lam + lower @ low + upper @ up)
    primal = float(c @ result.x)
    residual = float(max(0., np.max(A @ result.x - b)))
    stationarity = float(np.max(np.abs(c - A.T @ lam - low - up)))
    inequality_complementarity = float(np.max(np.abs(lam * (b - A @ result.x))))
    box_complementarity = float(max(np.max(np.abs(low * (result.x - lower) * np.array([v[0] is not None for v in bounds]))),np.max(np.abs(up * (upper - result.x) * np.array([v[1] is not None for v in bounds])))))
    return {'primal_value': primal, 'dual_value': value, 'duality_gap': abs(primal - value), 'max_inequality_violation': residual, 'stationarity_residual': stationarity, 'inequality_complementarity': inequality_complementarity, 'bound_complementarity': box_complementarity}


def witness_solutions(m, objective='funding', budget=True, cap=None, participation=True, incentive=True):
    allowed, effects, baselines = eligible_actions(m)
    if any(not x for x in allowed):
        return {'status': 'environmental_infeasible', 'eligible': allowed, 'baseline': baselines, 'effects': effects, 'enumerated': 0}
    best = None
    count = 0
    feasible = 0
    invalid = 0
    highest = 0.
    started = time.perf_counter()
    for witness in product(*allowed):
        count += 1
        result, A, b, c, bounds, families = linear_program(m, witness, objective, budget, cap, participation, incentive)
        if result.status == 2:
            continue
        if not result.success:
            raise RuntimeError(f'Numerical solver failure for witness {witness}: {result.message}')
        feasible += 1
        cert = kkt_certificate(result, A, b, c, bounds)
        if cert['duality_gap'] > 1e-5 or cert['max_inequality_violation'] > 1e-6 or cert['stationarity_residual'] > 1e-5 or cert['inequality_complementarity'] > 1e-5 or cert['bound_complementarity'] > 1e-5:
            invalid += 1
            raise RuntimeError(f'Failed LP optimality certificate for {witness}: {cert}')
        highest = max(highest, cert['duality_gap'])
        chosen_score = float(result.fun)
        if best is None or chosen_score < best['objective_value'] - 1e-8:
            w = result.x[:m.audit.shape[2]].copy()
            ph, _ = hedge_primal(m, w)
            dh, _ = hedge_dual(m, w)
            if abs(ph - dh) > 1e-6:
                raise RuntimeError('Financial superhedging primal-dual disagreement')
            best = {'status': 'optimal', 'w': w, 'witness':tuple(int(v) for v in witness), 'funding':m.audit_cost + ph, 'hedge_primal': ph, 'hedge_dual': dh, 'hedge_gap':abs(ph-dh), 'reference_expected_payment':float(m.reference_law @ w), 'objective_value':chosen_score, 'kkt': cert, 'active_families': tuple(sorted({families[i] for i in np.flatnonzero(abs(A @ result.x - b)<1e-6)}))}
    metadata = {'eligible':allowed,'baseline':baselines,'effects':effects,'enumerated':count,'feasible_witnesses':feasible,'max_witness_kkt_gap':highest,'runtime_seconds':time.perf_counter()-started}
    if best is None:
        return {'status':'infeasible',**metadata}
    return {**best, **metadata}


def independent_verification(m, sol, enforce_budget=True):
    if sol['status'] != 'optimal':
        return {'verified':False, 'reason':'No feasible contract'}
    w = sol['w']
    allowed, effects, baselines = eligible_actions(m)
    incentive_slack = []
    participation_slack = []
    eligible_tie_slack = []
    for t in range(m.utility.shape[0]):
        utility = m.utility[t] - m.reporting_cost[t] + m.audit[t] @ w
        good = np.array(allowed[t], dtype=int)
        bad = np.array([a for a in range(len(utility)) if a not in allowed[t]], dtype=int)
        best = np.max(utility[good])
        participation_slack.append(best - np.max(m.utility[t]))
        if len(bad):
            incentive_slack.append(best - np.max(utility[bad]) - m.incentive_margin)
        eligible_tie_slack.append(best - utility[sol['witness'][t]])
    primal, portfolio = hedge_primal(m,w)
    dual, q = hedge_dual(m,w)
    hedging_shortfall = float(np.max(np.eye(len(w))[m.signal_index] @ w - np.column_stack((m.payoff,m.gross_return * np.ones(len(m.signal_index)))) @ portfolio.x))
    funding = m.audit_cost + primal
    checks = {'reward_lower':float(np.min(w)), 'reward_upper_slack':float(m.payment_cap-np.max(w)), 'minimum_incentive_slack':float(min(incentive_slack)) if incentive_slack else float('inf'), 'minimum_participation_slack':float(min(participation_slack)), 'minimum_witness_slack':float(min(eligible_tie_slack)), 'budget_slack':float(m.budget-funding), 'hedging_shortfall':hedging_shortfall, 'hedge_duality_gap':abs(primal-dual),'lp_duality_gap':sol['kkt']['duality_gap'],'lp_stationarity_residual':sol['kkt']['stationarity_residual']}
    passkeys = ['reward_lower','reward_upper_slack','minimum_incentive_slack','minimum_participation_slack','minimum_witness_slack']
    if enforce_budget:
        passkeys.append('budget_slack')
    correct = all(checks[k]>=-1e-6 for k in passkeys) and checks['hedging_shortfall'] <= 1e-6 and checks['hedge_duality_gap']<=1e-6 and checks['lp_duality_gap']<=1e-5 and checks['lp_stationarity_residual']<=1e-5 and sol['kkt']['inequality_complementarity']<=1e-5 and sol['kkt']['bound_complementarity']<=1e-5
    return {'verified':bool(correct),**checks}


def farkas_witness(m, witness):
    result, A, b, c, bounds, families = linear_program(m,witness)
    if result.success:
        return {'certificate_exists':False,'status':'feasible'}
    if result.status != 2:
        raise RuntimeError('Unable to certify infeasibility: numerical solver failed')
    k = m.audit.shape[2]
    size = A.shape[0]
    cols = size+k
    obj = np.r_[b, np.ones(k)*m.payment_cap]
    free = A[:,k:].T
    equality = np.column_stack((free,np.zeros((free.shape[0],k))))
    equality = np.vstack((equality,np.r_[np.ones(size),np.zeros(k)]))
    rhs = np.r_[np.zeros(free.shape[0]),1.]
    ineq = np.column_stack((-A[:,:k].T,-np.eye(k)))
    cert = linprog(obj,A_ub=ineq,b_ub=np.zeros(k),A_eq=equality,b_eq=rhs,bounds=[(0,None)]*cols,method='highs')
    if not cert.success:
        raise RuntimeError(f'No Farkas certificate found for witness {witness}')
    lam = cert.x[:size]
    nu = cert.x[size:]
    exact_residual = float(np.max(np.abs(A[:,k:].T @ lam)))
    box_condition = float(np.max(-A[:,:k].T @ lam - nu))
    verified = cert.fun < -1e-7 and exact_residual < 1e-6 and box_condition < 1e-6
    if not verified:
        raise RuntimeError(f'Unverified Farkas certificate for witness {witness}: {cert.fun}')
    contribution = [(families[i], float(lam[i])) for i in range(size) if lam[i] > 1e-7]
    return {'certificate_exists':True,'verified':True,'separation_margin':float(-cert.fun),'dual_free_residual':exact_residual,'dual_box_residual':max(0.,box_condition),'support':contribution}


def diagnostics(m, optimum):
    if optimum['status']=='optimal':
        return {'status':'optimal','binding_families':list(optimum['active_families']),'relaxation_tests':{}}
    if optimum['status']=='environmental_infeasible':
        return {'status':'environmental_infeasible','binding_families':['environmental_eligibility'],'relaxation_tests':{}}
    settings = {'remove_budget': {'budget':False},'increase_cap': {'cap':max(1e5,m.payment_cap*1e3)},'remove_participation':{'participation':False},'remove_incentive':{'incentive':False},'increase_cap_and_remove_budget':{'budget':False,'cap':max(1e5,m.payment_cap*1e3)}}
    tests = {}
    for label,kwargs in settings.items():
        ans = witness_solutions(m,**kwargs)
        tests[label] = ans['status']=='optimal'
    if tests['remove_budget']:
        reason='budget_critical'
    elif tests['increase_cap']:
        reason='payout_cap_critical'
    elif tests['remove_participation']:
        reason='participation_critical'
    elif tests['remove_incentive']:
        reason='information_incentive_critical'
    else:
        reason='joint_constraints_or_audit_structure'
    return {'status':reason,'binding_families':[], 'relaxation_tests':tests}


def financial_model(signals, gross, audit_security=False):
    carbon = np.array([30.,50.,70.])
    state_price = np.repeat(carbon,signals)
    state_signal = np.tile(np.arange(signals),len(carbon))
    payoff = state_price[:,None]
    prices = [50.]
    if audit_security:
        proxy = 20. * state_signal / (signals-1)
        payoff = np.column_stack((payoff,proxy))
        prices.append(float(proxy.mean()/gross))
    return payoff,np.array(prices),state_signal


def named_model(name='base', audit_security=False):
    t,a,k = 2,3,3
    gross = float(np.exp(.04))
    payoff,price,signal_index = financial_model(k,gross,audit_security)
    v = np.array([[0.,-40.,-74.],[0.,-43.,-69.]])
    audits = np.array([[[.82,.12,.06],[.35,.45,.20],[.10,.18,.72]],[[.80,.14,.06],[.33,.48,.19],[.11,.20,.69]]])
    emissions = np.array([[100.,85.,64.],[100.,84.,63.]])
    h = np.array([4.,6.])
    cap=350.
    budget=270.
    target=12.
    margin=.5
    if name=='low_budget':budget=30.
    elif name=='low_cap':cap=15.
    elif name=='weak_audit':audits[:] = 1./k
    elif name=='baseline_green':v[:,2]=15.
    elif name=='high_target':target=60.
    elif name=='high_cost':v[:,1:]-=160.
    elif name=='baseline_tie':v[:,1]=0.
    elif name=='no_informative_eligible_action':
        target=30.
        audits[:,2,:] = audits[:,0,:]
    elif name=='high_budget':budget=600.
    elif name=='expensive_participation':h[:] = 140.
    elif name=='no_report_cost':h[:] = 0.
    elif name=='single_eligible':target=30.
    elif name=='audit_proxy': pass
    elif name != 'base':raise ValueError(f'Unknown case {name}')
    return Model(v,audits,emissions.copy(),emissions.copy(),h,payoff,price,signal_index,gross,cap,7.,budget,target,margin,audits[:,2,:].mean(axis=0))


def random_model(rng, actions=3, signals=3, types=2, audit_security=False, quality=None, budget=None):
    if quality is None: quality = float(rng.uniform(.12,.92))
    gross = float(np.exp(.04))
    payoff,price,signal_index = financial_model(signals,gross,audit_security)
    v = np.empty((types,actions))
    emission = np.empty((types,actions))
    p = np.empty((types,actions,signals))
    for t in range(types):
        v[t] = np.linspace(0.,-90.*rng.uniform(.5,1.6),actions)+rng.normal(0,6,actions)
        v[t,0]=0.
        emission[t]=np.linspace(100.,62.,actions)+rng.normal(0,1.5,actions)
        for a in range(actions):
            landmark = min(round(a*(signals-1)/max(actions-1,1)),signals-1)
            center=(1-quality)/signals*np.ones(signals)+quality*np.eye(signals)[landmark]
            p[t,a]=rng.dirichlet(np.maximum(center*30,.2))
    baseline = v.max(axis=1)
    reward_ref = p[:,actions-1].mean(axis=0)
    return Model(v,p,emission.copy(),emission.copy(),rng.uniform(0,10,types),payoff,price,signal_index,gross,float(rng.uniform(60,370)),7.,float(rng.uniform(45,370)) if budget is None else float(budget),float(rng.uniform(5,40)),.5,reward_ref)


import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


def robust_margin_primal(positive, negative, costs, cap):
    delta = np.asarray(positive)-np.asarray(negative)
    C = np.asarray(costs)
    scenarios, outcomes = delta.shape
    objective = np.r_[np.zeros(outcomes),-1.]
    A = np.column_stack((-delta,np.ones(scenarios)))
    result = linprog(objective,A_ub=A,b_ub=-C,bounds=[(0.,cap)]*outcomes+[(None,None)],method='highs')
    if not result.success:
        raise RuntimeError(f'Robust audit margin primal failed: {result.message}')
    return float(-result.fun),result


def robust_margin_dual(positive, negative, costs, cap):
    delta = np.asarray(positive)-np.asarray(negative)
    C = np.asarray(costs)
    scenarios, outcomes = delta.shape
    objective = np.r_[-C,cap*np.ones(outcomes)]
    A = np.column_stack((delta.T,-np.eye(outcomes)))
    equality = np.r_[np.ones(scenarios),np.zeros(outcomes)]
    result = linprog(objective,A_ub=A,b_ub=np.zeros(outcomes),A_eq=equality.reshape(1,-1),b_eq=np.array([1.]),bounds=(0,None),method='highs')
    if not result.success:
        raise RuntimeError(f'Robust audit margin dual failed: {result.message}')
    return float(result.fun),result


def gaussian_total_variation(mu_green,mu_false,standard_deviation):
    if standard_deviation <= 0 or mu_green <= mu_false:
        raise ValueError('Require positive standard deviation and a higher green mean')
    return 2.*norm.cdf((mu_green-mu_false)/(2*standard_deviation))-1.


def sde_terminal_checks(seed=20261012,paths=80000):
    rng=np.random.default_rng(seed)
    S0=50.
    T=1.
    mu=.06
    sigma=.30
    intensity=.25
    mark_mean=-.08
    mark_sd=.14
    kappa=math.exp(mark_mean+mark_sd**2/2.)-1.
    event_counts=rng.poisson(intensity*T,size=paths)
    compound_jumps=event_counts*mark_mean+np.sqrt(event_counts)*mark_sd*rng.normal(size=paths)
    brown=np.sqrt(T)*rng.normal(size=paths)
    terminal=S0*np.exp((mu-.5*sigma**2-intensity*kappa)*T+sigma*brown+compound_jumps)
    if not np.all(np.isfinite(terminal)) or not np.all(terminal>0):
        raise AssertionError('Nonpositive or nonfinite jump-diffusion terminal price')
    records=[]
    for p in (1.,2.):
        observations=terminal**p
        theoretical=S0**p*math.exp(p*mu*T+.5*p*(p-1)*sigma**2*T+intensity*T*(math.exp(p*mark_mean+.5*p*p*mark_sd**2)-1.-p*kappa))
        sample=float(np.mean(observations))
        stderr=float(np.std(observations,ddof=1)/math.sqrt(paths))
        standard_score=abs(sample-theoretical)/stderr
        if standard_score>6.:
            raise AssertionError(f'SDE moment sanity check failed for p={p}: z={standard_score:.4f}')
        records.append({'moment_order':int(p),'analytical_moment':theoretical,'monte_carlo_moment':sample,'monte_carlo_standard_error':stderr,'absolute_standardized_deviation':standard_score,'sample_paths':paths})
    return records


import unittest
from dataclasses import replace
from itertools import product
import numpy as np


class ContractTests(unittest.TestCase):
    def test_all_named_cases(self):
        for name in ('base','low_budget','low_cap','weak_audit','baseline_green','high_target','high_cost','baseline_tie','no_informative_eligible_action','high_budget','expensive_participation','no_report_cost','single_eligible','audit_proxy'):
            with self.subTest(name=name):
                m = named_model(name, name=='audit_proxy')
                validate(m)
                answer = witness_solutions(m)
                if answer['status'] == 'optimal':
                    check = independent_verification(m, answer)
                    self.assertTrue(check['verified'], (name,check))
                elif answer['status'] == 'infeasible':
                    allowed, _, _ = eligible_actions(m)
                    for witness in product(*allowed):
                        result = farkas_witness(m, witness)
                        self.assertTrue(result['verified'])
                else:
                    self.assertEqual(answer['status'],'environmental_infeasible')

    def test_expected_category(self):
        kinds = {'base':'optimal','low_budget':'infeasible','low_cap':'infeasible','weak_audit':'infeasible','baseline_green':'environmental_infeasible','high_target':'environmental_infeasible','baseline_tie':'optimal','no_informative_eligible_action':'infeasible','high_budget':'optimal'}
        for name, expected in kinds.items():
            self.assertEqual(witness_solutions(named_model(name))['status'],expected,name)

    def test_closed_form_two_action_binary(self):
        R = float(np.exp(.04))
        payoff,price,signal_index = financial_model(2,R)
        model = Model(np.array([[0.,-100.]]),np.array([[[.7,.3],[.1,.9]]]),np.array([[100.,60.]]),np.array([[100.,60.]]),np.array([0.]),payoff,price,signal_index,R,250.,5.,250.,15.,.5,np.array([.1,.9]))
        answer = witness_solutions(model)
        self.assertEqual(answer['status'],'optimal')
        expected = (100.+.5)/(.9-.3)
        self.assertAlmostEqual(float(answer['w'][0]),0.,places=6)
        self.assertAlmostEqual(float(answer['w'][1]),expected,places=6)
        self.assertAlmostEqual(answer['funding'],5.+expected/R,places=6)
        self.assertTrue(independent_verification(model,answer)['verified'])

    def test_baseline_tie_worst_case(self):
        m = named_model('baseline_tie')
        allowed, effects, baselines = eligible_actions(m)
        self.assertEqual(baselines,((0,1),(0,1)))
        self.assertEqual(allowed,((2,),(2,)))
        self.assertAlmostEqual(effects[0,2],21.)

    def test_nonspanned_audit_funding(self):
        m = named_model('base')
        for seed in range(15):
            rng = np.random.default_rng(seed)
            w = rng.uniform(0,m.payment_cap,m.audit.shape[2])
            primal,_ = hedge_primal(m,w)
            dual,_ = hedge_dual(m,w)
            self.assertAlmostEqual(primal,float(np.max(w)/m.gross_return),places=7)
            self.assertAlmostEqual(dual,primal,places=7)

    def test_partially_spanned_audit(self):
        plain = named_model('base')
        richer = named_model('base',audit_security=True)
        w = np.array([0.,50.,100.])
        self.assertLessEqual(hedge_primal(richer,w)[0],hedge_primal(plain,w)[0]+1e-7)
        for m in (plain,richer):
            for seed in range(10):
                w = np.random.default_rng(seed).uniform(0,300,len(m.reference_law))
                self.assertAlmostEqual(hedge_primal(m,w)[0],hedge_dual(m,w)[0],places=6)

    def test_budget_monotonicity(self):
        m = named_model('base')
        level = witness_solutions(m)['funding']
        self.assertEqual(witness_solutions(replace(m,budget=level+1e-3))['status'],'optimal')
        self.assertEqual(witness_solutions(replace(m,budget=level-1e-2))['status'],'infeasible')
        self.assertEqual(witness_solutions(replace(m,budget=level+100))['status'],'optimal')

    def test_cap_monotonicity(self):
        m = named_model('base')
        a = witness_solutions(replace(m,payment_cap=30))
        b = witness_solutions(replace(m,payment_cap=350))
        self.assertEqual(a['status'],'infeasible')
        self.assertEqual(b['status'],'optimal')

    def test_payout_benchmark(self):
        m = named_model('base')
        standard = witness_solutions(m,objective='reference_expectation')
        proposed = witness_solutions(m)
        self.assertEqual(standard['status'],'optimal')
        self.assertTrue(independent_verification(m,standard)['verified'])
        self.assertLessEqual(proposed['funding'],standard['funding']+1e-6)
        self.assertLessEqual(standard['reference_expected_payment'],proposed['reference_expected_payment']+1e-6)

    def test_randomized_optimality(self):
        rng = np.random.default_rng(20261012)
        for i in range(70):
            m = random_model(rng,actions=2+i%3,signals=2+i%4,types=1+i%3,audit_security=i%4==0)
            validate(m)
            solution = witness_solutions(m)
            if solution['status']=='optimal':
                check = independent_verification(m,solution)
                self.assertTrue(check['verified'],(i,check))
                self.assertLess(solution['max_witness_kkt_gap'],1e-6)
            if solution['status']=='infeasible' and solution['enumerated']<=18 and i%5==0:
                choices,_,_=eligible_actions(m)
                for witness in product(*choices):
                    self.assertTrue(farkas_witness(m,witness)['verified'])

    def test_diagnostics(self):
        for name in ('low_budget','low_cap','weak_audit','no_informative_eligible_action','baseline_green'):
            m=named_model(name)
            result=witness_solutions(m)
            detail=diagnostics(m,result)
            self.assertIn('status',detail)
            self.assertNotEqual(detail['status'],'optimal')
        self.assertTrue(diagnostics(named_model('low_budget'),witness_solutions(named_model('low_budget')))['relaxation_tests']['remove_budget'])
        self.assertTrue(diagnostics(named_model('low_cap'),witness_solutions(named_model('low_cap')))['relaxation_tests']['increase_cap'])

    def test_robust_audit_minimax(self):
        rng=np.random.default_rng(241)
        for scenario_count in (1,2,4):
            for outcomes in (2,3,5):
                for repeat in range(5):
                    green=rng.dirichlet(np.ones(outcomes),size=scenario_count)
                    false=rng.dirichlet(np.ones(outcomes),size=scenario_count)
                    costs=rng.uniform(0,40,scenario_count)
                    upper,_=robust_margin_primal(green,false,costs,160.)
                    lower,_=robust_margin_dual(green,false,costs,160.)
                    self.assertAlmostEqual(upper,lower,places=7)
        green=np.array([[.2,.8],[.8,.2]])
        false=np.array([[.8,.2],[.2,.8]])
        upper,_=robust_margin_primal(green,false,np.array([10.,10.]),250.)
        self.assertAlmostEqual(upper,-10.,places=7)

    def test_gaussian_audit_discrimination(self):
        from scipy.stats import norm
        for spread in (1.,3.,7.):
            for noise in (.5,2.,5.):
                midpoint=spread/2
                direct=norm.sf(midpoint,loc=0,scale=noise)-norm.sf(midpoint,loc=spread,scale=noise)
                margin=gaussian_total_variation(spread,0.,noise)
                self.assertAlmostEqual(margin,-direct,places=10)

    def test_sde_exact_terminal_moments(self):
        moments=sde_terminal_checks(seed=20261012,paths=25000)
        self.assertEqual(len(moments),2)
        self.assertTrue(all(row['absolute_standardized_deviation']<6 for row in moments))

    def test_reproducible_inputs(self):
        one=random_model(np.random.default_rng(223),types=2)
        two=random_model(np.random.default_rng(223),types=2)
        np.testing.assert_allclose(one.audit,two.audit)
        np.testing.assert_allclose(one.utility,two.utility)
        np.testing.assert_allclose(one.emissions_without,two.emissions_without)


import argparse
import csv
import io
import json
import math
import platform
import sys
import time
import unittest
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CASES = ('base','low_budget','low_cap','weak_audit','baseline_green','high_target','high_cost','baseline_tie','no_informative_eligible_action','high_budget','expensive_participation','no_report_cost','single_eligible','audit_proxy')


def save_csv(path, records, columns=None):
    if not records:
        raise ValueError(f'No rows to save: {path}')
    names = columns if columns is not None else tuple(records[0])
    with path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=names,extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow({key:record.get(key) for key in names})


def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2,default=to_json),encoding='utf-8')


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.floating):
        return float(value)
    if isinstance(value,np.bool_):
        return bool(value)
    raise TypeError(f'Object of type {type(value)} is not JSON serializable')


def model_record(m,identifier):
    return {'id':identifier,'utility':m.utility,'audit_probabilities':m.audit,'emissions_without_contract':m.emissions_without,'emissions_with_contract':m.emissions_with,'reporting_cost':m.reporting_cost,'traded_asset_payoffs':m.payoff,'traded_asset_prices':m.price,'financial_signal_index':m.signal_index,'gross_return':m.gross_return,'payment_cap':m.payment_cap,'audit_cost':m.audit_cost,'budget':m.budget,'reduction_target':m.reduction_target,'incentive_margin':m.incentive_margin,'reference_audit_distribution':m.reference_law}


def status_row(label,m,result,benchmark=None,category=None):
    valid = independent_verification(m,result) if result['status']=='optimal' else {}
    row = {'instance':label,'status':result['status'],'diagnostic_category':category or result['status'],'types':int(m.utility.shape[0]),'actions':int(m.utility.shape[1]),'signals':int(m.audit.shape[2]),'traded_assets':int(m.payoff.shape[1]),'budget':float(m.budget),'payment_cap':float(m.payment_cap),'target_reduction':float(m.reduction_target),'enumerated_witnesses':result['enumerated'],'feasible_witnesses':result.get('feasible_witnesses',0),'solve_seconds':result.get('runtime_seconds',0.),'verified':valid.get('verified',False),'funding':result.get('funding'),'reference_expected_payment':result.get('reference_expected_payment'),'reward_schedule':json.dumps(result['w'].tolist()) if result['status']=='optimal' else '', 'chosen_witness':json.dumps(result.get('witness',[])),'lp_duality_gap':result.get('kkt',{}).get('duality_gap'),'lp_stationarity_residual':result.get('kkt',{}).get('stationarity_residual'),'hedge_duality_gap':result.get('hedge_gap'),'minimum_incentive_slack':valid.get('minimum_incentive_slack'),'minimum_participation_slack':valid.get('minimum_participation_slack'),'budget_slack':valid.get('budget_slack'),'benchmark_funding':benchmark.get('funding') if benchmark and benchmark['status']=='optimal' else None,'benchmark_expected_payment':benchmark.get('reference_expected_payment') if benchmark and benchmark['status']=='optimal' else None}
    if result['status']=='optimal' and not valid['verified']:
        raise AssertionError(f'Independent verification failed for {label}: {valid}')
    if benchmark is not None and result['status']=='optimal':
        if benchmark['status']!='optimal':
            raise AssertionError(f'Reference optimization unexpectedly infeasible: {label}')
        check=independent_verification(m,benchmark)
        if not check['verified'] or result['funding']>benchmark['funding']+1e-5 or benchmark['reference_expected_payment']>result['reference_expected_payment']+1e-5:
            raise AssertionError(f'Objective ordering or benchmark feasibility failed: {label}')
    return row


def run_named(out):
    data=[]
    certificates=[]
    details=[]
    for label in CASES:
        m=named_model(label,audit_security=label=='audit_proxy')
        validate(m)
        sol=witness_solutions(m)
        bench=witness_solutions(m,objective='reference_expectation') if sol['status']=='optimal' else None
        diag=diagnostics(m,sol)
        data.append(status_row(label,m,sol,bench,diag['status']))
        details.append({'name':label,'diagnostic':diag,'eligible_actions':eligible_actions(m)[0],'baseline_actions':eligible_actions(m)[2]})
        if sol['status']=='infeasible':
            allowed,_,_=eligible_actions(m)
            for wi,witness in enumerate(product(*allowed)):
                cert=farkas_witness(m,witness)
                if not cert.get('verified'):
                    raise AssertionError(f'Missing Farkas certificate: {label}/{witness}')
                certificates.append({'case':label,'witness':json.dumps(witness),'certificate_verified':cert['verified'],'separation_margin':cert['separation_margin'],'dual_free_residual':cert['dual_free_residual'],'dual_box_residual':cert['dual_box_residual'],'constraint_support':json.dumps(cert['support'])})
    save_csv(out/'named_cases.csv',data)
    if certificates:
        save_csv(out/'farkas_certificates.csv',certificates)
    save_json(out/'named_diagnostics.json',details)
    return data,certificates


def run_random(out,n,seed):
    rng=np.random.default_rng(seed)
    rows=[]
    total_checked=0
    total_infeasible=0
    model_file=out/'synthetic_models.jsonl'
    with model_file.open('w',encoding='utf-8') as f:
        for i in range(n):
            num_a=int(rng.integers(2,5))
            num_k=int(rng.integers(2,6))
            num_t=int(rng.integers(1,4))
            use_proxy=bool(i%5==0)
            m=random_model(rng,actions=num_a,signals=num_k,types=num_t,audit_security=use_proxy)
            validate(m)
            f.write(json.dumps(model_record(m,f'random_{i:04d}'),default=to_json)+'\n')
            sol=witness_solutions(m)
            bench=witness_solutions(m,objective='reference_expectation') if sol['status']=='optimal' else None
            diag=diagnostics(m,sol) if sol['status']!='optimal' else {'status':'optimal','relaxation_tests':{}}
            row=status_row(f'random_{i:04d}',m,sol,bench,diag['status'])
            row['audit_distinguishability']=float(np.mean([.5*np.sum(abs(m.audit[t,0]-m.audit[t,-1])) for t in range(num_t)]))
            row['diagnostics_json']=json.dumps(diag['relaxation_tests'])
            row['complete_farkas_checked']=False
            row['farkas_checked_witnesses']=0
            if sol['status']=='infeasible':
                total_infeasible+=1
                allowed,_,_=eligible_actions(m)
                for witness in product(*allowed):
                    cert=farkas_witness(m,witness)
                    if not cert.get('verified'):
                        raise AssertionError(f'Farkas certificate failed for random_{i}/{witness}')
                    row['farkas_checked_witnesses']+=1
                row['complete_farkas_checked']=True
                total_checked+=1
            rows.append(row)
    save_csv(out/'random_cases.csv',rows)
    return rows,{'infeasible_random_instances':total_infeasible,'fully_farkas_certified_random_instances':total_checked}


def change_audit_quality(m,strength):
    k=m.audit.shape[2]
    p=(1.-strength)*np.ones_like(m.audit)/k+strength*m.audit
    return replace(m,audit=p,reference_law=p[:,-1,:].mean(axis=0))


def run_sensitivity(out):
    base=named_model('base')
    rows=[]
    grids=[('audit_signal_strength',np.linspace(.05,1.,25)),('payment_cap',np.linspace(20,350,24)),('budget',np.linspace(20,350,24)),('reduction_target',np.linspace(2,50,25)),('reporting_cost_multiplier',np.linspace(0,8,24))]
    for param,values in grids:
        for value in values:
            if param=='audit_signal_strength':
                m=change_audit_quality(base,float(value))
            elif param=='payment_cap':
                m=replace(base,payment_cap=float(value))
            elif param=='budget':
                m=replace(base,budget=float(value))
            elif param=='reduction_target':
                m=replace(base,reduction_target=float(value))
            else:
                m=replace(base,reporting_cost=base.reporting_cost*float(value))
            sol=witness_solutions(m)
            row=status_row(f'{param}_{value:.8f}',m,sol)
            row['parameter']=param
            row['parameter_value']=float(value)
            rows.append(row)
    save_csv(out/'sensitivity.csv',rows)
    return rows


def run_scaling(out,seed,repeats):
    rng=np.random.default_rng(seed+1009)
    rows=[]
    sizes=[(2,2,1),(3,3,2),(4,3,2),(4,4,3),(5,5,3),(6,5,3)]
    for a,k,t in sizes:
        for rep in range(repeats):
            m=random_model(rng,actions=a,signals=k,types=t,budget=1000,quality=.7)
            validate(m)
            sol=witness_solutions(m,budget=False)
            check=independent_verification(m,sol,enforce_budget=False) if sol['status']=='optimal' else {}
            if sol['status']=='optimal' and not check['verified']:
                raise AssertionError('Failed scaling verification')
            rows.append({'actions':a,'signals':k,'types':t,'replicate':rep,'status':sol['status'],'witnesses':sol['enumerated'],'feasible_witnesses':sol.get('feasible_witnesses',0),'time_seconds':sol.get('runtime_seconds',0.),'lp_duality_gap':sol.get('kkt',{}).get('duality_gap'),'hedge_duality_gap':sol.get('hedge_gap'),'verified':check.get('verified',False)})
    save_csv(out/'scaling.csv',rows)
    return rows


def check_analytic_results(out,seed):
    rng=np.random.default_rng(seed+238)
    margin_results=[]
    for j in range(75):
        scenarios=int(rng.integers(1,5))
        signals=int(rng.integers(2,6))
        green=rng.dirichlet(np.ones(signals),size=scenarios)
        false=rng.dirichlet(np.ones(signals),size=scenarios)
        costs=rng.uniform(0,100,size=scenarios)
        primal,_=robust_margin_primal(green,false,costs,250.)
        dual,_=robust_margin_dual(green,false,costs,250.)
        gap=abs(primal-dual)
        if gap>1e-6:
            raise AssertionError(f'Robust margin minimax discrepancy: {gap}')
        margin_results.append({'instance':j,'scenarios':scenarios,'signals':signals,'audit_margin_primal':primal,'audit_margin_mixture_dual':dual,'absolute_minimax_gap':gap})
    save_csv(out/'robust_audit_minimax.csv',margin_results)
    gaussian=[]
    for mean_gap in (1.,3.,7.):
        for standard_dev in (.5,1.,2.,3.,5.):
            gaussian.append({'green_minus_false_mean':mean_gap,'common_standard_deviation':standard_dev,'total_variation_analytical':gaussian_total_variation(mean_gap,0.,standard_dev)})
    save_csv(out/'gaussian_audit.csv',gaussian)
    moments=sde_terminal_checks(seed+552,paths=80000)
    save_csv(out/'sde_terminal_moments.csv',moments)
    return {'minimax_checks':len(margin_results),'max_minimax_gap':max(r['absolute_minimax_gap'] for r in margin_results),'gaussian_cases':len(gaussian),'sde_moment_checks':len(moments),'max_sde_standardized_error':max(r['absolute_standardized_deviation'] for r in moments)}


def make_plots(out, named,random_rows,sens,scaling):
    categories=Counter(r['diagnostic_category'] for r in random_rows)
    fig,ax=plt.subplots(figsize=(10,5.3))
    names,nums=zip(*sorted(categories.items(),key=lambda x:-x[1]))
    ax.bar(range(len(names)),nums)
    ax.set_xticks(range(len(names)),names,rotation=23,ha='right')
    ax.set_ylabel('Number of synthetic instances')
    ax.set_title('Feasibility diagnostics across synthetic models')
    fig.tight_layout()
    fig.savefig(out/'classification.png',dpi=210)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7.6))
    for ax,param in zip(axs.flat,('audit_signal_strength','payment_cap','budget','reporting_cost_multiplier')):
        vals=[r for r in sens if r['parameter']==param]
        x=[r['parameter_value'] for r in vals]
        y=[r['funding'] if r['status']=='optimal' else float('nan') for r in vals]
        ax.plot(x,y,marker='o',markersize=3)
        ax.set_title(param.replace('_',' ').title())
        ax.set_xlabel(param.replace('_',' '))
        ax.set_ylabel('Minimum guaranteed expenditure')
        ax.grid(alpha=.2)
    fig.suptitle('One-factor sensitivity with remaining parameters held fixed')
    fig.tight_layout()
    fig.savefig(out/'sensitivity.png',dpi=210)
    plt.close(fig)
    good=[r for r in random_rows if r['status']=='optimal']
    fig,ax=plt.subplots(figsize=(7.5,6))
    if good:
        ax.scatter([r['benchmark_funding'] for r in good],[r['funding'] for r in good],alpha=.55,s=17)
        hi=max(max(r['benchmark_funding'] for r in good),max(r['funding'] for r in good))
        ax.plot([0,hi],[0,hi],linestyle='--')
    ax.set(xlabel='Guaranteed capital under minimum reference expectation',ylabel='Minimum guaranteed capital',title='Objectives evaluated on the same feasible contract set')
    fig.tight_layout()
    fig.savefig(out/'objective_comparison.png',dpi=210)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    sizes=sorted(set((r['actions'],r['signals'],r['types']) for r in scaling))
    x=np.arange(len(sizes))
    means=[np.mean([r['time_seconds'] for r in scaling if (r['actions'],r['signals'],r['types'])==z]) for z in sizes]
    ax.bar(x,means)
    ax.set_xticks(x,[f'{a}/{k}/{t}' for a,k,t in sizes])
    ax.set(xlabel='Actions / signals / types',ylabel='Mean wall-clock solve time (seconds)',title='Finite witness-enumeration scaling')
    fig.tight_layout()
    fig.savefig(out/'scaling.png',dpi=210)
    plt.close(fig)


def markdown_table(rows,keys):
    cells=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
    for record in rows:
        val=[]
        for key in keys:
            v=record.get(key)
            if isinstance(v,float):
                val.append(f'{v:.6g}' if math.isfinite(v) else 'N/A')
            elif v is None:
                val.append('N/A')
            else:
                val.append(str(v))
        cells.append('| '+' | '.join(val)+' |')
    return '\n'.join(cells)


def write_latex_tables(out,named,random_rows):
    diagnostics=Counter(row['diagnostic_category'] for row in random_rows)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Named synthetic stress tests for robust carbon-contract feasibility. All funding values are model units.}',r'\begin{tabular}{lrrr}',r'\toprule',r'Instance & Status & Minimum capital & Reference capital \\',r'\midrule']
    for row in named:
        a=row['funding']
        b=row['benchmark_funding']
        name=row['instance'].replace('_',r'\_')
        capital=f'{a:.3f}' if a is not None else '--'
        comparator=f'{b:.3f}' if b is not None else '--'
        status='Feasible' if row['status']=='optimal' else 'Infeasible'
        lines.append(f'{name} & {status} & {capital} & {comparator} ' + '\\\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\label{tab:named_stress_tests}',r'\end{table}','',r'\begin{table}[htbp]',r'\centering',r'\caption{Feasibility classifications in the specified synthetic instance generator; frequencies are not empirical market estimates.}',r'\begin{tabular}{lr}',r'\toprule',r'Diagnostic category & Count \\',r'\midrule'])
    for category,count in sorted(diagnostics.items()):
        category_safe=category.replace('_',r'\_')
        lines.append(f'{category_safe} & {count} ' + '\\\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\label{tab:synthetic_classifications}',r'\end{table}',''])
    (out/'PAPER_TABLES_EN.tex').write_text('\n'.join(lines),encoding='utf-8')


def write_report(out,seed,n,named,certificates,random_rows,sens,scaling,validation,elapsed,random_certificates,analytic):
    categ=Counter(r['diagnostic_category'] for r in random_rows)
    feasible=[r for r in random_rows if r['status']=='optimal']
    savings=[r['benchmark_funding']-r['funding'] for r in feasible]
    expectation_penalty=[r['reference_expected_payment']-r['benchmark_expected_payment'] for r in feasible]
    upper_gap=max((r['lp_duality_gap'] for r in feasible),default=0.)
    hedge_gap=max((r['hedge_duality_gap'] for r in feasible),default=0.)
    named_cols=('instance','status','diagnostic_category','funding','benchmark_funding','verified')
    reason_rows=[{'diagnostic':key,'count':value} for key,value in sorted(categ.items(),key=lambda x:-x[1])]
    stats={'seed':seed,'sample_size':n,'synthetic_only':True,'named_cases':len(named),'named_farkas_certificates':len(certificates),'random_status_counts':dict(Counter(r['status'] for r in random_rows)),'diagnostic_counts':dict(categ),'verified_feasible':sum(bool(r['verified']) for r in feasible),'feasible_random':len(feasible),'max_lp_duality_gap':upper_gap,'max_hedge_duality_gap':hedge_gap,'mean_capital_saving_under_reference_objective':float(np.mean(savings)) if savings else None,'mean_change_in_reference_expected_payment':float(np.mean(expectation_penalty)) if expectation_penalty else None,'random_infeasibility_certification':random_certificates,'elapsed_seconds':elapsed,'validation':validation,'analytical_checks':analytic,'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()}
    save_json(out/'manifest.json',stats)
    report='\n'.join([
        '# Numerical Experiments: Robust Audit-Contingent Carbon Contract Design',
        '',
        '## Model and computational protocol',
        '',
        'All inputs are synthetic. No observed firm, verifier, or carbon-market data are used. The experiment solves the finite-type, finite-action, finite-audit problem with a robust no-contract emissions baseline and a financial superhedging budget. Strict separation from all environmentally ineligible actions is imposed by a positive incentive margin. A common reward vector is selected by enumerating eligible witness actions and solving linear programs.',
        '',
        'The primary objective minimizes upfront guaranteed funding (audit cost plus the cost of a superhedging portfolio). A reference benchmark minimizes reward expectation under a fixed exogenous audit law on exactly the same feasible contract set. The reference law is not asserted to equal the endogenous distribution generated by every optimal agent response.',
        '',
        'The main financial experiment trades a bond and a carbon-price-linked asset. An explicitly hypothetical audit-linked tradable asset is introduced in one named scenario and a subset of the synthetic simulations. The main setup includes every carbon-price and audit-signal pair and therefore the audit risk is unspanned by the carbon asset alone.',
        '',
        '## Design and sample sizes',
        '',
        f'Random seed: `{seed}`. Synthetic instances: **{n}**. Named stress-test instances: **{len(named)}**. Scaling replicates: **{len(scaling)}**. Overall run time: **{elapsed:.3f} seconds**.',
        '',
        '## Named stress tests',
        '',
        markdown_table(named,named_cols),
        '',
        '## Diagnostic distribution',
        '',
        markdown_table(reason_rows,('diagnostic','count')),
        '',
        'These counts describe this particular generator and seed; they are not estimates of real-world carbon-contract success rates. Diagnostic labels arise from structured constraint relaxation, are not mutually exclusive causal explanations, and are not statistical hypothesis tests.',
        '',
        '## Verification and optimality evidence',
        '',
        f'Independently revalidated feasible synthetic instances: **{stats["verified_feasible"]}/{len(feasible)}**. Maximum absolute joint LP primal-dual gap: **{upper_gap:.3e}**. Maximum independent financial superhedging primal-dual gap: **{hedge_gap:.3e}**.',
        '',
        f'Farkas certificates have been constructed and numerically verified for **{len(certificates)}** infeasible witness polytopes in named tests. Fully certified random infeasible instances (all enumerated witness polytopes): **{random_certificates["fully_farkas_certified_random_instances"]}/{random_certificates["infeasible_random_instances"]}**. Infeasibility due to an empty environmentally eligible set is established directly from the model constraints rather than through a witness-level Farkas certificate.',
        '',
        'The solver reports an optimal solution for every feasible witness LP. The program checks global optimality across the complete enumerated finite witness family and validates a primal-dual certificate for each successful witness LP. This is a finite-dimensional optimality claim, not a guarantee for continuous-action or unbounded-state models.',
        '',
        '## Mathematical model diagnostics',
        '',
        f'The robust-audit minimax identity was checked in **{analytic["minimax_checks"]}** independently solved primal-dual instances, with maximum discrepancy **{analytic["max_minimax_gap"]:.3e}**. Gaussian separation was tabulated in **{analytic["gaussian_cases"]}** parameter configurations. Exact terminal jump-diffusion samples passed **{analytic["sde_moment_checks"]}** first- and second-moment Monte Carlo sanity checks; the largest standardized deviation was **{analytic["max_sde_standardized_error"]:.3f}** standard errors.',
        '',
        'The Gaussian and jump-diffusion diagnostics are analytical or synthetic numerical checks of model formulas, not estimation of carbon prices from data. Brownian and compound-Poisson terminal draws use the declared physical-law parameters.',
        '',
        '## Comparison of optimization objectives',
        '',
        f'Feasible paired runs: **{len(feasible)}**. Mean reduction in guaranteed capital when optimizing capital rather than reference expected payouts: **{stats["mean_capital_saving_under_reference_objective"]:.6g}**. Mean increase in reference expected payouts: **{stats["mean_change_in_reference_expected_payment"]:.6g}**.' if feasible else 'No paired feasible runs were generated.',
        '',
        'These nonnegative differences follow from optimizing different objectives on the same feasible set; their magnitudes depend on the sampled parameters and reference audit law. They are not estimates of market savings.',
        '',
        '## Controlled sensitivity and computational scaling',
        '',
        'The files `sensitivity.csv` and `sensitivity.png` vary audit signal strength, payment caps, sponsor budgets, reporting-cost multiples, and target abatement one factor at a time. The files `scaling.csv` and `scaling.png` record time and witness counts across model sizes. The witness-enumeration approach can grow exponentially in the number of uncertain firm types; the experiment does not claim polynomial running time.',
        '',
        '## Reproducibility and scope',
        '',
        '`synthetic_models.jsonl` contains every generated random input in machine-readable form. `named_diagnostics.json` contains individual constraint-relaxation diagnostics. `farkas_certificates.csv`, `named_cases.csv`, `random_cases.csv`, `robust_audit_minimax.csv`, `gaussian_audit.csv`, `sde_terminal_moments.csv`, `sensitivity.csv`, `scaling.csv`, `manifest.json`, and the PNG figures are generated by `run_experiments.py`.',
        '',
        'The experiments do not calibrate audit distributions or counterfactual emissions from observational data, do not simulate complete SDE allowance-price paths, and do not establish novelty or empirical effectiveness. The finite no-arbitrage market is a computational assumption; the audit-linked tradable proxy is hypothetical.',
        ''
    ])
    (out/'RESULTS_EN.md').write_text(report,encoding='utf-8')
    return stats


def main():
    parser=argparse.ArgumentParser(description='Reproducible numerical experiments for robust carbon-contract optimization')
    parser.add_argument('--out',default='outputs')
    parser.add_argument('--samples',type=int,default=500)
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--scaling-repeats',type=int,default=4)
    args=parser.parse_args()
    if args.samples<1 or args.scaling_repeats<1:
        parser.error('Sample counts and scaling replicates must be positive')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    validation=stream.getvalue()
    (out/'TEST_RESULTS_EN.txt').write_text(validation,encoding='utf-8')
    if not result.wasSuccessful():
        raise RuntimeError(f'Automated unit tests failed. See {out / "TEST_RESULTS_EN.txt"}')
    named,certificates=run_named(out)
    random_rows,certification=run_random(out,args.samples,args.seed)
    sensitivity=run_sensitivity(out)
    scaling=run_scaling(out,args.seed,args.scaling_repeats)
    analytic=check_analytic_results(out,args.seed)
    make_plots(out,named,random_rows,sensitivity,scaling)
    write_latex_tables(out,named,random_rows)
    elapsed=time.perf_counter()-started
    stats=write_report(out,args.seed,args.samples,named,certificates,random_rows,sensitivity,scaling,validation,elapsed,certification,analytic)
    print(f'PASS: {result.testsRun} automated tests')
    print(f'PASS: {stats["verified_feasible"]}/{stats["feasible_random"]} feasible synthetic solutions independently checked')
    print(f'PASS: maximum joint LP duality gap = {stats["max_lp_duality_gap"]:.3e}')
    print(f'PASS: maximum hedge duality gap = {stats["max_hedge_duality_gap"]:.3e}')
    print(f'PASS: {certification["fully_farkas_certified_random_instances"]}/{certification["infeasible_random_instances"]} random infeasible instances completely Farkas-certified')
    print(f'PASS: maximum audit-minimax duality gap = {analytic["max_minimax_gap"]:.3e}')
    print(f'FINISHED: {args.samples} synthetic instances, {len(named)} named tests, {len(scaling)} scaling runs')
    print(f'REPORT: {(out/"RESULTS_EN.md").resolve()}')


from scipy.optimize import milp, Bounds, LinearConstraint
import hashlib


def mixed_integer_solution(m, objective='funding', time_limit=30.0):
    allowed, effects, baselines = eligible_actions(m)
    if any(len(a) == 0 for a in allowed):
        return {'status': 'environmental_infeasible'}
    types, actions, signals = m.audit.shape
    assets = m.payoff.shape[1]
    candidates = [(t, a) for t in range(types) for a in allowed[t]]
    core = signals + assets + 1
    dimension = core + len(candidates)
    rows = []
    lower = []
    upper = []
    def add(row, lo=-np.inf, hi=np.inf):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)
    for t in range(types):
        row = np.zeros(dimension)
        for j, (tt, a) in enumerate(candidates):
            if tt == t:
                row[core + j] = 1.
        add(row, 1., 1.)
    def activated(t, a, coefficients, rhs):
        row = np.zeros(dimension)
        row[:signals] = coefficients
        big = max(0., m.payment_cap * np.maximum(coefficients, 0).sum() - rhs) + 1.
        j = candidates.index((t, a))
        row[core + j] = big
        add(row, hi=rhs + big)
    for t, a in candidates:
        for g in allowed[t]:
            if g != a:
                activated(t, a, m.audit[t, g] - m.audit[t, a], m.utility[t, a] - m.utility[t, g])
        for b in range(actions):
            if b not in allowed[t]:
                activated(t, a, m.audit[t, b] - m.audit[t, a], m.utility[t, a] - m.utility[t, b] - m.incentive_margin)
        activated(t, a, -m.audit[t, a], m.utility[t, a] - m.reporting_cost[t] - np.max(m.utility[t]))
    h, x = financial_matrices(m)
    for i in range(x.shape[0]):
        row = np.zeros(dimension)
        row[:signals] = h[i]
        row[signals:signals + assets] = -x[i]
        row[signals + assets] = -m.gross_return
        add(row, hi=0.)
    row = np.zeros(dimension)
    row[signals:signals + assets] = m.price
    row[signals + assets] = 1.
    add(row, hi=m.budget - m.audit_cost)
    costs = np.zeros(dimension)
    if objective == 'funding':
        costs[signals:signals + assets] = m.price
        costs[signals + assets] = 1.
    elif objective == 'reference_expectation':
        costs[:signals] = m.reference_law
    else:
        raise ValueError('Invalid objective')
    low = np.r_[np.zeros(signals), np.full(assets + 1, -np.inf), np.zeros(len(candidates))]
    high = np.r_[np.full(signals, m.payment_cap), np.full(assets + 1, np.inf), np.ones(len(candidates))]
    integrality = np.r_[np.zeros(core), np.ones(len(candidates))]
    started = time.perf_counter()
    result = milp(costs, integrality=integrality, bounds=Bounds(low, high), constraints=LinearConstraint(np.array(rows), np.array(lower), np.array(upper)), options={'time_limit': float(time_limit), 'mip_rel_gap': 1e-9})
    elapsed = time.perf_counter() - started
    if result.status == 2:
        return {'status': 'infeasible', 'runtime_seconds': elapsed, 'milp_status': int(result.status)}
    if result.status != 0:
        return {'status': 'unresolved', 'runtime_seconds': elapsed, 'milp_status': int(result.status), 'message': str(result.message)}
    w = result.x[:signals]
    witnesses = tuple(next(a for j, (tt, a) in enumerate(candidates) if tt == t and result.x[core + j] > .5) for t in range(types))
    ph, _ = hedge_primal(m, w)
    dh, _ = hedge_dual(m, w)
    value = m.audit_cost + ph if objective == 'funding' else float(m.reference_law @ w)
    return {'status': 'optimal', 'w': w, 'witness': witnesses, 'funding': m.audit_cost + ph, 'reference_expected_payment': float(m.reference_law @ w), 'objective_value': value, 'hedge_gap': abs(ph-dh), 'runtime_seconds': elapsed, 'milp_gap': float(getattr(result, 'mip_gap', 0.) or 0.), 'nodes': int(getattr(result, 'mip_node_count', 0) or 0), 'binary_variables': len(candidates)}


def independent_milp_check(m, result):
    if result['status'] != 'optimal':
        return result['status']
    w = result['w']
    good, _, _ = eligible_actions(m)
    for t in range(len(good)):
        utilities = m.utility[t] - m.reporting_cost[t] + m.audit[t] @ w
        best = max(utilities[a] for a in good[t])
        bad = [a for a in range(len(utilities)) if a not in good[t]]
        if bad and best - max(utilities[a] for a in bad) < m.incentive_margin - 1e-6:
            raise AssertionError('MILP incentive violation')
        if best < max(m.utility[t]) - 1e-6:
            raise AssertionError('MILP participation violation')
    if np.min(w) < -1e-6 or np.max(w) > m.payment_cap + 1e-6 or result['funding'] > m.budget + 1e-6:
        raise AssertionError('MILP box or funding violation')
    if result['hedge_gap'] > 1e-6:
        raise AssertionError('MILP hedge gap')
    return True


def bootstrap_interval(values, rng, replications=1000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {'n': 0, 'mean': None, 'median': None, 'p05': None, 'p95': None, 'mean_ci95': None}
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(replications)])
    return {'n': int(len(values)), 'mean': float(values.mean()), 'median': float(np.median(values)), 'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95)), 'mean_ci95': [float(v) for v in np.quantile(means, [.025, .975])]}


def run_upgrade_analysis(out, random_rows, samples, seed, milp_cases=80):
    rng = np.random.default_rng(seed + 74621)
    good = [r for r in random_rows if r['status'] == 'optimal']
    savings = np.array([r['benchmark_funding'] - r['funding'] for r in good])
    percentages = np.array([100 * (r['benchmark_funding'] - r['funding']) / r['benchmark_funding'] for r in good if r['benchmark_funding'] > 0])
    premiums = np.array([r['reference_expected_payment'] - r['benchmark_expected_payment'] for r in good])
    groups = {}
    for has_proxy in (False, True):
        subset = [r for r in good if (r['traded_assets'] > 1) == has_proxy]
        groups['audit_security' if has_proxy else 'carbon_only'] = {'count': len(subset), 'funding_saving': bootstrap_interval([r['benchmark_funding']-r['funding'] for r in subset], rng, 400)}
    measures = {'capital_saving_absolute': bootstrap_interval(savings, rng), 'capital_saving_percent': bootstrap_interval(percentages, rng), 'expected_transfer_tradeoff': bootstrap_interval(premiums, rng), 'zero_capital_savings_fraction': float(np.mean(np.abs(savings)<1e-7)) if len(savings) else None, 'feasible_fraction': float(len(good)/len(random_rows)), 'groups': groups, 'sampling_warning': 'Bootstrap intervals describe the synthetic generator and are not empirical confidence intervals for actual carbon contracts.'}
    save_json(out/'objective_statistics.json', measures)
    milp_rows = []
    for i in range(milp_cases):
        actions = 2 + i % 4
        signals = 2 + i % 4
        types = 1 + i % 4
        quality = float(rng.uniform(.2, .9))
        m = random_model(rng, actions=actions, signals=signals, types=types, quality=quality, audit_security=i%3==0)
        reference = witness_solutions(m)
        independent = mixed_integer_solution(m)
        if independent['status'] == 'optimal':
            independent_milp_check(m, independent)
        if independent['status'] == 'unresolved':
            match = None
        else:
            match = reference['status'] == independent['status']
            if match and reference['status'] == 'optimal':
                match = abs(reference['funding'] - independent['funding']) <= 1e-5
        if match is False:
            raise AssertionError(f'Enumeration/MILP disagreement in case {i}: {reference["status"]} vs {independent["status"]}')
        milp_rows.append({'case': i, 'actions': actions, 'signals': signals, 'types': types, 'quality': quality, 'witness_count': reference.get('enumerated', 0), 'enumeration_status': reference['status'], 'milp_status': independent['status'], 'matches': match, 'enumeration_seconds': reference.get('runtime_seconds', 0.), 'milp_seconds': independent.get('runtime_seconds'), 'enumeration_funding': reference.get('funding'), 'milp_funding': independent.get('funding'), 'milp_nodes': independent.get('nodes'), 'milp_gap': independent.get('milp_gap')})
    save_csv(out/'milp_vs_enumeration.csv', milp_rows)
    with (out/'milp_models.jsonl').open('w', encoding='utf-8') as handle:
        rng_models = np.random.default_rng(seed + 74621)
        for i in range(milp_cases):
            actions = 2 + i % 4
            signals = 2 + i % 4
            types = 1 + i % 4
            quality = float(rng_models.uniform(.2, .9))
            m = random_model(rng_models, actions=actions, signals=signals, types=types, quality=quality, audit_security=i%3==0)
            handle.write(json.dumps(model_record(m,f'milp_{i:05d}'),default=to_json)+'\n')
    base = named_model('base')
    robustness = []
    for shift in np.linspace(0, 1, 21):
        for perturbation in ('neutral_noise', 'adversarial_flip'):
            p = base.audit.copy()
            if perturbation == 'neutral_noise':
                p = (1-shift)*p + shift/p.shape[2]
            else:
                p = (1-shift)*p + shift*p[..., ::-1]
            candidate = replace(base, audit=p)
            answer = witness_solutions(candidate)
            robustness.append({'method': perturbation, 'uncertainty_fraction': float(shift), 'status': answer['status'], 'funding': answer.get('funding'), 'witnesses': answer.get('enumerated',0)})
    save_csv(out/'audit_misspecification.csv', robustness)
    fig, ax = plt.subplots(figsize=(8,5))
    for label in ('neutral_noise', 'adversarial_flip'):
        selected = [r for r in robustness if r['method']==label]
        ax.plot([r['uncertainty_fraction'] for r in selected], [r['funding'] if r['funding'] is not None else np.nan for r in selected], marker='o', markersize=3, label=label)
    ax.set(xlabel='Audit distribution perturbation intensity', ylabel='Minimum guaranteed expenditure', title='Audit-law misspecification stress test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'audit_misspecification.png', dpi=200)
    plt.close(fig)
    output = {'samples': samples, 'milp_cross_checks': milp_cases, 'milp_matches': sum(r['matches'] is True for r in milp_rows), 'milp_unresolved': sum(r['matches'] is None for r in milp_rows), 'objective_statistics': measures, 'audit_perturbations': len(robustness)}
    save_json(out/'UPGRADE_MANIFEST.json', output)
    return output


original_main = main


def main():
    if '--samples' not in sys.argv:
        sys.argv.extend(['--samples', '1500'])
    original_main()
    args = argparse.ArgumentParser(add_help=False)
    args.add_argument('--out', default='outputs')
    args.add_argument('--samples', type=int, default=1500)
    args.add_argument('--seed', type=int, default=20261012)
    args.add_argument('--scaling-repeats', type=int, default=4)
    parsed, _ = args.parse_known_args()
    out = Path(parsed.out)
    with (out/'random_cases.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ('funding','benchmark_funding','reference_expected_payment','benchmark_expected_payment'):
            row[field] = float(row[field]) if row[field] else None
        row['traded_assets'] = int(row['traded_assets'])
    results = run_upgrade_analysis(out, rows, parsed.samples, parsed.seed)
    file_hashes = {}
    for filename in ('synthetic_models.jsonl', 'random_cases.csv', 'milp_models.jsonl', 'milp_vs_enumeration.csv'):
        file_hashes[filename] = hashlib.sha256((out/filename).read_bytes()).hexdigest()
    save_json(out/'SHA256.json', file_hashes)
    with (out/'RESULTS_EN.md').open('a', encoding='utf-8') as handle:
        handle.write('\n## Extended research-grade checks\n\n')
        handle.write(f'The upgraded experiment includes {parsed.samples} synthetic instances and {results["milp_cross_checks"]} independent MILP comparisons. Matches: {results["milp_matches"]}; unresolved: {results["milp_unresolved"]}.\n\n')
        handle.write('See objective_statistics.json for bootstrap intervals, milp_vs_enumeration.csv for solver comparisons, audit_misspecification.csv for perturbation studies, and SHA256.json for input/output checksums. All figures are from synthetic experiments, not observed emissions. None of these numerical additions establishes a new mathematical theorem or empirical additionality.\n')
    print(f'UPGRADE: {results["milp_matches"]}/{results["milp_cross_checks"]} MILP cross-checks passed, {results["milp_unresolved"]} unresolved')

if __name__=='__main__':
    main()
