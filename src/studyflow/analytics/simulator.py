"""
What-If Counterfactual Simulation and Marginal Gain Analytics for AcademIQ.
Allows students and teachers to simulate intervention scenarios and compute effort ROI.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import copy

from studyflow import config
from studyflow.schemas import StudentInput, PredictionOutput
from studyflow.models.forest import RandomForestStudentPredictor


@dataclass
class SimulationResult:
    """
    Structured outcome of an intervention simulation scenario.
    """
    baseline_grade: float
    simulated_grade: float
    grade_delta: float
    old_risk_level: str
    new_risk_level: str
    risk_improved: bool
    status_changed_to_pass: bool
    applied_changes: Dict[str, Any]


def simulate_intervention(
    student: StudentInput,
    new_studytime: Optional[int] = None,
    new_absences: Optional[int] = None,
    new_schoolsup: Optional[str] = None,
    predictor: Optional[RandomForestStudentPredictor] = None
) -> SimulationResult:
    """
    Simulates a hypothetical student profile with altered habits and compares against baseline.

    Parameters:
    -----------
    student : StudentInput
        The student's original baseline profile.
    new_studytime : Optional[int]
        Target weekly study time tier (1 to 4).
    new_absences : Optional[int]
        Target total school absences (>= 0).
    new_schoolsup : Optional[str]
        Target extra support status ('yes' or 'no').
    predictor : Optional[RandomForestStudentPredictor]
        Trained predictor instance.

    Returns:
    --------
    SimulationResult
        Detailed comparison between current reality and simulated future.
    """
    model = predictor or RandomForestStudentPredictor()

    # 1. Baseline Run
    base_pred = model.predict_student(student)

    # 2. Construct Modified Student
    sim_studytime = new_studytime if new_studytime is not None else student.studytime
    sim_absences = new_absences if new_absences is not None else student.absences
    sim_schoolsup = new_schoolsup if new_schoolsup is not None else student.schoolsup

    # Enforce boundaries
    sim_studytime = max(1, min(4, int(sim_studytime)))
    sim_absences = max(0, int(sim_absences))

    modified_student = StudentInput(
        age=student.age,
        studytime=sim_studytime,
        failures=student.failures,
        absences=sim_absences,
        G1=student.G1,
        G2=student.G2,
        higher=student.higher,
        Medu=student.Medu,
        Fedu=student.Fedu,
        schoolsup=sim_schoolsup
    )

    # 3. Simulated Run
    sim_pred = model.predict_student(modified_student)

    # 4. Compute Deltas & Transitions
    delta = round(sim_pred.predicted_grade - base_pred.predicted_grade, 2)
    risk_order = {"High Risk": 0, "Moderate Risk": 1, "On Track": 2}
    risk_improved = risk_order.get(sim_pred.risk_level, 0) > risk_order.get(base_pred.risk_level, 0)
    status_changed_to_pass = (not base_pred.passed) and sim_pred.passed

    changes = {}
    if sim_studytime != student.studytime:
        changes["studytime"] = {"from": student.studytime, "to": sim_studytime}
    if sim_absences != student.absences:
        changes["absences"] = {"from": student.absences, "to": sim_absences}
    if sim_schoolsup != student.schoolsup:
        changes["schoolsup"] = {"from": student.schoolsup, "to": sim_schoolsup}

    return SimulationResult(
        baseline_grade=base_pred.predicted_grade,
        simulated_grade=sim_pred.predicted_grade,
        grade_delta=delta,
        old_risk_level=base_pred.risk_level,
        new_risk_level=sim_pred.risk_level,
        risk_improved=risk_improved,
        status_changed_to_pass=status_changed_to_pass,
        applied_changes=changes
    )


def compute_marginal_gains(
    student: StudentInput,
    predictor: Optional[RandomForestStudentPredictor] = None
) -> Dict[str, float]:
    """
    Calculates the isolated point value of individual habit improvements.
    Helps students prioritize high-yield decisions.

    Returns:
    --------
    Dict[str, float]
        Point gain for each isolated intervention.
    """
    model = predictor or RandomForestStudentPredictor()
    base_pred = model.predict_student(student).predicted_grade
    gains: Dict[str, float] = {}

    # Yield of +1 Study Time Tier
    if student.studytime < 4:
        sim_st = simulate_intervention(student, new_studytime=student.studytime + 1, predictor=model)
        gains["studytime_plus_one_tier"] = sim_st.grade_delta

    # Yield of 50% Absence Reduction
    if student.absences > 0:
        sim_abs = simulate_intervention(student, new_absences=student.absences // 2, predictor=model)
        gains["reduce_absences_50_pct"] = sim_abs.grade_delta

    # Yield of Enrolling in Extra Educational Support
    if student.schoolsup == "no":
        sim_sup = simulate_intervention(student, new_schoolsup="yes", predictor=model)
        gains["enroll_remedial_support"] = sim_sup.grade_delta

    return gains