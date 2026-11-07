"""
Goal Planning and Reverse Grade Optimization for AcademIQ.
Solves for the minimal behavioral interventions required to reach a target grade.
"""

from dataclasses import dataclass, asdict
from typing import List, Optional
import copy

from studyflow import config
from studyflow.schemas import StudentInput, PredictionOutput
from studyflow.models.forest import RandomForestStudentPredictor


@dataclass
class GoalPlanResult:
    """
    Structured outcome of reverse-engineering a target grade.
    """
    current_predicted_grade: float
    target_grade: float
    is_achievable: bool
    point_gap: float
    recommended_studytime: int
    recommended_max_absences: int
    recommended_schoolsup: str
    simulated_new_grade: float
    action_steps: List[str]


def plan_target_grade(
    student: StudentInput,
    target_grade: float,
    predictor: Optional[RandomForestStudentPredictor] = None
) -> GoalPlanResult:
    """
    Reverse-engineers the required behavioral changes needed to achieve target_grade.

    Parameters:
    -----------
    student : StudentInput
        Current student input profile.
    target_grade : float
        Target grade on 0.0 - 20.0 scale.
    predictor : Optional[RandomForestStudentPredictor]
        Model predictor instance (if None, initializes a new one).

    Returns:
    --------
    GoalPlanResult
        Actionable roadmap to bridge the academic gap.
    """
    model = predictor or RandomForestStudentPredictor()

    # 1. Baseline prediction
    baseline_output = model.predict_student(student)
    current_pred = baseline_output.predicted_grade

    # Clamp target to domain boundaries
    clamped_target = max(config.MIN_GRADE, min(config.MAX_GRADE, float(target_grade)))
    gap = round(clamped_target - current_pred, 2)

    # If student is already at or above target
    if gap <= 0:
        return GoalPlanResult(
            current_predicted_grade=current_pred,
            target_grade=clamped_target,
            is_achievable=True,
            point_gap=0.0,
            recommended_studytime=student.studytime,
            recommended_max_absences=student.absences,
            recommended_schoolsup=student.schoolsup,
            simulated_new_grade=current_pred,
            action_steps=[
                f"Current trajectory ({current_pred}/20) already meets or exceeds your goal ({clamped_target}/20).",
                "Maintain your current study commitment and attendance consistency through finals."
            ]
        )

    # 2. Search for achievable behavioral levers
    # Levers: studytime (up to 4), absences (down to min 1), schoolsup ("yes")
    best_candidate = copy.deepcopy(student)
    best_simulated_grade = current_pred
    best_studytime = student.studytime
    best_absences = student.absences
    best_schoolsup = student.schoolsup

    # Grid search through realistic student adjustments
    found_solution = False
    for st_level in [student.studytime, min(4, student.studytime + 1), min(4, student.studytime + 2), 4]:
        for abs_level in [student.absences, max(0, student.absences // 2), min(student.absences, 4), 1]:
            for sup in [student.schoolsup, "yes"]:
                candidate = StudentInput(
                    age=student.age,
                    studytime=st_level,
                    failures=student.failures,
                    absences=abs_level,
                    G1=student.G1,
                    G2=student.G2,
                    higher=student.higher,
                    Medu=student.Medu,
                    Fedu=student.Fedu,
                    schoolsup=sup
                )
                sim_res = model.predict_student(candidate)
                if sim_res.predicted_grade >= clamped_target:
                    best_candidate = candidate
                    best_simulated_grade = sim_res.predicted_grade
                    best_studytime = st_level
                    best_absences = abs_level
                    best_schoolsup = sup
                    found_solution = True
                    break
                elif sim_res.predicted_grade > best_simulated_grade:
                    best_candidate = candidate
                    best_simulated_grade = sim_res.predicted_grade
                    best_studytime = st_level
                    best_absences = abs_level
                    best_schoolsup = sup
            if found_solution:
                break
        if found_solution:
            break

    # 3. Formulate human-readable action plan
    actions: List[str] = []
    
    # Study time guidance
    st_labels = {1: "< 2 hrs/week", 2: "2 - 5 hrs/week", 3: "5 - 10 hrs/week", 4: "> 10 hrs/week"}
    if best_studytime > student.studytime:
        actions.append(
            f"Increase weekly study investment from Tier {student.studytime} ({st_labels[student.studytime]}) "
            f"to Tier {best_studytime} ({st_labels[best_studytime]})."
        )
    else:
        actions.append(f"Maintain consistent weekly study investment at Tier {best_studytime} ({st_labels[best_studytime]}).")

    # Absences guidance
    if best_absences < student.absences:
        actions.append(f"Cap remaining absences strictly: reduce from current {student.absences} to maximum {best_absences}.")

    # Educational support
    if best_schoolsup == "yes" and student.schoolsup != "yes":
        actions.append("Enroll in formal school remedial tutoring / extra educational support.")

    if not found_solution:
        actions.append(
            f"Target of {clamped_target}/20 is ambitious given historical term grades. "
            f"Maximum achievable trajectory with optimal habits is {best_simulated_grade}/20."
        )

    return GoalPlanResult(
        current_predicted_grade=current_pred,
        target_grade=clamped_target,
        is_achievable=found_solution,
        point_gap=gap,
        recommended_studytime=best_studytime,
        recommended_max_absences=best_absences,
        recommended_schoolsup=best_schoolsup,
        simulated_new_grade=best_simulated_grade,
        action_steps=actions
    )